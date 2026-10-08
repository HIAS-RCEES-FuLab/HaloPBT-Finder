import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
import os

# -----------------------------
# parameter
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PEAKS = 50
MAX_LOSSES = 50
MAX_MZ = 1000.0
MAX_NL_MASS = 500.0
precursor_dim = 3 + 7 + 1   # exact_mass + KMD_Cl + KMD_Br + isotope(M0–M6) + ion_mode
latent_dim = 128
epochs = 50
cl_num_classes = 8
br_num_classes = 6
name = "multi_cl_br_transformer"

# -----------------------------
# Dataset
# -----------------------------
class MultiModalDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        self.cl_labels = []
        self.br_labels = []
        self.formula = []
        self.source_file = []
        self.source_index = []
        self.raw_data = []
        for f in file_list:
            df = pd.read_csv(f)
            ion_mode = 1.0 if "POS" in f else 0.0
            for row_idx, row in df.iterrows():
                label_cl = row["Cl_count"]
                label_br = row["Br_count"]
                label_cl = 7 if label_cl > 6 else label_cl
                label_br = 5 if label_br > 4 else label_br

                # ---------- MS/MS ----------
                peak_list = []
                if pd.notna(row["Mass_spectral_features"]):
                    feats = ast.literal_eval(row["Mass_spectral_features"])
                    for _, mz, inten, _, _ in feats:
                        if inten and inten > 0:
                            peak_list.append((mz, inten))
                if len(peak_list) == 0:
                    continue
                peak_list = peak_list[:MAX_PEAKS]
                max_inten = max(p[1] for p in peak_list)
                msms_x = np.zeros((MAX_PEAKS, 2), dtype=np.float32)
                msms_mask = np.zeros(MAX_PEAKS, dtype=bool)
                for i, (mz, inten) in enumerate(peak_list):
                    msms_x[i] = [mz / MAX_MZ, inten / max_inten]
                    msms_mask[i] = True

                # ---------- Neutral loss (mz-only) ----------
                nl_x = np.zeros((MAX_LOSSES, 1), dtype=np.float32)
                nl_mask = np.zeros(MAX_LOSSES, dtype=bool)
                if pd.notna(row["Neutral_losses_binned"]):
                    nl_list = ast.literal_eval(row["Neutral_losses_binned"])
                    for i, (_, mz) in enumerate(nl_list[:MAX_LOSSES]):
                        nl_x[i, 0] = mz / MAX_NL_MASS
                        nl_mask[i] = True

                # ---------- Precursor ----------
                isotope_pattern = ast.literal_eval(row["isotope_pattern_M0_M6"])
                assert len(isotope_pattern) == 7
                precursor_x = np.array(
                    [row["Exact_mass"] / MAX_MZ,
                     row["KMD_Cl"],
                     row["KMD_Br"]] +
                    isotope_pattern +
                    [ion_mode],
                    dtype=np.float32
                )
                self.data.append(
                    (msms_x, msms_mask, nl_x, nl_mask, precursor_x)
                )
                self.cl_labels.append(label_cl)
                self.br_labels.append(label_br)
                self.formula.append(row["Formula"])
                self.raw_data.append(row.to_dict())
                self.source_file.append(f)
                self.source_index.append(row_idx)
        self.formula = np.array(self.formula)
        self.source_file = np.array(self.source_file)
        self.source_index = np.array(self.source_index)
        self.cl_labels = torch.tensor(np.array(self.cl_labels), dtype=torch.long)
        self.br_labels = torch.tensor(np.array(self.br_labels), dtype=torch.long)
        print(f"Total samples: {len(self.cl_labels)}")

    def __len__(self):
        return len(self.cl_labels)

    def __getitem__(self, idx):
        return self.data[idx], (self.cl_labels[idx], self.br_labels[idx])

class SubNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(d, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2)
            ]
            d = h
        layers += [nn.Linear(d, latent_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# -----------------------------
# transformer
# -----------------------------
class MSMSTransformer(nn.Module):
    def __init__(
        self,
        peak_dim=2,        # [mz_norm, rel_inten]
        latent_dim=128,
        nhead=4,
        num_layers=3,
        max_peaks=50
    ):
        super().__init__()
        self.proj = nn.Linear(peak_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # CLS token + position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_peaks + 1, latent_dim)
        )
        self.norm = nn.LayerNorm(latent_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, mask):
        """
        x: [B, K, 2]
        """
        B, K, _ = x.shape
        x = self.proj(x)                  # [B, K, D]
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)          # [B, K+1, D]
        x = x + self.pos_embed[:, :K+1]
        cls_mask = torch.ones(B, 1, device=x.device, dtype=torch.bool)
        key_mask = torch.cat([cls_mask, mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=~key_mask)
        # pooling
        x = self.norm(x[:, 0])                  # CLS token
        return x                                # [B, latent_dim]

class MultiModalNet(nn.Module):
    def __init__(self, precursor_dim, latent_dim, cl_num_classes, br_num_classes, max_peaks=50):
        super().__init__()
        self.msms_net = MSMSTransformer(2, latent_dim, max_peaks=max_peaks)
        # Neutral loss（mz-only → pooling → MLP）
        self.nl_proj = nn.Linear(1, 32)
        self.nl_net = SubNet(
            input_dim=32,
            latent_dim=latent_dim,
            hidden_dims=[128, 64]
        )
        self.precursor_net = SubNet(precursor_dim, latent_dim, hidden_dims=[32, 64])
        self.cl_head = nn.Sequential(
            nn.Linear(latent_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, cl_num_classes)
        )

        self.br_head = nn.Sequential(
            nn.Linear(latent_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, br_num_classes)
        )

    def forward(self, x):
        msms_x, msms_mask, nl_x, nl_mask, precursor_x = x
        msms_latent = self.msms_net(msms_x, msms_mask)
        nl_feat = self.nl_proj(nl_x)  # [B, L, 32]
        nl_mask = nl_mask.unsqueeze(-1)  # [B, L, 1]
        nl_feat = nl_feat * nl_mask  # mask
        nl_latent = self.nl_net(nl_feat.sum(1) / nl_mask.sum(1).clamp(min=1))
        precursor_latent = self.precursor_net(precursor_x)
        fused = torch.cat([msms_latent, nl_latent, precursor_latent], dim=1)
        cl_logits = self.cl_head(fused)
        br_logits = self.br_head(fused)
        return cl_logits, br_logits

def test_model(model_path, test_loader, fold):
    model = MultiModalNet(precursor_dim, latent_dim, cl_num_classes, br_num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cl_labels, br_labels, cl_probs, br_probs = [], [], [], []

    with torch.no_grad():
        for (msms_x, msms_mask, nl_x, nl_mask, precursor_x), (cl_y, br_y) in test_loader:
            msms_x, msms_mask = msms_x.to(device), msms_mask.to(device)
            nl_x, nl_mask = nl_x.to(device), nl_mask.to(device)
            precursor_x = precursor_x.to(device)
            cl_logits, br_logits = model((msms_x, msms_mask, nl_x, nl_mask, precursor_x))
            cl_probs.extend(torch.softmax(cl_logits, dim=1).cpu().numpy())
            br_probs.extend(torch.softmax(br_logits, dim=1).cpu().numpy())
            cl_labels.extend(cl_y.numpy())
            br_labels.extend(br_y.numpy())

    cl_labels, br_labels = np.array(cl_labels), np.array(br_labels)
    cl_probs, br_probs = np.array(cl_probs), np.array(br_probs)
    cl_preds, br_preds = cl_probs.argmax(axis=1), br_probs.argmax(axis=1)

    def multiclass_metrics(y_true, y_prob):
        n_classes = y_prob.shape[1]
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        pr_aucs, roc_aucs = [], []

        for i in range(n_classes):
            if 0 < y_bin[:, i].sum() < len(y_bin):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                pr_aucs.append(auc(recall, precision))

                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_aucs.append(auc(fpr, tpr))

        return (
            accuracy_score(y_true, y_prob.argmax(axis=1)),
            np.mean(pr_aucs),
            np.mean(roc_aucs)
        )

    cl_acc, cl_pr_auc, cl_roc_auc = multiclass_metrics(cl_labels, cl_probs)
    br_acc, br_pr_auc, br_roc_auc = multiclass_metrics(br_labels, br_probs)

    overall_acc = np.mean((cl_preds == cl_labels) & (br_preds == br_labels))

    overall_labels = np.concatenate([
        label_binarize(cl_labels, classes=np.arange(cl_num_classes)),
        label_binarize(br_labels, classes=np.arange(br_num_classes))
    ], axis=1)
    overall_probs = np.concatenate([cl_probs, br_probs], axis=1)

    precision, recall, _ = precision_recall_curve(overall_labels.ravel(), overall_probs.ravel())
    fpr, tpr, _ = roc_curve(overall_labels.ravel(), overall_probs.ravel())
    combined_pr_auc = auc(recall, precision)
    combined_roc_auc = auc(fpr, tpr)

    print(f"Fold {fold}: Cl Acc={cl_acc:.4f}, PR-AUC={cl_pr_auc:.4f}, ROC-AUC={cl_roc_auc:.4f}")
    print(f"Fold {fold}: Br Acc={br_acc:.4f}, PR-AUC={br_pr_auc:.4f}, ROC-AUC={br_roc_auc:.4f}")
    print(f"Fold {fold}: Overall Acc={overall_acc:.4f}, PR-AUC={combined_pr_auc:.4f}, ROC-AUC={combined_roc_auc:.4f}")

    return {
        "fold": fold,
        "cl_acc": cl_acc,
        "cl_pr_auc": cl_pr_auc,
        "cl_roc_auc": cl_roc_auc,
        "br_acc": br_acc,
        "br_pr_auc": br_pr_auc,
        "br_roc_auc": br_roc_auc,
        "overall_acc": overall_acc,
        "combined_pr_auc": combined_pr_auc,
        "combined_roc_auc": combined_roc_auc
    }


def main():
    # =============================
    # Load fixed test data
    # =============================
    test_dir = r"Test_data_Cl_Br"
    pos_file = os.path.join(test_dir, "Cl_Br_test_POS.csv")
    neg_file = os.path.join(test_dir, "Cl_Br_test_NEG.csv")

    pos_dataset = MultiModalDataset([pos_file])
    neg_dataset = MultiModalDataset([neg_file])

    pos_test_idx = np.arange(len(pos_dataset))
    neg_test_idx = np.arange(len(neg_dataset))

    # 合并测试集
    test_files = [pos_file, neg_file]
    test_dataset = MultiModalDataset(test_files)

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    print(f"\nTest set size: {len(test_dataset)}")
    print(f"POS test samples: {len(pos_test_idx)}")
    print(f"NEG test samples: {len(neg_test_idx)}")

    all_test_results = []

    for fold in range(1, 6):
        model_path = f"multimodal_{name}_fold{fold}.pth"
        if os.path.exists(model_path):
            all_test_results.append(test_model(model_path, test_loader, fold))
        else:
            print(f"Warning: {model_path} not found")

    if not all_test_results:
        return

    metrics = [
        ("Cl Accuracy", "cl_acc"),
        ("Cl PR-AUC", "cl_pr_auc"),
        ("Cl ROC-AUC", "cl_roc_auc"),
        ("Br Accuracy", "br_acc"),
        ("Br PR-AUC", "br_pr_auc"),
        ("Br ROC-AUC", "br_roc_auc"),
        ("Overall Accuracy", "overall_acc"),
        ("Combined PR-AUC", "combined_pr_auc"),
        ("Combined ROC-AUC", "combined_roc_auc")
    ]

    print(f"\n{'=' * 70}")
    print("5-Fold Test Results: Mean ± SD")
    print(f"{'=' * 70}")

    summary = []
    for result in all_test_results:
        summary.append(result)

    for display_name, key in metrics:
        values = np.array([r[key] for r in all_test_results])
        print(f"{display_name:<25} {values.mean():.4f} ± {values.std():.4f}")

if __name__ == "__main__":
    main()