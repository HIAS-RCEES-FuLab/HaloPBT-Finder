import os
import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

# -----------------------------
# Parameter
# -----------------------------
test_dir = r"Test_data"
test_files = [f for f in os.listdir(test_dir) if f.endswith(".csv") and ("POS" in f or "NEG" in f)]
print(f"Training on files: {test_files}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PEAKS = 50
MAX_LOSSES = 50
MAX_MZ = 1000.0
MAX_NL_MASS = 500.0
precursor_dim = 3 + 7 + 1   # exact_mass + KMD_Cl + KMD_Br + isotope(M0–M6) + ion_mode
latent_dim = 128
epochs = 50
mode = "all"
name = "multi_PBT_transformer"

# -----------------------------
# Dataset
# -----------------------------
class MultiModalDataset(Dataset):
    def __init__(self, file_list, mode="all"):
        self.data = []
        self.labels = []
        self.smiles = []
        for f in file_list:
            df = pd.read_csv(os.path.join(test_dir, f))
            ion_mode = 1.0 if "POS" in f else 0.0
            for _, row in df.iterrows():
                if mode == "table2_0":
                    if "Table2_PBT" not in df.columns or row["Table2_PBT"] != 0:
                        continue
                elif mode == "pbt1_table2_1":
                    if row["PBT_label"] == 1:
                        if "Table2_PBT" not in df.columns or row["Table2_PBT"] != 1:
                            continue
                # label
                if row["PBT_label"] not in [0, 1]:
                    continue
                label = int(row["PBT_label"])

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
                self.labels.append(label)
                self.smiles.append(row["Canonical_smiles"])

        self.smiles = np.array(self.smiles)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
        print(f"Total samples: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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
        # 位置编码 D维度元素叠加
        x = x + self.pos_embed[:, :K+1]
        cls_mask = torch.ones(B, 1, device=x.device, dtype=torch.bool)
        key_mask = torch.cat([cls_mask, mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=~key_mask)
        x = self.norm(x[:, 0])                  # CLS token
        return x                                # [B, latent_dim]

class MultiModalNet(nn.Module):
    def __init__(self, precursor_dim, latent_dim, max_peaks=50):
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
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        msms_x, msms_mask, nl_x, nl_mask, precursor_x = x
        msms_latent = self.msms_net(msms_x, msms_mask)
        nl_feat = self.nl_proj(nl_x)  # [B, L, 32]
        nl_mask = nl_mask.unsqueeze(-1)  # [B, L, 1]
        nl_feat = nl_feat * nl_mask  # mask
        nl_latent = self.nl_net(nl_feat.sum(1) / nl_mask.sum(1).clamp(min=1))
        precursor_latent = self.precursor_net(precursor_x)
        return self.fusion(
            torch.cat([msms_latent, nl_latent, precursor_latent], dim=1)
        )

# -----------------------------
# testing function
# -----------------------------
def test_model(model_path, test_loader,fold):
    print(f"\nLoading model: {model_path}")

    # load models
    model = MultiModalNet(precursor_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_labels, test_probs, test_modes = [], [], []
    with torch.no_grad():
        for (msms_x, msms_mask, nl_x, nl_mask, precursor_x), y in test_loader:
            msms_x, msms_mask = msms_x.to(device), msms_mask.to(device)
            nl_x, nl_mask = nl_x.to(device), nl_mask.to(device)
            precursor_x, y = precursor_x.to(device), y.to(device)
            logits = model((msms_x, msms_mask, nl_x, nl_mask, precursor_x))
            test_labels.append(y.cpu().numpy())
            test_probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            test_modes.append(precursor_x[:, -1].cpu().numpy())
    test_labels = np.concatenate(test_labels, axis=0)
    test_probs = np.concatenate(test_probs, axis=0)
    test_preds = (test_probs > 0.5).astype(int)

    # metric
    test_acc_all = accuracy_score(test_labels, test_preds)
    test_pr_all = average_precision_score(test_labels, test_probs)
    test_roc_all = roc_auc_score(test_labels, test_probs)

    # count
    total_samples = len(test_labels)

    return {
        "fold": fold,
        "test_acc_all": test_acc_all,
        "test_pr_all": test_pr_all,
        "test_roc_all": test_roc_all,
        "total_samples": total_samples,
    }


# -----------------------------
# main
# -----------------------------
def main():
    test_dataset = MultiModalDataset(test_files, mode)

    labels = test_dataset.labels.numpy()
    print(f"\nDataset mode: {mode}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Class distribution: {np.bincount(labels)}")

    test_ion_modes = np.array([data[4][-1] for data, _ in test_dataset])
    pos_test_idx = np.where(test_ion_modes == 1)[0]
    neg_test_idx = np.where(test_ion_modes == 0)[0]

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"\nTest set size: {len(test_dataset)}")
    print(f"POS test samples: {len(pos_test_idx)}")
    print(f"NEG test samples: {len(neg_test_idx)}")

    # 5 fold models
    all_test_results = []

    for fold in range(1, 6):
        model_path = f"multimodal_{name}_{mode}_fold{fold}.pth"

        if os.path.exists(model_path):
            results = test_model(model_path, test_loader, fold)
            all_test_results.append(results)

            print(f"\nFold {fold} Test Results:")
            print(f"  Accuracy: {results['test_acc_all']:.4f}")
            print(f"  PR-AUC:   {results['test_pr_all']:.4f}")
            print(f"  ROC-AUC:  {results['test_roc_all']:.4f}")
        else:
            print(f"Warning: Model file {model_path} not found")

    # =========================
    # Average ± Std across 5 folds
    # =========================
    if all_test_results:
        print("\n" + "=" * 60)
        print("Average Test Results Across 5 Folds:")
        print("=" * 60)
        print(f"{'Metric':<15} {'Mean':<10} {'Std':<10}")
        print("-" * 35)

        metrics = [
            ('test_acc_all', 'Accuracy'),
            ('test_pr_all', 'PR-AUC'),
            ('test_roc_all', 'ROC-AUC')
        ]

        for metric_key, metric_name in metrics:
            values = np.array([r[metric_key] for r in all_test_results])
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name:<15} {mean_val:.4f}     {std_val:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()