import os
import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

base_dir = r"Train_data"
files = [
    f for f in os.listdir(base_dir)
    if f.endswith(".csv") and ("POS" in f or "NEG" in f)
]
print(f"Training on files: {files}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_PEAKS = 50
MAX_LOSSES = 50
MAX_MZ = 1000.0
MAX_NL_MASS = 500.0

precursor_dim = 3 + 7 + 1
latent_dim = 128
epochs = 50
mode = "all"
name = "multi_PBT_transformer_0.01_0119"

class MultiModalDataset(Dataset):
    def __init__(self, file_list, mode="all"):

        self.data = []
        self.labels = []
        self.smiles = []

        for f in file_list:
            df = pd.read_csv(os.path.join(base_dir, f))
            ion_mode = 1.0 if "POS" in f else 0.0
            for _, row in df.iterrows():

                if mode == "table2_0":
                    if "Table2_PBT" not in df.columns or row["Table2_PBT"] != 0:
                        continue
                elif mode == "pbt1_table2_1":
                    if row["PBT_label"] == 1:
                        if "Table2_PBT" not in df.columns or row["Table2_PBT"] != 1:
                            continue

                if row["PBT_label"] not in [0, 1]:
                    continue
                label = int(row["PBT_label"])

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

                nl_x = np.zeros((MAX_LOSSES, 1), dtype=np.float32)
                nl_mask = np.zeros(MAX_LOSSES, dtype=bool)
                if pd.notna(row["Neutral_losses_binned"]):
                    nl_list = ast.literal_eval(row["Neutral_losses_binned"])
                    for i, (_, mz) in enumerate(nl_list[:MAX_LOSSES]):
                        nl_x[i, 0] = mz / MAX_NL_MASS
                        nl_mask[i] = True

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
    def __init__(self, precursor_dim, latent_dim, max_peaks=50):
        super().__init__()
        self.msms_net = MSMSTransformer(2, latent_dim, max_peaks=max_peaks)
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

dataset = MultiModalDataset(files, mode)
labels = dataset.labels.numpy()
groups = dataset.smiles
# ion mode: POS=1 / NEG=0
ion_modes = np.array([data[4][-1] for data, _ in dataset])

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.1,
    random_state=42
)

# ---------- POS ----------
pos_idx = np.where(ion_modes == 1)[0]

pos_trainval_idx, pos_test_idx = next(
    gss.split(
        X=np.zeros(len(pos_idx)),
        groups=groups[pos_idx]
    )
)

pos_trainval_idx = pos_idx[pos_trainval_idx]
pos_test_idx = pos_idx[pos_test_idx]

# ---------- NEG ----------
neg_idx = np.where(ion_modes == 0)[0]

neg_trainval_idx, neg_test_idx = next(
    gss.split(
        X=np.zeros(len(neg_idx)),
        groups=groups[neg_idx]
    )
)

neg_trainval_idx = neg_idx[neg_trainval_idx]
neg_test_idx = neg_idx[neg_test_idx]

# -----------------------------
# POS + NEG
# -----------------------------
trainval_idx = np.concatenate([pos_trainval_idx, neg_trainval_idx])
test_idx = np.concatenate([pos_test_idx, neg_test_idx])

print(f"Train/Val size: {len(trainval_idx)}")
print(f"Test size: {len(test_idx)}")

labels_trainval = labels[trainval_idx]
groups_trainval = groups[trainval_idx]

gkf = GroupKFold(n_splits=5)

test_dataset = Subset(dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_fold_results = []
all_fold_test_results = []

for fold, (train_sub_idx, val_sub_idx) in enumerate(
    gkf.split(trainval_idx, labels_trainval, groups=groups_trainval)
):
    #if fold == 0:  # fold1
    #    continue
    print(f"\n===== Fold {fold+1} =====")

    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")  # <-- 新增

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    n_samples = len(train_dataset)
    n_positive = sum(dataset.labels[train_idx] == 1).item()
    n_negative = sum(dataset.labels[train_idx] == 0).item()
    weight = torch.tensor([n_samples / n_negative, n_samples / n_positive], dtype=torch.float32)
    weight = torch.clamp(weight, max=4.0).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    model = MultiModalNet(precursor_dim, latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4  # L2
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # Early Stopping
    early_stop_patience = 10
    early_stop_counter = 0
    best_val_pr_auc = 0.0
    save_path = f"multimodal_{name}_{mode}_fold{fold+1}.pth"

    # ----------------- trianing -----------------
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_labels, train_probs = [], []

        for (msms_x, msms_mask, nl_x, nl_mask, precursor_x), y in train_loader:
            msms_x, msms_mask = msms_x.to(device), msms_mask.to(device)
            nl_x, nl_mask = nl_x.to(device), nl_mask.to(device)
            precursor_x, y = precursor_x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model((msms_x, msms_mask, nl_x, nl_mask, precursor_x))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            train_probs.append(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy())
            train_labels.append(y.cpu().numpy())

        train_labels = np.concatenate(train_labels, axis=0)
        train_probs = np.concatenate(train_probs, axis=0)
        train_preds = (train_probs > 0.5).astype(int)
        train_loss /= len(train_dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        train_pr = average_precision_score(train_labels, train_probs)
        train_roc = roc_auc_score(train_labels, train_probs)

        # 验证
        model.eval()
        val_labels, val_probs, val_modes = [], [], []
        val_loss = 0.0

        with torch.no_grad():
            for (msms_x, msms_mask, nl_x, nl_mask, precursor_x), y_val in val_loader:
                msms_x, msms_mask = msms_x.to(device), msms_mask.to(device)
                nl_x, nl_mask = nl_x.to(device), nl_mask.to(device)
                precursor_x, y_val = precursor_x.to(device), y_val.to(device)

                logits = model((msms_x, msms_mask, nl_x, nl_mask, precursor_x))
                loss = criterion(logits, y_val)
                val_loss += loss.item() * y_val.size(0)

                val_labels.append(y_val.cpu().numpy())
                val_probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                val_modes.append(precursor_x[:, -1].cpu().numpy())

        val_labels = np.concatenate(val_labels, axis=0)
        val_probs = np.concatenate(val_probs, axis=0)
        val_modes = np.concatenate(val_modes, axis=0)
        val_preds = (val_probs > 0.5).astype(int)
        val_loss /= len(val_dataset)

        val_acc = accuracy_score(val_labels, val_preds)
        val_pr_all = average_precision_score(val_labels, val_probs)
        val_roc_all = roc_auc_score(val_labels, val_probs)

        # POS
        pos_mask = val_modes == 1
        val_acc_pos = accuracy_score(val_labels[pos_mask], val_preds[pos_mask])
        val_pr_pos = average_precision_score(val_labels[pos_mask], val_probs[pos_mask])
        val_roc_pos = roc_auc_score(val_labels[pos_mask], val_probs[pos_mask])

        # NEG
        neg_mask = val_modes == 0
        val_acc_neg = accuracy_score(val_labels[neg_mask], val_preds[neg_mask])
        val_pr_neg = average_precision_score(val_labels[neg_mask], val_probs[neg_mask])
        val_roc_neg = roc_auc_score(val_labels[neg_mask], val_probs[neg_mask])

        all_fold_results.append({
            "fold": fold + 1, "epoch": epoch + 1,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "train_pr": train_pr, "train_roc": train_roc,
            "val_acc_all": val_acc, "val_pr_all": val_pr_all, "val_roc_all": val_roc_all,
            "val_acc_pos": val_acc_pos, "val_pr_pos": val_pr_pos, "val_roc_pos": val_roc_pos,
            "val_acc_neg": val_acc_neg, "val_pr_neg": val_pr_neg, "val_roc_neg": val_roc_neg,
            "lr": optimizer.param_groups[0]['lr']
        })

        print(f"Fold {fold + 1} | Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} PR: {train_pr:.4f} ROC: {train_roc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} PR: {val_pr_all:.4f} ROC: {val_roc_all:.4f}")

        # Early stop + save best
        if val_pr_all > best_val_pr_auc:
            best_val_pr_auc = val_pr_all
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"💾 Best model saved (Epoch {epoch + 1})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"⏹ Early stopping at epoch {epoch + 1}")
                break

        scheduler.step(val_pr_all)

    # --------- Test ---------
    model.load_state_dict(torch.load(save_path))
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
    test_modes = np.concatenate(test_modes, axis=0)
    test_preds = (test_probs > 0.5).astype(int)

    # Test指标
    test_acc_all = accuracy_score(test_labels, test_preds)
    test_pr_all = average_precision_score(test_labels, test_probs)
    test_roc_all = roc_auc_score(test_labels, test_probs)

    # POS
    pos_mask = test_modes == 1
    test_acc_pos = accuracy_score(test_labels[pos_mask], test_preds[pos_mask])
    test_pr_pos = average_precision_score(test_labels[pos_mask], test_probs[pos_mask])
    test_roc_pos = roc_auc_score(test_labels[pos_mask], test_probs[pos_mask])

    # NEG
    neg_mask = test_modes == 0
    test_acc_neg = accuracy_score(test_labels[neg_mask], test_preds[neg_mask])
    test_pr_neg = average_precision_score(test_labels[neg_mask], test_probs[neg_mask])
    test_roc_neg = roc_auc_score(test_labels[neg_mask], test_probs[neg_mask])

    all_fold_test_results.append({
        "fold": fold + 1,
        "test_acc_all": test_acc_all, "test_pr_all": test_pr_all, "test_roc_all": test_roc_all,
        "test_acc_pos": test_acc_pos, "test_pr_pos": test_pr_pos, "test_roc_pos": test_roc_pos,
        "test_acc_neg": test_acc_neg, "test_pr_neg": test_pr_neg, "test_roc_neg": test_roc_neg
    })

    print(f"Fold {fold + 1} TEST | Acc All: {test_acc_all:.4f} PR: {test_pr_all:.4f} ROC: {test_roc_all:.4f} "
          f"POS PR: {test_pr_pos:.4f} ROC: {test_roc_pos:.4f} NEG PR: {test_pr_neg:.4f} ROC: {test_roc_neg:.4f}")

df_results = pd.DataFrame(all_fold_results)
df_results.to_csv(f"PBT_trainval_results.csv", index=False)

df_test_results = pd.DataFrame(all_fold_test_results)
df_test_results.to_csv(f"PBT_test_summary.csv", index=False)
