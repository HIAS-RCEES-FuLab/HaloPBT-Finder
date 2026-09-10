import os
import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

base_dir = r"Train_data_Cl_Br"
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
cl_num_classes = 8
br_num_classes = 6
name = "multi_cl_br_transformer_0.01_0122"

class MultiModalDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        self.cl_labels = []
        self.br_labels = []
        self.formula = []
        for f in file_list:
            df = pd.read_csv(os.path.join(base_dir, f))
            ion_mode = 1.0 if "POS" in f else 0.0
            for _, row in df.iterrows():
                label_cl = row["Cl_count"]
                label_br = row["Br_count"]
                label_cl = 7 if label_cl > 6 else label_cl
                label_br = 5 if label_br > 4 else label_br

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
                self.cl_labels.append(label_cl)
                self.br_labels.append(label_br)
                self.formula.append(row["Formula"])
        self.formula = np.array(self.formula)
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

dataset = MultiModalDataset(files)
groups = np.array(dataset.formula)
ion_modes = np.array([
    data[4][-1] for data, _ in dataset
]) #POS=1 NEG=0

cl_labels = dataset.cl_labels.numpy()
br_labels = dataset.br_labels.numpy()

combined_labels = cl_labels * 100 + br_labels

def stratified_group_split(indices, labels, groups, test_size=0.1, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X=np.zeros(len(indices)), y=labels[indices], groups=groups[indices]))
    return indices[train_idx], indices[test_idx]

pos_idx = np.where(ion_modes == 1)[0]
pos_trainval_idx, pos_test_idx = stratified_group_split(pos_idx, combined_labels, groups, test_size=0.1)

neg_idx = np.where(ion_modes == 0)[0]
neg_trainval_idx, neg_test_idx = stratified_group_split(neg_idx, combined_labels, groups, test_size=0.1)

trainval_idx = np.concatenate([pos_trainval_idx, neg_trainval_idx])
test_idx = np.concatenate([pos_test_idx, neg_test_idx])

print(f"Train/Val size: {len(trainval_idx)}, Test size: {len(test_idx)}")

groups_trainval = groups[trainval_idx]

cl_trainval = cl_labels[trainval_idx]
br_trainval = br_labels[trainval_idx]
labels_trainval = cl_trainval * 100 + br_trainval

gkf = GroupKFold(n_splits=5)

test_dataset = Subset(dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

all_fold_results = []
all_fold_test_results = []

for fold, (train_sub_idx, val_sub_idx) in enumerate(gkf.split(trainval_idx, labels_trainval, groups=groups_trainval)):
    print(f"\n===== Fold {fold+1} =====")

    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    criterion_cl = nn.CrossEntropyLoss()
    criterion_br = nn.CrossEntropyLoss()

    # 模型
    model = MultiModalNet(precursor_dim, latent_dim,cl_num_classes, br_num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=3, min_lr=1e-6)
    # =========================
    # Early Stopping
    # =========================
    early_stop_patience = 10
    early_stop_counter = 0
    best_val_pr_auc = 0.0
    save_path = f"multimodal_{name}_Cl_Br_fold{fold + 1}.pth"

    for epoch in range(epochs):
        # ---------- Train ----------
        model.train()
        total_train_loss = 0
        for (msms_x, msms_mask, nl_x, nl_mask, precursor_x), (cl_y, br_y) in train_loader:
            msms_x, msms_mask = msms_x.to(device), msms_mask.to(device)
            nl_x, nl_mask = nl_x.to(device), nl_mask.to(device)
            precursor_x = precursor_x.to(device)
            cl_y = cl_y.to(device)
            br_y = br_y.to(device)

            optimizer.zero_grad()
            cl_logits, br_logits = model((msms_x, msms_mask, nl_x, nl_mask, precursor_x))
            loss_cl = criterion_cl(cl_logits, cl_y)
            loss_br = criterion_br(br_logits, br_y)
            loss = loss_cl + loss_br
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * msms_x.size(0)

        train_loss_avg = total_train_loss / len(train_dataset)

        # ---------- Validation ----------
        model.eval()
        val_cl_labels, val_br_labels = [], []
        val_cl_probs, val_br_probs = [], []
        all_val_modes = []
        total_val_loss = 0
        with torch.no_grad():
            for (msms_x, msms_mask, nl_x, nl_mask, precursor_x), (cl_y, br_y) in val_loader:
                msms_x, msms_mask = msms_x.to(device), msms_mask.to(device)
                nl_x, nl_mask = nl_x.to(device), nl_mask.to(device)
                precursor_x = precursor_x.to(device)
                cl_y = cl_y.to(device)
                br_y = br_y.to(device)

                cl_logits, br_logits = model((msms_x, msms_mask, nl_x, nl_mask, precursor_x))
                loss_cl = criterion_cl(cl_logits, cl_y)
                loss_br = criterion_br(br_logits, br_y)
                total_val_loss += (loss_cl + loss_br).item() * msms_x.size(0)

                cl_prob = torch.softmax(cl_logits, 1).cpu().numpy()
                br_prob = torch.softmax(br_logits, 1).cpu().numpy()
                ion_modes = precursor_x[:, -1].cpu().numpy()

                val_cl_labels.extend(cl_y.cpu().numpy())
                val_br_labels.extend(br_y.cpu().numpy())
                val_cl_probs.extend(cl_prob)
                val_br_probs.extend(br_prob)
                all_val_modes.extend(ion_modes)

        val_loss_avg = total_val_loss / len(val_dataset)
        val_cl_labels = np.array(val_cl_labels)
        val_br_labels = np.array(val_br_labels)
        val_cl_probs = np.array(val_cl_probs)
        val_br_probs = np.array(val_br_probs)
        val_cl_preds = val_cl_probs.argmax(axis=1)
        val_br_preds = val_br_probs.argmax(axis=1)
        all_val_modes = np.array(all_val_modes)

        val_cl_acc = accuracy_score(val_cl_labels, val_cl_preds)
        val_br_acc = accuracy_score(val_br_labels, val_br_preds)
        val_cl_pr = average_precision_score(np.eye(cl_num_classes)[val_cl_labels].ravel(), val_cl_probs.ravel())
        val_br_pr = average_precision_score(np.eye(br_num_classes)[val_br_labels].ravel(), val_br_probs.ravel())
        val_pr = (val_cl_pr + val_br_pr) / 2

        pos_mask = all_val_modes == 1
        pos_cl_acc = accuracy_score(val_cl_labels[pos_mask], val_cl_preds[pos_mask])
        pos_br_acc = accuracy_score(val_br_labels[pos_mask], val_br_preds[pos_mask])
        pos_cl_pr = average_precision_score(np.eye(cl_num_classes)[val_cl_labels[pos_mask]].ravel(),
                                            val_cl_probs[pos_mask].ravel())
        pos_br_pr = average_precision_score(np.eye(br_num_classes)[val_br_labels[pos_mask]].ravel(),
                                            val_br_probs[pos_mask].ravel())

        neg_mask = all_val_modes == 0
        neg_cl_acc = accuracy_score(val_cl_labels[neg_mask], val_cl_preds[neg_mask])
        neg_br_acc = accuracy_score(val_br_labels[neg_mask], val_br_preds[neg_mask])
        neg_cl_pr = average_precision_score(np.eye(cl_num_classes)[val_cl_labels[neg_mask]].ravel(),
                                            val_cl_probs[neg_mask].ravel())
        neg_br_pr = average_precision_score(np.eye(br_num_classes)[val_br_labels[neg_mask]].ravel(),
                                            val_br_probs[neg_mask].ravel())

        all_fold_results.append({
            "fold": fold+1, "epoch": epoch+1,
            "train_loss": train_loss_avg, "val_loss": val_loss_avg,
            "val_cl_acc": val_cl_acc, "val_br_acc": val_br_acc, "val_pr_auc": val_pr,
            "pos_cl_acc": pos_cl_acc, "pos_br_acc": pos_br_acc, "pos_pr_auc": (pos_cl_pr + pos_br_pr)/2,
            "neg_cl_acc": neg_cl_acc, "neg_br_acc": neg_br_acc, "neg_pr_auc": (neg_cl_pr + neg_br_pr)/2,
            "lr": optimizer.param_groups[0]['lr']
        })

        print(f"Fold {fold+1} | Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | "
              f"All Cl Acc: {val_cl_acc:.4f} Br Acc: {val_br_acc:.4f} | PR-AUC: {val_pr:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # =========================
        # Save best + Early Stop
        # =========================
        if val_pr > best_val_pr_auc:
            best_val_pr_auc = val_pr
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"💾 Best model saved (Epoch {epoch + 1})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"⏹ Early stopping at Epoch {epoch + 1}")
                break

        # ---------- Scheduler ----------
        scheduler.step(val_pr)

    # =========================
    # Test
    # =========================
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_cl_labels, test_br_labels = [], []
    test_cl_probs, test_br_probs = [], []
    all_test_modes = []

    with torch.no_grad():
        for (msms_x, msms_mask, nl_x, nl_mask, precursor_x), (cl_y, br_y) in test_loader:
            msms_x, msms_mask = msms_x.to(device), msms_mask.to(device)
            nl_x, nl_mask = nl_x.to(device), nl_mask.to(device)
            precursor_x = precursor_x.to(device)
            cl_y = cl_y.to(device)
            br_y = br_y.to(device)

            cl_logits, br_logits = model((msms_x, msms_mask, nl_x, nl_mask, precursor_x))
            test_cl_probs.extend(torch.softmax(cl_logits, 1).cpu().numpy())
            test_br_probs.extend(torch.softmax(br_logits, 1).cpu().numpy())
            test_cl_labels.extend(cl_y.cpu().numpy())
            test_br_labels.extend(br_y.cpu().numpy())
            all_test_modes.extend(precursor_x[:, -1].cpu().numpy())

    test_cl_labels = np.array(test_cl_labels)
    test_br_labels = np.array(test_br_labels)
    test_cl_probs = np.array(test_cl_probs)
    test_br_probs = np.array(test_br_probs)
    test_cl_preds = test_cl_probs.argmax(axis=1)
    test_br_preds = test_br_probs.argmax(axis=1)
    all_test_modes = np.array(all_test_modes)

    # All metrics
    test_cl_acc = accuracy_score(test_cl_labels, test_cl_preds)
    test_br_acc = accuracy_score(test_br_labels, test_br_preds)
    test_cl_pr = average_precision_score(np.eye(cl_num_classes)[test_cl_labels].ravel(), test_cl_probs.ravel())
    test_br_pr = average_precision_score(np.eye(br_num_classes)[test_br_labels].ravel(), test_br_probs.ravel())
    test_pr = (test_cl_pr + test_br_pr) / 2

    # POS-only
    pos_mask = all_test_modes == 1
    pos_cl_acc = accuracy_score(test_cl_labels[pos_mask], test_cl_preds[pos_mask])
    pos_br_acc = accuracy_score(test_br_labels[pos_mask], test_br_preds[pos_mask])
    pos_cl_pr = average_precision_score(np.eye(cl_num_classes)[test_cl_labels[pos_mask]].ravel(),
                                        test_cl_probs[pos_mask].ravel())
    pos_br_pr = average_precision_score(np.eye(br_num_classes)[test_br_labels[pos_mask]].ravel(),
                                        test_br_probs[pos_mask].ravel())

    # NEG-only
    neg_mask = all_test_modes == 0
    neg_cl_acc = accuracy_score(test_cl_labels[neg_mask], test_cl_preds[neg_mask])
    neg_br_acc = accuracy_score(test_br_labels[neg_mask], test_br_preds[neg_mask])
    neg_cl_pr = average_precision_score(np.eye(cl_num_classes)[test_cl_labels[neg_mask]].ravel(),
                                        test_cl_probs[neg_mask].ravel())
    neg_br_pr = average_precision_score(np.eye(br_num_classes)[test_br_labels[neg_mask]].ravel(),
                                        test_br_probs[neg_mask].ravel())

    all_fold_test_results.append({
        "fold": fold+1,
        "test_cl_acc": test_cl_acc, "test_br_acc": test_br_acc, "test_pr_auc": test_pr,
        "pos_cl_acc": pos_cl_acc, "pos_br_acc": pos_br_acc, "pos_pr_auc": (pos_cl_pr+pos_br_pr)/2,
        "neg_cl_acc": neg_cl_acc, "neg_br_acc": neg_br_acc, "neg_pr_auc": (neg_cl_pr+neg_br_pr)/2
    })

    print(f"Fold {fold+1} testing result | All Cl Acc: {test_cl_acc:.4f} Br Acc: {test_br_acc:.4f} | PR-AUC: {test_pr:.4f}")
    print(f"    [POS] Cl Acc: {pos_cl_acc:.4f} Br Acc: {pos_br_acc:.4f} | PR-AUC: {(pos_cl_pr+pos_br_pr)/2:.4f}")
    print(f"    [NEG] Cl Acc: {neg_cl_acc:.4f} Br Acc: {neg_br_acc:.4f} | PR-AUC: {(neg_cl_pr+neg_br_pr)/2:.4f}")

df_fold = pd.DataFrame(all_fold_results)
df_fold.to_csv(f"Halogen_trainval_results.csv", index=False)

# 保存每 fold 测试指标
df_test = pd.DataFrame(all_fold_test_results)
df_test.to_csv(f"Halogen_test_summary.csv", index=False)
