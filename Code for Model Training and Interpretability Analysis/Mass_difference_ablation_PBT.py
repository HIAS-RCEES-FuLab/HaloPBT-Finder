import os
import pandas as pd
import ast
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import shap
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

base_dir = r"Train_data"
files = [
    f for f in os.listdir(base_dir)
    if f.endswith(".csv") and ("POS" in f or "NEG" in f)
]
print(f"Using files: {files}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_PEAKS = 50
MAX_LOSSES = 50
MAX_MZ = 1000.0
MAX_NL_MASS = 500.0

precursor_dim = 4
latent_dim = 128

class MultiModalDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        self.labels = []
        self.smiles = []

        for f in file_list:
            df = pd.read_csv(os.path.join(base_dir, f))
            ion_mode = 1.0 if "POS" in f else 0.0

            for _, row in df.iterrows():
                if row["PBT_label"] not in [0, 1]:
                    continue
                label = int(row["PBT_label"])

                # MS/MS
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

                # Neutral Loss
                nl_x = np.zeros((MAX_LOSSES, 1), dtype=np.float32)
                nl_mask = np.zeros(MAX_LOSSES, dtype=bool)
                if pd.notna(row["Neutral_losses_binned"]):
                    nl_list = ast.literal_eval(row["Neutral_losses_binned"])
                    for i, (_, mz) in enumerate(nl_list[:MAX_LOSSES]):
                        nl_x[i, 0] = mz / MAX_NL_MASS
                        nl_mask[i] = True

                # Precursor
                precursor_x = np.array(
                    [
                        row["Exact_mass"] / MAX_MZ,
                        row["KMD_Cl"],
                        row["KMD_Br"],
                        ion_mode,
                    ],
                    dtype=np.float32,
                )

                self.data.append((
                    torch.tensor(msms_x),
                    torch.tensor(msms_mask, dtype=torch.bool),
                    torch.tensor(nl_x),
                    torch.tensor(nl_mask, dtype=torch.bool),
                    torch.tensor(precursor_x),
                ))
                self.labels.append(label)
                self.smiles.append(row["Canonical_smiles"])

        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.smiles = np.array(self.smiles)
        print(f"Total samples: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# -----------------------------
# SubNet
# -----------------------------
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
                nn.Dropout(0.2),
            ]
            d = h
        layers += [nn.Linear(d, latent_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------
# MS/MS Transformer
# -----------------------------
class MSMSTransformer(nn.Module):
    def __init__(self, peak_dim=2, latent_dim=128, nhead=4, num_layers=3, max_peaks=50):
        super().__init__()
        self.proj = nn.Linear(peak_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_peaks + 1, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, mask):
        B, K, _ = x.shape
        x = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : K + 1]
        cls_mask = torch.ones(B, 1, device=x.device, dtype=torch.bool)
        key_mask = torch.cat([cls_mask, mask.to(x.device)], dim=1)
        x = self.encoder(x, src_key_padding_mask=~key_mask)
        return self.norm(x[:, 0])

# -----------------------------
# MultiModalNet
# -----------------------------
class MultiModalNet(nn.Module):
    def __init__(self, precursor_dim, latent_dim):
        super().__init__()
        self.msms_net = MSMSTransformer(latent_dim=latent_dim)
        self.nl_proj = nn.Linear(1, 32)
        self.nl_net = SubNet(32, latent_dim, [128, 64])
        self.precursor_net = SubNet(precursor_dim, latent_dim, [32, 64])
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        msms_x, msms_mask, nl_x, nl_mask, precursor_x = x
        msms_latent = self.msms_net(msms_x, msms_mask)
        nl_feat = self.nl_proj(nl_x) * nl_mask.unsqueeze(-1)
        nl_latent = self.nl_net(nl_feat.sum(1) / nl_mask.sum(1).clamp(min=1).unsqueeze(-1))
        precursor_latent = self.precursor_net(precursor_x)
        return self.fusion(torch.cat([msms_latent, nl_latent, precursor_latent], dim=1))

dataset = MultiModalDataset(files)
labels = dataset.labels.numpy()
groups = dataset.smiles
ion_modes = np.array([d[4][-1].item() for d, _ in dataset])

full_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False, 
    num_workers=0
)
"""
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
pos_idx = np.where(ion_modes == 1)[0]
neg_idx = np.where(ion_modes == 0)[0]
pos_trainval, pos_test = next(gss.split(pos_idx, groups=groups[pos_idx]))
neg_trainval, neg_test = next(gss.split(neg_idx, groups=groups[neg_idx]))
trainval_idx = np.concatenate([pos_idx[pos_trainval], neg_idx[neg_trainval]])
test_idx = np.concatenate([pos_idx[pos_test], neg_idx[neg_test]])

test_dataset = Subset(dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
"""

model = MultiModalNet(precursor_dim, latent_dim).to(device)
fold = 1
model_path = f"multimodal_multi_PBT_transformer_change_0.01_0201_all_fold{fold}.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

threshold = 0.5

high_conf_indices = []
all_probs = []
all_preds = []
all_labels = []

counter = 0

with torch.no_grad():
    for x, y in full_loader:
        x = [i.to(device) for i in x]
        y = y.to(device)

        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = (probs >= threshold).long()

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

        for i, (pred, t) in enumerate(zip(preds, y)):
            #if t.item() == 1 and pred.item() == 1:
            high_conf_indices.append(counter + i)

        counter += len(y)

all_probs = np.concatenate(all_probs)
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
acc = accuracy_score(all_labels, all_preds)
print(f"Overall Accuracy (full dataset): {acc:.4f}")
print(f"Total samples: {len(all_preds)}")
print(f"High-confidence PBT count: {len(high_conf_indices)}")

@torch.no_grad()
def neutral_loss_ablation(
    model,
    msms_x,          # (K, 2)
    nl_x,            # (N, 1)
    precursor_x,     # (4,)
):
    N = nl_x.shape[0]
    valid = nl_x[:, 0] > 0

    msms_x = msms_x.unsqueeze(0).to(device)
    nl_x = nl_x.unsqueeze(0).to(device)
    precursor_x = precursor_x.unsqueeze(0).to(device)

    msms_mask = (msms_x[:, :, 1] > 0).to(device)
    nl_mask = valid.unsqueeze(0).to(device)

    # baseline
    base_logit = model([
        msms_x,
        msms_mask,
        nl_x,
        nl_mask,
        precursor_x
    ])[:, 1].item()

    importances = np.zeros(N)

    for i in range(N):
        if not valid[i]:
            continue

        mask_i = nl_mask.clone()
        mask_i[0, i] = False

        logit_i = model([
            msms_x,
            msms_mask,
            nl_x,
            mask_i,
            precursor_x
        ])[:, 1].item()

        importances[i] = base_logit - logit_i

    return importances

rows = []

for sample_idx in high_conf_indices:
    x, y = dataset[sample_idx]

    smiles = dataset.smiles[sample_idx]

    msms_x = x[0]
    nl_x = x[2]
    precursor_x = x[4]

    imp = neutral_loss_ablation(model, msms_x, nl_x, precursor_x)

    nl_np = nl_x.cpu().numpy()[:, 0]
    valid = nl_np > 0
    imp_valid = imp[valid]

    if len(imp_valid) == 0:
        continue

    rows.append({
        "sample_idx": int(sample_idx),
        "Canonical_smiles": smiles,
        "label": int(y.item()),
        "n_neutral_losses": int(valid.sum()),
        "mean_abs_importance": float(np.mean(np.abs(imp_valid))),
        "mean_signed_importance": float(np.mean(imp_valid)),
        "max_abs_importance": float(np.max(np.abs(imp_valid))),
        "sum_abs_importance": float(np.sum(np.abs(imp_valid))),
    })

df_nl_samples = pd.DataFrame(rows)
df_nl_samples.to_csv(
    "PBT_neutral_loss_importance_sample_level.csv",
    index=False
)

print("Saved: PBT_neutral_loss_importance_sample_level.csv")

frag_rows = []

for sample_idx in high_conf_indices:
    x, _ = dataset[sample_idx]

    smiles = dataset.smiles[sample_idx]

    msms_x = x[0]
    nl_x = x[2]
    precursor_x = x[4]

    imp = neutral_loss_ablation(model, msms_x, nl_x, precursor_x)

    nl_np = nl_x.cpu().numpy()[:, 0]

    for i in range(len(nl_np)):
        if nl_np[i] <= 0:
            continue

        frag_rows.append({
            "sample_idx": int(sample_idx),
            "Canonical_smiles": smiles,
            "neutral_loss": float(nl_np[i]),
            "importance": float(imp[i]),
            "abs_importance": float(abs(imp[i]))
        })

df_nl_frags = pd.DataFrame(frag_rows)
df_nl_frags.to_csv(
    "PBT_neutral_loss_importance_loss_level.csv",
    index=False
)

print("Saved: PBT_neutral_loss_importance_fragment_level.csv")