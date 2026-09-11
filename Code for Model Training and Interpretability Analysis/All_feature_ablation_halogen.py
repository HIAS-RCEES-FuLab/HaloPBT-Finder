import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# -----------------------------
# 配置
# -----------------------------
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
precursor_dim = 3 + 7 + 1   # exact_mass + KMD_Cl + KMD_Br + isotope(M0–M6) + ion_mode
latent_dim = 128
epochs = 50
cl_num_classes = 8
br_num_classes = 6
name = "multi_cl_br_transformer_0.01_0122"

# -----------------------------
# Dataset
# -----------------------------
class MultiModalDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        self.cl_labels = []
        self.br_labels = []
        self.formula = []
        self.smiles = []
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
                self.data.append((msms_x, msms_mask, nl_x, nl_mask, precursor_x))
                self.cl_labels.append(label_cl)
                self.br_labels.append(label_br)
                self.formula.append(row["Formula"])
                self.smiles.append(row["SMILES"])  # ← 新增
        self.formula = np.array(self.formula)
        self.smiles = np.array(self.smiles)
        self.cl_labels = torch.tensor(np.array(self.cl_labels), dtype=torch.long)
        self.br_labels = torch.tensor(np.array(self.br_labels), dtype=torch.long)
        print(f"Total samples: {len(self.cl_labels)}")

    def __len__(self):
        return len(self.cl_labels)

    def __getitem__(self, idx):
        return self.data[idx], (self.cl_labels[idx], self.br_labels[idx])

# -----------------------------
# 子网络
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
                nn.Dropout(0.2)
            ]
            d = h
        layers += [nn.Linear(d, latent_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# -----------------------------
# Transformer
# -----------------------------
class MSMSTransformer(nn.Module):
    def __init__(self, peak_dim=2, latent_dim=128, nhead=4, num_layers=3, max_peaks=50):
        super().__init__()
        self.proj = nn.Linear(peak_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=nhead, dim_feedforward=latent_dim*4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1,1,latent_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_peaks+1, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, mask):
        B, K, _ = x.shape
        x = self.proj(x)
        cls = self.cls_token.expand(B,-1,-1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :K+1]
        cls_mask = torch.ones(B, 1, device=x.device, dtype=torch.bool)
        key_mask = torch.cat([cls_mask, mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=~key_mask)
        x = self.norm(x[:,0])
        return x

# -----------------------------
# 多模态网络
# -----------------------------
class MultiModalNet(nn.Module):
    def __init__(self, precursor_dim, latent_dim, cl_num_classes, br_num_classes, max_peaks=50):
        super().__init__()
        self.msms_net = MSMSTransformer(2, latent_dim, max_peaks=max_peaks)
        self.nl_proj = nn.Linear(1,32)
        self.nl_net = SubNet(32, latent_dim, [128,64])
        self.precursor_net = SubNet(precursor_dim, latent_dim, [32,64])
        self.cl_head = nn.Sequential(nn.Linear(latent_dim*3,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, cl_num_classes))
        self.br_head = nn.Sequential(nn.Linear(latent_dim*3,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, br_num_classes))

    def forward(self, x):
        msms_x, msms_mask, nl_x, nl_mask, precursor_x = x
        msms_latent = self.msms_net(msms_x, msms_mask)
        nl_feat = self.nl_proj(nl_x)
        nl_mask = nl_mask.unsqueeze(-1)
        nl_feat = nl_feat * nl_mask
        nl_latent = self.nl_net(nl_feat.sum(1)/nl_mask.sum(1).clamp(min=1))
        precursor_latent = self.precursor_net(precursor_x)
        fused = torch.cat([msms_latent, nl_latent, precursor_latent], dim=1)
        cl_logits = self.cl_head(fused)
        br_logits = self.br_head(fused)
        return cl_logits, br_logits

# ============================================================
# 消融函数
# ============================================================
@torch.no_grad()
def precursor_ablation(model, msms_x, msms_mask, nl_x, nl_mask,
                       precursor_x, cl_label, br_label):
    D = precursor_x.shape[0]

    base_cl, base_br = model(
        (msms_x.unsqueeze(0), msms_mask.unsqueeze(0),
         nl_x.unsqueeze(0), nl_mask.unsqueeze(0),
         precursor_x.unsqueeze(0))
    )
    base_cl = base_cl[0, cl_label].item()
    base_br = base_br[0, br_label].item()

    cl_imp = np.zeros(D)
    br_imp = np.zeros(D)

    for d in range(D):
        px = precursor_x.clone()
        px[d] = 0.0   # 单维度消融

        cl_l, br_l = model(
            (msms_x.unsqueeze(0), msms_mask.unsqueeze(0),
             nl_x.unsqueeze(0), nl_mask.unsqueeze(0),
             px.unsqueeze(0))
        )
        cl_imp[d] = base_cl - cl_l[0, cl_label].item()
        br_imp[d] = base_br - br_l[0, br_label].item()

    return cl_imp, br_imp

@torch.no_grad()
def msms_fragment_ablation(model, msms_x, msms_mask, nl_x, nl_mask, precursor_x, cl_label, br_label):
    K = msms_x.shape[0]
    base_cl, base_br = model(
        (msms_x.unsqueeze(0), msms_mask.unsqueeze(0),
         nl_x.unsqueeze(0), nl_mask.unsqueeze(0),
         precursor_x.unsqueeze(0))
    )
    base_cl = base_cl[0, cl_label].item()
    base_br = base_br[0, br_label].item()

    cl_imp = np.zeros(K)
    br_imp = np.zeros(K)

    for k in range(K):
        if not msms_mask[k]:
            continue
        mask = msms_mask.clone()
        mask[k] = False
        cl_l, br_l = model(
            (msms_x.unsqueeze(0), mask.unsqueeze(0),
             nl_x.unsqueeze(0), nl_mask.unsqueeze(0),
             precursor_x.unsqueeze(0))
        )
        cl_imp[k] = base_cl - cl_l[0, cl_label].item()
        br_imp[k] = base_br - br_l[0, br_label].item()

    return cl_imp, br_imp


@torch.no_grad()
def nl_ablation(model, msms_x, msms_mask, nl_x, nl_mask, precursor_x, cl_label, br_label):
    N = nl_x.shape[0]
    base_cl, base_br = model(
        (msms_x.unsqueeze(0), msms_mask.unsqueeze(0),
         nl_x.unsqueeze(0), nl_mask.unsqueeze(0),
         precursor_x.unsqueeze(0))
    )
    base_cl = base_cl[0, cl_label].item()
    base_br = base_br[0, br_label].item()

    cl_imp = np.zeros(N)
    br_imp = np.zeros(N)

    for i in range(N):
        if not nl_mask[i]:
            continue
        mask = nl_mask.clone()
        mask[i] = False
        cl_l, br_l = model(
            (msms_x.unsqueeze(0), msms_mask.unsqueeze(0),
             nl_x.unsqueeze(0), mask.unsqueeze(0),
             precursor_x.unsqueeze(0))
        )
        cl_imp[i] = base_cl - cl_l[0, cl_label].item()
        br_imp[i] = base_br - br_l[0, br_label].item()

    return cl_imp, br_imp

# ============================================================
# 主流程
# ============================================================
dataset = MultiModalDataset(files)
model_path = "multimodal_multi_cl_br_transformer_0.01_0122_fold1.pth"
model = MultiModalNet(precursor_dim, latent_dim, cl_num_classes, br_num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 分组划分 train/val/test（按 formula）
indices = np.arange(len(dataset))
labels = dataset.cl_labels * 100 + dataset.br_labels
groups = dataset.smiles

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_val_idx, test_idx = next(gss1.split(indices, labels, groups))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(gss2.split(train_val_idx, labels[train_val_idx], groups[train_val_idx]))

subset_dict = {
    "train": train_idx,
    "val": val_idx,
    "test": test_idx
}

precursor_names = (
    ["Exact_mass", "KMD_Cl", "KMD_Br"] +
    [f"iso_M{i}" for i in range(7)] +
    ["ion_mode"]
)

# ============================================================
# 消融分析（只对预测正确样本）
# ============================================================
rows = []

output_prefix = "ClBr_fragment_smiles_ablation_correct_only"
chunk_size = 100000
file_count = 0

def save_chunk(rows, file_count):
    if len(rows) == 0:
        return file_count
    df = pd.DataFrame(rows)
    output_file = f"{output_prefix}_{file_count}.csv"
    df.to_csv(
        output_file,
        index=False
    )
    print(f"Saved {output_file}, rows={len(df)}")
    rows.clear()
    return file_count + 1

for subset_name, idx_list in subset_dict.items():
    print(f"Processing {subset_name} set, {len(idx_list)} samples...")
    for idx in idx_list:
        (msms_x, msms_mask, nl_x, nl_mask, precursor_x), (cl, br) = dataset[idx]
        smiles = dataset.smiles[idx]

        # 转为 tensor
        msms_x_t = torch.tensor(msms_x).to(device)
        msms_mask_t = torch.tensor(msms_mask).to(device)
        nl_x_t = torch.tensor(nl_x).to(device)
        nl_mask_t = torch.tensor(nl_mask).to(device)
        precursor_x_t = torch.tensor(precursor_x).to(device)

        # 模型预测
        cl_logits, br_logits = model(
            (msms_x_t.unsqueeze(0), msms_mask_t.unsqueeze(0),
             nl_x_t.unsqueeze(0), nl_mask_t.unsqueeze(0),
             precursor_x_t.unsqueeze(0))
        )

        #correct = (cl_logits.argmax(1).item() == cl) and (br_logits.argmax(1).item() == br)
        #if not correct:
            #continue   # 只处理预测正确样本

        # 消融计算
        cl_imp_pre, br_imp_pre = precursor_ablation(model, msms_x_t, msms_mask_t, nl_x_t, nl_mask_t, precursor_x_t, cl, br)
        cl_imp_ms, br_imp_ms = msms_fragment_ablation(model, msms_x_t, msms_mask_t, nl_x_t, nl_mask_t, precursor_x_t, cl, br)
        cl_imp_nl, br_imp_nl = nl_ablation(model, msms_x_t, msms_mask_t, nl_x_t, nl_mask_t, precursor_x_t, cl, br)

        # 保存 precursor
        for d, name in enumerate(precursor_names):
            rows.append({
                "subset": subset_name,
                "sample_idx": idx,
                "Formula": dataset.formula[idx],
                "SMILES": smiles,
                "type": "Precursor",
                "feature": name,
                "target": "Cl",
                "true_label": cl,
                "importance": cl_imp_pre[d],
            })
            rows.append({
                "subset": subset_name,
                "sample_idx": idx,
                "Formula": dataset.formula[idx],
                "SMILES": smiles,
                "type": "Precursor",
                "feature": name,
                "target": "Br",
                "true_label": br,
                "importance": br_imp_pre[d],
            })

        # 保存 MSMS fragments
        for k in range(len(msms_x)):
            if not msms_mask[k]:
                continue
            rows.append({
                "subset": subset_name,
                "sample_idx": idx,
                "Formula": dataset.formula[idx],
                "SMILES": smiles,
                "type": "MSMS",
                "fragment_idx": k,
                "mz": msms_x[k,0]*MAX_MZ,
                "target": "Cl",
                "true_label": cl,
                "importance": cl_imp_ms[k],
            })
            rows.append({
                "subset": subset_name,
                "sample_idx": idx,
                "Formula": dataset.formula[idx],
                "SMILES": smiles,
                "type": "MSMS",
                "fragment_idx": k,
                "mz": msms_x[k,0]*MAX_MZ,
                "target": "Br",
                "true_label": br,
                "importance": br_imp_ms[k],
            })

        # 保存 Neutral losses
        for i in range(len(nl_x)):
            if not nl_mask[i]:
                continue
            rows.append({
                "subset": subset_name,
                "sample_idx": idx,
                "Formula": dataset.formula[idx],
                "type": "NeutralLoss",
                "fragment_idx": i,
                "neutral_loss": nl_x[i,0]*MAX_NL_MASS,
                "target": "Cl",
                "true_label": cl,
                "importance": cl_imp_nl[i],
            })
            rows.append({
                "subset": subset_name,
                "sample_idx": idx,
                "Formula": dataset.formula[idx],
                "type": "NeutralLoss",
                "fragment_idx": i,
                "neutral_loss": nl_x[i,0]*MAX_NL_MASS,
                "target": "Br",
                "true_label": br,
                "importance": br_imp_nl[i],
            })
        # ===============================
        # 达到10万行保存
        # ===============================
        if len(rows) >= chunk_size:
            file_count = save_chunk(rows, file_count)
# 保存最后不足10万行的数据
file_count = save_chunk(rows, file_count)

print("All chunks saved.")