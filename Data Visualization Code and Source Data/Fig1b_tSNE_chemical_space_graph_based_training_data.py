import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# =====================================================
# This script analyzes MS/MS PBT training data.
# The data in the CSV files is already preprocessed and counted
# =====================================================

# =============================
# Configuration
# =============================
data_dir = r"D:\MSMS\MSMS_PBT\MS2_PBT_database_H+H-\Train_data"
out_csv = os.path.join(data_dir, "MSMS_statistics.csv")

# =============================
# Global containers (across files)
# =============================
global_canonical_set = set()
global_mol_records = []

# =============================
# File-level statistics
# =============================
stats = []

for fname in os.listdir(data_dir):
    if not fname.endswith(".csv"):
        continue

    fpath = os.path.join(data_dir, fname)
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        print(f"Cannot read {fname}: {e}")
        continue

    # ---------- MS/MS level ----------
    msms_count = len(df)

    # ---------- MS/MS level: PBT / non-PBT ----------
    if "PBT_label" in df.columns:
        msms_label_clean = df["PBT_label"].astype(str).str.strip()
        msms_pbt = (msms_label_clean == "1").sum()
        msms_nonpbt = (msms_label_clean == "0").sum()
    else:
        msms_pbt = 0
        msms_nonpbt = 0

    # ---------- Molecule level (Canonical SMILES) ----------
    if "Canonical_smiles" in df.columns:
        canonical = df["Canonical_smiles"].dropna()
        canonical = canonical[canonical.astype(str).str.strip() != ""]
        unique_molecule_count = canonical.nunique()
    else:
        unique_molecule_count = 0

    # ---------- PBT distribution (molecule level) ----------
    if {"Canonical_smiles", "PBT_label"}.issubset(df.columns):
        mol_df = df[["Canonical_smiles", "PBT_label"]].dropna(subset=["Canonical_smiles"])
        mol_df = mol_df[mol_df["Canonical_smiles"].astype(str).str.strip() != ""]
        mol_df = mol_df.drop_duplicates(subset="Canonical_smiles")
        pbt_dist = mol_df["PBT_label"].value_counts().to_dict()
    else:
        mol_df = None
        pbt_dist = {}

    stats.append({
        "file": fname,
        "msms_count": msms_count,
        "msms_pbt": msms_pbt,
        "msms_nonpbt": msms_nonpbt,
        "unique_molecule_count": unique_molecule_count,
        "PBT_label_distribution_molecule_level": pbt_dist
    })

    # ---------- Accumulate across files ----------
    if "Canonical_smiles" in df.columns:
        global_canonical_set.update(canonical.unique())

    if mol_df is not None:
        global_mol_records.append(mol_df)

# =============================
# Global statistics (all files)
# =============================
global_unique_molecule_count = len(global_canonical_set)

if global_mol_records:
    global_mol_df = pd.concat(global_mol_records, ignore_index=True)
    global_mol_df = global_mol_df.drop_duplicates(subset="Canonical_smiles")
    global_pbt_dist = global_mol_df["PBT_label"].value_counts().to_dict()
else:
    raise RuntimeError("No molecules available for t-SNE")

# =============================
# Add summary row for ALL_FILES
# =============================
stats_df = pd.DataFrame(stats)
stats_df.loc[len(stats_df)] = {
    "file": "ALL_FILES",
    "msms_count": stats_df["msms_count"].sum(),
    "msms_pbt": stats_df["msms_pbt"].sum(),
    "msms_nonpbt": stats_df["msms_nonpbt"].sum(),
    "unique_molecule_count": global_unique_molecule_count,
    "PBT_label_distribution_molecule_level": global_pbt_dist
}

stats_df.to_csv(out_csv, index=False)
print(f"Statistics saved to: {out_csv}")

# =====================================================
# t-SNE on unique molecules
# =====================================================
print("\nRunning t-SNE on unique molecules...")

def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)

fps = []
labels = []

for _, row in global_mol_df.iterrows():
    fp = smiles_to_morgan(row["Canonical_smiles"])
    if fp is None:
        continue
    fps.append(fp)
    labels.append(row["PBT_label"])

X = np.array(fps)
y = np.array(labels)
print(f"Molecules used for t-SNE: {X.shape[0]}")

# ---------- t-SNE ----------
tsne = TSNE(
    n_components=2,
    perplexity=50,
    max_iter=2000,
    init="pca",
    random_state=42,
    learning_rate="auto"
)
X_tsne = tsne.fit_transform(X)

# ---------- Debug ----------
print("X_tsne shape:", X_tsne.shape)
print("y length:", len(y))
print("Unique labels:", np.unique(y))

# ---------- Plot ----------
plt.figure(figsize=(6, 6))

# Clean labels for consistency
y_clean = np.array([str(lbl).strip() for lbl in labels])
color_map = {
    "1": "#fb8072",  # pink/red
    "0": "#80b1d3"   # blue
}

for label, color in color_map.items():
    idx = y_clean == label
    n_points = np.sum(idx)
    if n_points == 0:
        print(f"Warning: no points for label '{label}'")
        continue
    plt.scatter(
        X_tsne[idx, 0],
        X_tsne[idx, 1],
        s=40,
        alpha=1,
        edgecolors='gray',
        linewidths=0.5,
        color=color,
        label=f"{label.upper()} ({n_points})"
    )

# Remove axes and frame for publication-style figure
plt.xticks([])
plt.yticks([])
plt.gca().set_frame_on(False)
plt.tight_layout()

# Save transparent PNG
out_png = os.path.join(data_dir, "tSNE_plot.png")
plt.savefig(out_png, dpi=300, bbox_inches='tight', transparent=True)
plt.show()

print(f"t-SNE plot saved to: {out_png}")