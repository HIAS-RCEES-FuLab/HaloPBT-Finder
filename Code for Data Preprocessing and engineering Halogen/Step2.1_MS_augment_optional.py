import os
import pandas as pd
import numpy as np
import ast

base_dir = r"D:\MSMS\MSMS_ClBr_H+H-\Train_data"
aug_dir = os.path.join(base_dir, "weighted_augmented")
os.makedirs(aug_dir, exist_ok=True)

noise_std = 0.05  # ±5% random noise

# --- Read Cl/Br augmentation factors ---
factor_csv = os.path.join(base_dir, "ClBr_count_stats", "ClBr_total_counts_with_factor.csv")
factor_df = pd.read_csv(factor_csv)
factor_map = {(int(row.Cl_count), int(row.Br_count)): int(row.augment_factor)
              for _, row in factor_df.iterrows()}

# --- Utility functions ---
def parse_list(x):
    try:
        return ast.literal_eval(x) if pd.notna(x) else []
    except:
        return []

def add_noise(pattern):
    pattern = np.array(pattern, dtype=float)
    noisy = pattern * (1 + np.random.normal(0, noise_std, len(pattern)))
    return noisy.tolist()

# --- Augmentation loop ---
for fname in os.listdir(base_dir):
    if not fname.endswith(".csv"):
        continue
    fpath = os.path.join(base_dir, fname)
    if "ClBr_count_stats" in fpath:
        continue

    try:
        df = pd.read_csv(fpath)
        required_cols = {"isotope_pattern_M0_M6", "Cl_count", "Br_count"}
        if not required_cols.issubset(df.columns):
            print(f"Skipping {fname} (missing required columns)")
            continue

        df["isotope_pattern_M0_M6"] = df["isotope_pattern_M0_M6"].apply(parse_list)
        augmented_rows = []

        for _, row in df.iterrows():
            cl, br = int(row["Cl_count"]), int(row["Br_count"])
            factor = factor_map.get((cl, br), 0)

            # Original row
            base_row = row.copy()
            base_row["augmented"] = False
            augmented_rows.append(base_row)

            # Augmented rows
            for i in range(factor):
                aug_row = row.copy()
                aug_row["isotope_pattern_M0_M6"] = add_noise(row["isotope_pattern_M0_M6"])
                aug_row["augmented"] = True
                aug_row["aug_id"] = i + 1
                augmented_rows.append(aug_row)

        out_df = pd.DataFrame(augmented_rows)
        out_fname = fname.replace(".csv", "_weighted_augmented.csv")
        out_path = os.path.join(aug_dir, out_fname)
        out_df.to_csv(out_path, index=False)

        print(f"✅ Generated weighted augmented file: {out_path}")

    except Exception as e:
        print(f"❌ Failed to process {fname}: {e}")