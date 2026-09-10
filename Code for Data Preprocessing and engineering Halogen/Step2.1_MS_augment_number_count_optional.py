import os
import pandas as pd
import numpy as np

base_dir = r"D:\MSMS\MSMS_ClBr_H+H-\Train_data"
output_dir = os.path.join(base_dir, "ClBr_count_stats")
os.makedirs(output_dir, exist_ok=True)

all_stats = []

for fname in os.listdir(base_dir):
    if not fname.endswith(".csv") or "with" not in fname.lower():
        continue

    fpath = os.path.join(base_dir, fname)
    try:
        df = pd.read_csv(fpath)

        if "Cl_count" not in df.columns or "Br_count" not in df.columns:
            print(f"Skipping {fname} (missing Cl_count or Br_count columns)")
            continue

        # Count occurrences of each Cl/Br combination
        count_stats = df.groupby(['Cl_count', 'Br_count']).size().reset_index(name='count')
        count_stats['file'] = fname

        # Compute augmentation factor (cube root scaling)
        count_stats['augment_factor'] = np.ceil((count_stats['count'].max() / count_stats['count'])**(1/3)).astype(int)

        # Save per-file statistics
        out_fname = fname.replace(".csv", "_ClBr_stats_with_factor.csv")
        out_fpath = os.path.join(output_dir, out_fname)
        count_stats.to_csv(out_fpath, index=False)
        all_stats.append(count_stats)
        print(f"✅ Saved per-file statistics: {out_fpath}")

    except Exception as e:
        print(f"❌ Failed to process {fname}: {e}")

# Merge all statistics
if all_stats:
    merged_stats = pd.concat(all_stats, ignore_index=True)
    total_counts = merged_stats.groupby(['Cl_count', 'Br_count'])['count'].sum().reset_index()
    total_counts['augment_factor'] = np.ceil((total_counts['count'].max() / total_counts['count'])**(1/3)).astype(int)
    total_file = os.path.join(output_dir, "ClBr_total_counts_with_factor.csv")
    total_counts.to_csv(total_file, index=False)
    print(f"✅ Saved merged statistics: {total_file}")