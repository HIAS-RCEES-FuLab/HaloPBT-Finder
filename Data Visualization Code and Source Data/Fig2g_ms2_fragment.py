import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =============================
# Global font settings
# =============================
plt.rcParams["font.family"] = "Arial"

# =============================
# 1. Read CSV file
# =============================
csv_path = "PBT_msms_fragment_importance_fragment_level.csv"
df = pd.read_csv(csv_path)

# =============================
# 2. Prepare basic fields
# =============================
df["mz_plot"] = df["mz"] * 1000.0

# Retain only real peaks
df = df[df["intensity"] > 0].reset_index(drop=True)

# =============================
# 3. Retain only positive importance
# =============================
df_pos = df[df["importance"] > 0].copy().reset_index(drop=True)
print(f"Total positive-attribution fragments: {len(df_pos)}")

# =============================
# 4. 95th percentile threshold for positive importance
# =============================
v95 = np.percentile(df_pos["importance"], 95)
print(f"95th percentile (positive importance) = {v95:.6e}")

# =============================
# 5. Retain fragments with high importance
# =============================
high_imp_df = df_pos[df_pos["importance"] >= v95].copy()
high_imp_df = high_imp_df.sort_values("mz_plot").reset_index(drop=True)

high_imp_df.to_csv(
    "PBT_fragment_raw.csv",
    index=False
)

print(f"High-importance positive fragments (≥95%): {len(high_imp_df)}")

# =============================
# 6. Merge fragments within ±5 ppm
#    using only the top 5% positive fragments
# =============================
ppm_tol = 5
used = np.zeros(len(high_imp_df), dtype=bool)
merged_rows = []

for i, row in high_imp_df.iterrows():
    if used[i]:
        continue

    ref_mz = row["mz_plot"]
    ppm_diff = np.abs(high_imp_df["mz_plot"] - ref_mz) / ref_mz * 1e6
    group_mask = (ppm_diff <= ppm_tol) & (~used)

    group = high_imp_df[group_mask]
    used[group_mask.values] = True

    idx_max = group["importance"].idxmax()

    merged_rows.append({
        "mz": ref_mz,
        "n_samples": group["sample_idx"].nunique(),   # Number of unique samples
        "importance": group.loc[idx_max, "importance"],
        "n_fragments_merged": len(group)
    })

merged_df = pd.DataFrame(merged_rows)

merged_df = merged_df.sort_values(
    by=["n_samples", "importance"],
    ascending=[False, False]
).reset_index(drop=True)

merged_df.to_csv(
    "PBT_fragment_merged.csv",
    index=False
)

print("Merged high-importance fragments (positive, ±5 ppm):")
print(merged_df.head())

# =============================
# 7. Scatter plot
#    m/z vs occurrence frequency, colored by importance
# =============================
plot_df = merged_df.sort_values(
    by="importance",
    ascending=True
).reset_index(drop=True)

plt.figure(figsize=(8, 6))

sc = plt.scatter(
    plot_df["mz"],
    plot_df["n_samples"],
    c=plot_df["importance"],
    cmap="coolwarm",
    s=200,
    edgecolor="black",
    linewidth=0.6
)

#cbar = plt.colorbar(sc)
#cbar.set_label("Fragment importance", fontsize=14)

# =============================
# Keep tick marks but hide tick labels
# =============================
plt.tick_params(
    axis="both",
    which="both",
    labelbottom=False,
    labelleft=False,
    length=5,
    width=1
)

plt.grid(alpha=0.3)

plt.tight_layout()
plt.xlim(0, 1000)

plt.savefig(
    "PBT_msms_fragment_95pct_count_scatter_5ppm.png",
    dpi=300,
    transparent=True,
    bbox_inches="tight"
)

plt.show()