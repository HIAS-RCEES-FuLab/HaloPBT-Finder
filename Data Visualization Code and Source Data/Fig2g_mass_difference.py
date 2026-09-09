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
csv_path = "PBT_mass_difference_importance_fragment_level.csv"
df = pd.read_csv(csv_path)

required_cols = ["sample_idx", "neutral_loss", "importance"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# =============================
# 2. Prepare basic fields
# =============================
df["neutral_loss_Da"] = df["neutral_loss"] * 500.0
df["abs_importance"] = df["importance"].abs()

# Retain valid neutral losses
df = df[df["neutral_loss_Da"] > 0].reset_index(drop=True)

# Retain only positive importance values
# representing neutral losses with positive contributions
df = df[df["importance"] > 0].reset_index(drop=True)

print(f"Total neutral-loss entries (importance > 0): {len(df)}")

# =============================
# 3. 95th percentile importance threshold
#    within positive importance values
# =============================
v95 = np.percentile(df["abs_importance"], 95)
print(f"95th percentile importance = {v95:.4e}")

high_df = df[df["abs_importance"] >= v95].copy().reset_index(drop=True)
print(f"High-importance neutral-loss entries (≥95%): {len(high_df)}")

# =============================
# 4. Merge neutral losses within ±5 ppm
# =============================
ppm_tol = 5
used = np.zeros(len(high_df), dtype=bool)
merged_rows = []

for i, row in high_df.iterrows():
    if used[i]:
        continue

    ref_nl = row["neutral_loss_Da"]
    ppm_diff = np.abs(high_df["neutral_loss_Da"] - ref_nl) / ref_nl * 1e6
    group_mask = (ppm_diff <= ppm_tol) & (~used)

    group = high_df[group_mask]
    used[group_mask.values] = True

    idx_max = group["abs_importance"].idxmax()

    merged_rows.append({
        "neutral_loss_Da": ref_nl,
        "n_fragments": len(group),
        "n_samples": group["sample_idx"].nunique(),
        "max_importance": group.loc[idx_max, "importance"],
        "max_abs_importance": group["abs_importance"].max()
    })

nl_merged = pd.DataFrame(merged_rows)

# =============================
# 5. Filter neutral loss range (25–200 Da)
# =============================
nl_merged = nl_merged[
    (nl_merged["neutral_loss_Da"] >= 25) &
    (nl_merged["neutral_loss_Da"] <= 200)
].copy()

print(f"Neutral losses retained (25–200 Da): {len(nl_merged)}")

# =============================
# 6. Sort by occurrence frequency and importance
#    without truncation
# =============================
nl_sorted = (
    nl_merged
    .sort_values(
        by=["n_samples", "max_abs_importance"],
        ascending=[False, False]
    )
    .reset_index(drop=True)
)

# Save the complete results
nl_sorted.to_csv(
    "PBT_neutral_loss.csv",
    index=False
)

# =============================
# 7. Scatter plot of all results
#    without Top-N truncation
# =============================
plt.figure(figsize=(8, 6))

# Plot lower-importance points first to reduce overlap
plot_df = nl_sorted.sort_values(
    by="max_abs_importance",
    ascending=True
)

sc = plt.scatter(
    plot_df["neutral_loss_Da"],
    plot_df["n_samples"],
    c=plot_df["max_abs_importance"],
    s=200,
    cmap="coolwarm",
    edgecolor="black",
    linewidth=0.6,
)

# cbar = plt.colorbar(sc)
# cbar.set_label("Mass difference importance", fontsize=13)

# plt.xlabel("Mass difference (Da)", fontsize=14)
# plt.ylabel("Occurrence frequency", fontsize=14)

from matplotlib.ticker import MaxNLocator

ax = plt.gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

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

plt.xlim(0, 210)
plt.ylim(0, 20)
plt.yticks([0, 5, 10, 15, 20])
plt.xticks([0, 40, 80, 120, 160, 200])
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "neutral_loss_95pct_frequency_scatter_5ppm.png",
    dpi=300,
    transparent=True,
    bbox_inches="tight"
)

plt.show()