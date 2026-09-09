import pandas as pd
import matplotlib.pyplot as plt

input_file = r"C:\Users\13709\Desktop\论文撰写四\20260710HaloPBT\血浆矩阵\result\PBT_match_Plasma_final.csv"
input_matrix = r"C:\Users\13709\Desktop\论文撰写四\20260710HaloPBT\血浆矩阵\result\Plasma_combined_matrix_match_final.csv"
output_file = r"C:\Users\13709\Desktop\论文撰写四\20260710HaloPBT\血浆矩阵\result\Plasma_Pollutant_Type_summary.csv"
plot_file = r"C:\Users\13709\Desktop\Seafood_Pollutant_Type_violin.png"

df = pd.read_csv(input_file, low_memory=False)
matrix = pd.read_csv(input_matrix, low_memory=False)

if len(df) != len(matrix):
    raise ValueError(f"两个文件行数不一致: {len(df)} vs {len(matrix)}")

if "Pollutant_Type" not in df.columns:
    raise ValueError("缺少 Pollutant_Type 列")

print("匹配行数:", len(df))

# The first column contains compound IDs; the remaining columns contain samples
sample_cols = matrix.columns[1:]
print("样品数量:", len(sample_cols))

matrix[sample_cols] = matrix[sample_cols].apply(
    pd.to_numeric, errors="coerce"
).fillna(0)

# Calculate the total signal intensity for each compound
matrix["Total_Intensity"] = matrix[sample_cols].sum(axis=1)

df_result = df[["Pollutant_Type"]].copy()
df_result["Total_Intensity"] = matrix["Total_Intensity"].values

# Summarize by pollutant type
summary = (
    df_result.groupby("Pollutant_Type")
    .agg(
        Compound_Number=("Pollutant_Type", "count"),
        Total_Intensity=("Total_Intensity", "sum"),
        Mean_Intensity=("Total_Intensity", "mean")
    )
    .reset_index()
    .sort_values("Total_Intensity", ascending=False)
    .reset_index(drop=True)
)

summary.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n========== Summary ==========")
print(summary.to_string(index=False))

# Calculate the total intensity for each pollutant type in each sample
df_sample = df[["Pollutant_Type"]].copy()

for col in sample_cols:
    df_sample[col] = matrix[col].values

category_sample = (
    df_sample.groupby("Pollutant_Type")[sample_cols]
    .sum()
)

categories = summary["Pollutant_Type"].tolist()

violin_data = [
    category_sample.loc[c, sample_cols].values
    for c in categories
]

# Reorder pollutant types for the violin plot
new_order = [
    "Pharmaceuticals",
    "Industry and consumer goods",
    "Transformation and metabolic products",
    "Food and other consumption",
    "Personal care products",
    "Pesticides"
]

# Generate violin plot data according to the new order
violin_data = [
    category_sample.loc[c, sample_cols].values
    for c in new_order
]

categories = new_order

fig, ax = plt.subplots(figsize=(8, 6))

ax.violinplot(
    violin_data,
    positions=range(1, len(categories)+1),
    showmedians=True,
    showextrema=True
)

for i, values in enumerate(violin_data, 1):
    n = len(values)
    jitter = ((pd.Series(range(n)) % 9 - 4) * 0.018).values
    ax.scatter(
        i + jitter,
        values,
        s=10,
        alpha=1
    )

# Keep tick marks but hide category labels
ax.set_xticks(range(1, len(categories)+1))
ax.set_xticklabels([])

# Set the Y-axis range to 0–25 and hide numerical labels
ax.set_ylim(0, 25)

# Keep tick marks while hiding axis labels
ax.tick_params(
    axis="both",
    which="both",
    labelbottom=False,
    labelleft=False
)

print(categories)

plt.tight_layout()

plt.savefig(
    plot_file,
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

plt.show()