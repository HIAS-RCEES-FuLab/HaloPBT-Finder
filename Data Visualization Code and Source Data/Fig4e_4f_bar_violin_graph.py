import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# File paths
# =========================
input_file = r"C:\Users\13709\Desktop\论文撰写四\20260708HaloPBT\Seafood_combined_detection_matrix_mergeESI.csv"
output_violin = r"C:\Users\13709\Desktop\论文撰写四\20260708HaloPBT\Violin_SampleIntensity.png"
output_bar = r"C:\Users\13709\Desktop\论文撰写四\20260708HaloPBT\Province_Level_Count.png"

# =========================
# Bar plot mode
# "level": stacked bar plot by identification level
# "total": bar plot of total detected compounds
# =========================
bar_mode = "total"

# =========================
# Read detection matrix
# =========================
df = pd.read_csv(input_file, index_col=0)

# =====================================================
# Extract sample province information
# =====================================================
sample_info = pd.DataFrame({"Sample": df.columns})
sample_info["Province"] = sample_info["Sample"].str.split("_").str[0]

# =====================================================
# Figure 1: Sample summed intensity violin plot
# =====================================================
sample_sum = df.sum(axis=0)

plot_df = pd.DataFrame({
    "Sample": sample_sum.index,
    "Intensity": sample_sum.values
})

plot_df["Province"] = plot_df["Sample"].str.split("_").str[0]

# =========================
# Sort provinces by maximum sample intensity
# =========================
province_order = (
    plot_df
    .groupby("Province")["Intensity"]
    .max()
    .sort_values(ascending=False)
    .index
    .tolist()
)

print("\n========== Province order ==========")
for p in province_order:
    max_intensity = plot_df.loc[
        plot_df["Province"] == p,
        "Intensity"
    ].max()
    print(f"{p}: {max_intensity:.2f}")

# =========================
# Violin plot
# =========================
plt.figure(figsize=(10, 4))

sns.violinplot(
    data=plot_df,
    x="Province",
    y="Intensity",
    order=province_order,
    inner=None,
    cut=0,
    linewidth=1,
    color="#9CC6E4",
    saturation=1,
    width=0.8
)

sns.swarmplot(
    data=plot_df,
    x="Province",
    y="Intensity",
    order=province_order,
    color="black",
    size=4
)

plt.xlim(-0.5, len(province_order) - 0.5)
plt.xlabel("")
plt.ylabel("")

plt.tick_params(axis="y", labelleft=False)
plt.tick_params(axis="x", labelbottom=False)

plt.ylim(0, 1050)
plt.tight_layout()

plt.savefig(
    output_violin,
    dpi=600,
    transparent=True
)

plt.show()
plt.close()

print(f"Violin saved: {output_violin}")

# =====================================================
# Figure 2: Province × identification level detected compounds
# =====================================================
compound_info = pd.DataFrame({"Compound": df.index})
compound_info["Level"] = compound_info["Compound"].str.split("_").str[0]

level_order = ["L1", "L2a", "L2b", "L3"]

compound_info = compound_info[
    compound_info["Level"].isin(level_order)
]

result = []

for province in province_order:
    province_samples = sample_info.loc[
        sample_info["Province"] == province,
        "Sample"
    ]

    sub = df[province_samples]

    for level in level_order:
        compounds = compound_info.loc[
            compound_info["Level"] == level,
            "Compound"
        ]

        sub_level = sub.loc[
            sub.index.intersection(compounds)
        ]

        detected = (sub_level != 0).any(axis=1)

        result.append({
            "Province": province,
            "Level": level,
            "Count": detected.sum()
        })

level_df = pd.DataFrame(result)

stack_df = level_df.pivot(
    index="Province",
    columns="Level",
    values="Count"
).fillna(0)

# Keep the same province order as the violin plot
stack_df = stack_df.loc[
    province_order,
    level_order
]

# =========================
# Print detection statistics
# =========================
print("\n========== Province detected compounds ==========")

stack_df["Total"] = stack_df[level_order].sum(axis=1)

for province, row in stack_df.iterrows():
    print(
        f"{province}: "
        + " | ".join(
            [f"{level}={int(row[level])}" for level in level_order]
        )
        + f" | Total={int(row['Total'])}"
    )

# =====================================================
# Bar plot
# =====================================================
plt.figure(figsize=(10, 4))

if bar_mode == "total":
    # =========================
    # Total detected compound count
    # =========================
    plt.bar(
        stack_df.index,
        stack_df["Total"],
        color="#9CC6E4",
        edgecolor="black",
        linewidth=0.5,
        width=0.6
    )

elif bar_mode == "level":
    # =========================
    # Stacked bar plot by identification level
    # =========================
    colors = {
        "L1": "#E76F51",
        "L2a": "#9CC6E4",
        "L2b": "#6FB98F",
        "L3": "#F4A261"
    }

    bottom = None

    for level in level_order:
        values = stack_df[level].values

        plt.bar(
            stack_df.index,
            values,
            bottom=bottom,
            label=level,
            color=colors[level],
            edgecolor="black",
            linewidth=0.5,
            width=0.6
        )

        if bottom is None:
            bottom = values.copy()
        else:
            bottom += values

    # Uncomment to display the legend
    """
    plt.legend(
        title="Level",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    """

else:
    raise ValueError("bar_mode must be 'level' or 'total'")

# Keep the x-axis range consistent with the violin plot
plt.xlim(-0.5, len(province_order) - 0.5)

plt.xlabel("")
plt.ylabel("")

plt.tick_params(axis="y", labelleft=False)
plt.tick_params(axis="x", labelbottom=False)

plt.tight_layout()

plt.savefig(
    output_bar,
    dpi=600,
    transparent=True
)

plt.show()
plt.close()

print(f"Bar saved: {output_bar}")