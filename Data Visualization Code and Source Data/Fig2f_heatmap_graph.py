import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input_path = r"C:\Users\13709\Desktop\重要性分析\feature_importance_mean.csv"
output_path = r"C:\Users\13709\Desktop\重要性分析\Br_iso_heatmap.png"

df = pd.read_csv(input_path)

# =========================
# Change type: Cl or Br
# Halogen ordering
# The last group represents >4 Br atoms or 6 Cl atoms
# =========================
cl_order = [
    "Br_1","Br_2","Br_3","Br_4",
    "Br_5"
]

records = []

for cl in cl_order:
    sub = df[
        df["File"]
        .astype(str)
        .str.contains(cl)
    ]
    for iso in range(7):
        iso_sub = sub[
            sub["Feature"]
            .astype(str)
            .str.contains(f"M{iso}")
        ]
        value = (
            iso_sub["Mean_abs_importance"].mean()
            if len(iso_sub) > 0
            else 0
        )
        records.append({
            "Br": cl,
            "iso": f"M{iso}",
            "Importance": value
        })


heatmap_df = pd.DataFrame(records).pivot(
    index="Br",
    columns="iso",
    values="Importance"
)

heatmap_df = heatmap_df.loc[
    cl_order,
    [f"M{i}" for i in range(7)]
]


# =========================
# normalization 0-100
# =========================
heatmap_df = (
    heatmap_df
    .div(
        heatmap_df.max(axis=1),
        axis=0
    )
    * 100
)

heatmap_df = heatmap_df.fillna(0)


# =========================
# plotting
# =========================
fig, ax = plt.subplots(
    figsize=(4,4)
)

im = ax.imshow(
    heatmap_df.values,
    aspect="auto",
    cmap="coolwarm",
    vmin=0,
    vmax=100,
    alpha=0.8
)

ax.set_xticks(
    range(len(heatmap_df.columns))
)

ax.set_xticklabels(
    heatmap_df.columns
)


ax.set_yticks(
    range(len(heatmap_df.index))
)

ax.set_yticklabels(
    heatmap_df.index
)

ax.set_xlabel("")
ax.set_ylabel("")

ax.tick_params(
    axis="y",
    labelleft=False,
    left=True
)

ax.tick_params(
    axis="x",
    labelbottom=False,
    bottom=True
)

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=600,
    transparent=True,
    bbox_inches="tight"
)

plt.show()