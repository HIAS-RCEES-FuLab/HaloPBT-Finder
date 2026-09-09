import pandas as pd
import matplotlib.pyplot as plt
import os

input_file = r"C:\Users\13709\Desktop\重要性分析\importance_summary_halogen_type_feature_mean.csv"

output_file = r"C:\Users\13709\Desktop\重要性分析\importance_halogen_feature_lineplot.png"


# =========================
# read data
# =========================
df = pd.read_csv(input_file)


# =========================
# Select halogen type: Cl or Br
# =========================
df = df[
    df["Halogen_group"]
    .astype(str)
    .str.startswith("Br")
]


# =========================
# Halogen ordering
# The last group represents >4 Br atoms or 6 Cl atoms
# =========================
halogen_order = [
    "Br_1",
    "Br_2",
    "Br_3",
    "Br_4",
    "Br_5",
]


df["Halogen_group"] = pd.Categorical(
    df["Halogen_group"],
    categories=halogen_order,
    ordered=True
)


# =========================
# Generate Type + Feature categories
# =========================
df["Category"] = (
    df["Type"].astype(str)
    + "_"
    + df["Feature"].fillna("").astype(str)
)


# =========================
# Plotting
# =========================
plt.figure(figsize=(6,4))

colors = {
    "Precursor_Exact_mass": "#0072B2",   
    "Precursor_KMD": "#56B4E9",          
    "MSMS_": "#009E73",          
    "Precursor_ion_mode": "#CC79A7",     
    "Precursor_iso": "#D55E00",                  
    "NeutralLoss_": "#E69F00"            
}


for category, sub in df.groupby("Category"):

    sub = sub.sort_values(
        "Halogen_group"
    )

    plt.plot(
        sub["Halogen_group"],
        sub["Mean_abs_importance"],
        marker="o",
        linewidth=3,
        markersize=8,
        label=category,
        color=colors.get(category, "#333333")
    )


plt.xticks(
    rotation=45,
    ha="right"
)

ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
""""""
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
    output_file,
    dpi=600,
    transparent=True,
    bbox_inches="tight"
)


plt.show()