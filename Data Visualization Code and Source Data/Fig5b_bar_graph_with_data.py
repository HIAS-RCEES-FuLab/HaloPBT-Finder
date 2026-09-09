import pandas as pd
import matplotlib.pyplot as plt

categories = [
    "Industry and consumer goods",
    "Pharmaceuticals",
    "Transformation and metabolic products",
    "Pesticides",
    "Food and other consumption",
    "Personal care products"
]
values = [25, 18, 11, 2, 1, 1]

category_colors = {
    "Pharmaceuticals": "#1f77b4",
    "Industry and consumer goods": "#ff7f0e",
    "Transformation and metabolic products": "#2ca02c",
    "Food and other consumption": "#d62728",
    "Pesticides": "#9467bd",
    "Personal care products": "#8c564b"
}

colors = [category_colors[c] for c in categories]

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(
    range(len(categories)),
    values,
    width=0.65,
    alpha=0.6,
    color=colors
)

ax.set_xticks(range(len(categories)))
ax.set_xticklabels([])
ax.tick_params(axis="x", which="both", labelbottom=False)

ax.set_ylim(0, 25)
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.tick_params(axis="y", which="both", labelleft=False)

ax.set_xlabel("")
ax.set_ylabel("")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    r"C:\Users\13709\Desktop\PBT_use_category_bar.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

plt.show()

for category in categories:
    print(f"{category}: {category_colors[category]}")