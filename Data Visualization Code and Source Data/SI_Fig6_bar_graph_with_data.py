import pandas as pd
import matplotlib.pyplot as plt

categories = [
    "Pharmaceuticals",
    "Industry and consumer goods",
    "Transformation and metabolic products",
    "Food and other consumption",
    "Pesticides",
    "Personal care products"
]
# seafood 1679 1499 1285 163 135 66
# plasma 783 1632 1188 79 218 87
values = [783, 1632, 1188, 79, 218, 87]

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(
    range(len(categories)),
    values,
    width=0.65,
    alpha=0.7
)

# Use Matplotlib's default color cycle
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, bar in enumerate(bars):
    bar.set_color(colors[i % len(colors)])

# X-axis: keep tick marks but hide tick labels
ax.set_xticks(range(len(categories)))
ax.set_xticklabels([])
ax.tick_params(axis="x", which="both", labelbottom=False)

# Y-axis: specify tick positions, keep tick marks but hide tick labels
ax.set_ylim(0, 1800)
ax.set_yticks([0, 500, 1000, 1500, 2000])
ax.tick_params(axis="y", which="both", labelleft=False)

# Hide axis labels
ax.set_xlabel("")
ax.set_ylabel("")

# Remove the top and right spines
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