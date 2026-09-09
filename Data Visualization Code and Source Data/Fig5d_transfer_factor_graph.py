import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file1 = r"C:\Users\13709\Desktop\论文撰写四\ZJ_YUHUAN_PBT\Shared_PBT_information_matched_with_Mean.csv"

df = pd.read_csv(file1, low_memory=False)

if "Factor" not in df.columns:
    raise ValueError("The file does not contain a 'Factor' column")

# Convert to numeric
df["Factor"] = pd.to_numeric(df["Factor"], errors="coerce")

# Remove missing values and values <= 0
df = df[df["Factor"] > 0].copy()

# Sort by Factor in descending order
df = df.sort_values("Factor", ascending=False).reset_index(drop=True)

# Log10 transformation
df["log10_Factor"] = np.log10(df["Factor"])

# Color: light red for Factor > 1, light blue for Factor < 1, gray for Factor = 1
colors = np.where(
    df["Factor"] > 1,
    "#E89A9A",
    np.where(df["Factor"] < 1, "#9FC5E3", "#999999")
)

# Plot
fig, ax = plt.subplots(figsize=(6, 3))

ax.bar(
    range(len(df)),
    df["log10_Factor"],
    color=colors,
    width=0.8
)

# Factor = 1 corresponds to log10(Factor) = 0
ax.axhline(
    y=0,
    color="gray",
    linewidth=1.5
)

# Hide x-axis tick labels
ax.set_xticks([])
ax.tick_params(axis="x", which="both", labelbottom=False)

# Keep y-axis tick marks but hide labels
ax.tick_params(axis="y", which="both", labelleft=False)

# Hide top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel("")
ax.set_ylabel("")

plt.tight_layout()

plt.savefig(
    r"C:\Users\13709\Desktop\Factor_log10_bar.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

plt.show()