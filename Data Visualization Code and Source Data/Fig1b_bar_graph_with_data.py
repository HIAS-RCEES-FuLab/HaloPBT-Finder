import matplotlib.pyplot as plt
import numpy as np

# The data has already been preprocessed and counted, so we directly assign values for plotting
# x-axis positions: POS / NEG
x = np.arange(2)

width = 0.25   # bar width
gap = 0.15     # gap between non-PBT and PBT bars ⭐

# Data: [non-PBT, PBT]
pos_values = [1053, 374]
neg_values = [479, 99]

# Colors
color_map = {
    "non-PBT": "#80b1d3",
    "PBT": "#fb8072"
}

# Figure and twin y-axes
fig, ax_pos = plt.subplots(figsize=(6, 4))
ax_neg = ax_pos.twinx()

# POS bars (left y-axis)
ax_pos.bar(
    x[0] - (width/2 + gap/2),
    pos_values[0],
    width,
    color=color_map["non-PBT"]
)
ax_pos.bar(
    x[0] + (width/2 + gap/2),
    pos_values[1],
    width,
    color=color_map["PBT"]
)

# NEG bars (right y-axis)
ax_neg.bar(
    x[1] - (width/2 + gap/2),
    neg_values[0],
    width,
    color=color_map["non-PBT"]
)
ax_neg.bar(
    x[1] + (width/2 + gap/2),
    neg_values[1],
    width,
    color=color_map["PBT"]
)

# x-axis labels (hidden)
ax_pos.set_xticks(x)
ax_pos.set_xticklabels([])

# y-axis ticks (customized, hidden labels for cleaner figure)
ax_pos.set_yticks([0, 500, 1000])
ax_neg.set_yticks([0, 200, 400])
ax_pos.set_yticklabels([])
ax_neg.set_yticklabels([])

# Border styling for publication style
ax_pos.spines['top'].set_visible(False)
ax_neg.spines['top'].set_visible(False)
ax_pos.spines['left'].set_linewidth(1.5)
ax_pos.spines['bottom'].set_linewidth(1.5)
ax_neg.spines['right'].set_linewidth(1.5)

plt.tight_layout()

# Save figure with transparent background
plt.savefig("POS_NEG_barplot.png", dpi=300, transparent=True)
plt.show()