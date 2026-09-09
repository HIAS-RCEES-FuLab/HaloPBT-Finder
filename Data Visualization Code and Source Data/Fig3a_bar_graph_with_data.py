import matplotlib.pyplot as plt

values = [12, 80]
labels = ["Positive mode", "Negative mode"]

colors = [
    "#7FA8D1",
    "#EFA18B"
]

fig, ax = plt.subplots(figsize=(2, 3), dpi=300)

ax.bar(
    labels,
    values,
    color=colors,
    width=0.6
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="y",
    labelleft=False,
    length=4
)

ax.tick_params(
    axis="x",
    labelbottom=False,
    length=4
)
ax.set_yticks([0, 20, 40, 60, 80])
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

plt.tight_layout()

plt.savefig(
    r"C:\Users\13709\Desktop\Ion_mode_bar.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

plt.show()