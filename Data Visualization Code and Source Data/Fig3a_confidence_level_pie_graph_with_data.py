import matplotlib.pyplot as plt

values = [14, 12, 54]
labels = [
    "Mode 3\n14",
    "Mode 4\n12",
    "Mode 5\n54"
]

colors = [
    "#F4C97B",
    "#B7D7A8",
    "#C6A4D9"
]

fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

ax.pie(
    values,
    colors=colors,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(
        width=0.5,
        edgecolor="white"
    ),
    textprops=dict(
        fontsize=10
    )
)

ax.set_aspect("equal")

fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

plt.tight_layout()

plt.savefig(
    r"C:\Users\13709\Desktop\Ion_mode_donut.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

plt.show()