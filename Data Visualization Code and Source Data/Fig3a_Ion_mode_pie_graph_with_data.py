import matplotlib.pyplot as plt

values = [24, 56]
labels = [
    "Positive mode\n24 (30.0%)",
    "Negative mode\n56 (70.0%)"
]

colors = [
    "#FAADA5",
    "#A1C6DF"
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

# 去除背景
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