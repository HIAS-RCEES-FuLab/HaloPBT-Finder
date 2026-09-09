import matplotlib.pyplot as plt
# SRM 1957 110 568
# SRM 2974a 159 753
values = [159, 753]
labels = [
    "1",
    "0"
]

colors = [
    "#E07A75",
    "#BDBDBD",
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
    r"C:\Users\13709\Desktop\SRM_PBT.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

plt.show()