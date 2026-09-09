import matplotlib.pyplot as plt

values = [11, 107, 2050, 3258]
labels = ["1", "2", "3", "4"]

colors = [
    "#1F4E79",
    "#4F81BD", 
    "#7EA6D8", 
    "#B4C7E7"
]

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(3, 4),
    dpi=300,
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1]}
)

for ax in [ax1, ax2]:
    ax.bar(
        labels,
        values,
        color=colors,
        width=0.6
    )

ax1.set_ylim(120, 3320)
ax2.set_ylim(0, 120)

for ax in [ax1, ax2]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax1.set_yticks([1720, 3320])
ax2.set_yticks([0, 60])

for ax in [ax1, ax2]:
    ax.tick_params(
        axis="y",
        labelleft=False,
        length=4,
        width=1
    )

ax1.tick_params(
    axis="x",
    bottom=False,
    labelbottom=False
)

ax2.tick_params(
    axis="x",
    labelbottom=False,
    length=4,
    width=1
)

d = 0.015

kwargs = dict(
    color="black",
    clip_on=False,
    linewidth=1
)

ax1.plot(
    (-d, d),
    (-d, d),
    transform=ax1.transAxes,
    **kwargs
)

ax2.plot(
    (-d, d),
    (1-d, 1+d),
    transform=ax2.transAxes,
    **kwargs
)

fig.patch.set_alpha(0)
ax1.patch.set_alpha(0)
ax2.patch.set_alpha(0)

plt.tight_layout()

plt.savefig(
    r"C:\Users\13709\Desktop\bar_break.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

plt.show()