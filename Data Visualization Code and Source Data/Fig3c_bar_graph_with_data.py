import matplotlib.pyplot as plt
# SRM 1957 5 11 158 504
# SRM 2974a 8 51 279 574
values = [8, 51, 279, 574]
labels = ["1", "2", "3", "4"]

colors = [
    "#7FA8D1",
    "#A8D08D",
    "#EFA18B",
    "#C5A3D6"
]

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(4, 3),
    dpi=300,
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1]}
)

# =========================
# Plotting
# =========================
for ax in [ax1, ax2]:
    ax.bar(
        labels,
        values,
        color=colors,
        width=0.6
    )

ax1.set_ylim(75, 550)
ax2.set_ylim(0, 75)

for ax in [ax1, ax2]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax1.set_yticks([200, 400, 600])
ax2.set_yticks([0, 25, 50])

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

# ax1
ax1.plot(
    (-d, d),
    (-d, d),
    transform=ax1.transAxes,
    **kwargs
)

# ax2
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
    r"C:\Users\13709\Desktop\Ion_mode_bar_break.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

plt.show()