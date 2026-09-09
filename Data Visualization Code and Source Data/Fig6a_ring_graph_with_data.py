import matplotlib.pyplot as plt

# Values assigned according to country
values = [8, 7, 6, 0, 0, 0]

labels = [
    "Industry and consumer goods",
    "Pharmaceuticals",
    "Transformation and metabolic products",
    "Food and other consumption",
    "Personal care products",
    "Pesticides"
]

colors = [
    "#CD6A67",
    "#6FA8DC",
    "#FFD166",
    "#9079B5",
    "#F09255",
    "#83B85C"
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
    )
)

ax.set_aspect("equal")

fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

plt.tight_layout()

plt.savefig(
    r"C:\Users\13709\Desktop\Pollutant_Type_donut.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

plt.show()