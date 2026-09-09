import matplotlib.pyplot as plt

# The data has already been preprocessed and counted, so we directly assign values for plotting
sizes = [337058, 436057]  # no Cl/Br, with Cl/Br
colors = ["#FFE699", "#C5E0B4"]  # no Cl/Br: yellow, with Cl/Br: green
explode = (0.1, 0)  # emphasize the 'with Cl/Br' slice

# Create pie chart
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(
    sizes,
    labels=None,       # no labels
    autopct=None,      # no percentage
    startangle=90,     # start from top
    colors=colors,
    explode=explode,   # emphasize 'with Cl/Br'
    shadow=True
)

plt.tight_layout()

# Save figure with transparent background
fig.savefig("ClBr_pie.png", dpi=300, transparent=True)
plt.close(fig)