import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

country_data = {
    'United States of America': 53,
    'China': 103,
    'Spain': 2,
    'Austria': 4,
    'Canada': 9,
    'United Arab Emirates': 16,
    'India': 19,
    'Portugal': 3,
    'Germany': 8,
    'Finland': 2,
    'Norway': 2,
    'Italy': 8,
}

world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
world = world[world['NAME'] != 'Antarctica']
pd.set_option('display.max_rows', None)
world['value'] = world['NAME'].map(country_data).fillna(0)
print(world['NAME'])

# -----------------------------
# log scale
# -----------------------------
log_values = world['value'].copy()
log_values_nonzero = log_values[log_values > 0].apply(np.log10)

log_min = log_values_nonzero.min()
log_max = log_values_nonzero.max()

# -----------------------------
# colormap
# -----------------------------
cmap = mpl.cm.Reds
norm = mpl.colors.Normalize(vmin=log_min, vmax=log_max)

def color_func(val):
    if val == 0:
        return mpl.colors.to_rgba('#ffffff')
    else:
        return cmap(norm(np.log10(val)))

# -----------------------------
# plot
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 6))

world.plot(
    ax=ax,
    color=[color_func(v) for v in world['value']],
    linewidth=0.2,
    edgecolor='black'
)

ax.set_axis_off()

# -----------------------------
# save
# -----------------------------
plt.savefig(r"C:\Users\13709\Desktop\world_map_log_gradient_red.png",
            dpi=300, transparent=True, bbox_inches='tight')

plt.show()