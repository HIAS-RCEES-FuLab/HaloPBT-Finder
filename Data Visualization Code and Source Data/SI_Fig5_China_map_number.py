import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors

china = gpd.read_file("china.json")

province_value_dict = {
    "福建省": 4420,      # FJ
    "广东省": 4419,      # GD
    "广西壮族自治区": 4539,  # GX
    "河北省": 4536,      # HB
    "海南省": 4607,      # HN
    "江苏省": 4605,      # JS
    "辽宁省": 4266,      # LN
    "山东省": 4857,      # SD
    "上海市": 4382,      # SH
    "天津市": 3876,      # TJ
    "浙江省": 5009       # ZJ
}

province_avg = pd.DataFrame({
    "Province": list(province_value_dict.keys()),
    "Average_total": list(province_value_dict.values())
})

china_with_data = china.merge(
    province_avg,
    left_on='name',
    right_on='Province',
    how='left'
)
vmin = china_with_data['Average_total'].min()
vmax = china_with_data['Average_total'].max()

province_dir = "province"
all_cities_gdf = []

for fname in os.listdir(province_dir):
    if fname.endswith(".json") or fname.endswith(".geojson"):
        fpath = os.path.join(province_dir, fname)
        try:
            cities_gdf = gpd.read_file(fpath)
            all_cities_gdf.append(cities_gdf)
        except Exception as e:
            print(f" {fname}: {e}")


cities = gpd.GeoDataFrame(pd.concat(all_cities_gdf, ignore_index=True))

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_aspect('equal')

china_with_data.dropna(subset=['Average_total']).plot(
    column='Average_total',
    cmap = mcolors.LinearSegmentedColormap.from_list("softred_to_red", ["#FDEAEA", "#B22222"]),       
    linewidth=0.3,
    edgecolor='gray',
    #legend=True,
    ax=ax
)

cities.plot(
    facecolor='none',
    edgecolor='gray',
    linewidth=0.15,
    ax=ax,
    zorder=4
)

china.boundary.plot(
    edgecolor='black',
    linewidth=0.3,
    ax=ax,
    zorder=5
)

plt.axis('off')
plt.tight_layout()
ax.set_xlim(73, 135)
ax.set_ylim(18, 54)
#ax.set_xlim(100, 125)
#ax.set_ylim(3, 25)

plt.savefig(
    "china_province_city_map_level1.png",
    dpi=300,
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.05
)

plt.show()

fig_cb, ax_cb = plt.subplots(figsize=(0.6, 6))  
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = mpl.cm.YlOrRd

cb = mpl.colorbar.ColorbarBase(
    ax_cb,
    cmap=mcolors.LinearSegmentedColormap.from_list("softred_to_red", ["#FDEAEA", "#B22222"]), 
    norm=norm,
    orientation='vertical',
    ticks=[]
)

cb.ax.set_yticklabels([])

cb.set_label("")

plt.tight_layout()
plt.savefig(
    "china_province_colorbar_clean_level1.png",
    dpi=300,
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.05
)
plt.show()