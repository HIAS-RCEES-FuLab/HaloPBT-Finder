import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

china = gpd.read_file("china.json")

province_dir = "province"

all_cities = []

for fname in os.listdir(province_dir):
    if fname.endswith(".json") or fname.endswith(".geojson"):
        try:
            gdf = gpd.read_file(os.path.join(province_dir, fname))
            all_cities.append(gdf)
        except Exception as e:
            print(fname, e)

cities = gpd.GeoDataFrame(
    pd.concat(all_cities, ignore_index=True),
    crs=all_cities[0].crs
)

provinces_with_data = [
    '山东省',
    '广西壮族自治区',
    '福建省',
    '海南省',
    '广东省',
    '天津市',
    '浙江省',
    '上海市',
    '辽宁省',
    '河北省',
    '江苏省'
]

province_avg = pd.DataFrame({
    "Province": provinces_with_data,
    "Average_total": 1
})

china = china.merge(
    province_avg,
    left_on="name",
    right_on="Province",
    how="left"
)

highlight_cities = [
    '滨海新区','浦东新区',
    '宁德市','漳州市','福州市','莆田市','泉州市',
    '汕头市','阳江市','茂名市','汕尾市','深圳市','珠海市',
    '北海市','防城港市',
    '秦皇岛市','唐山市',
    '海口市','东方市','三亚市','文昌市',
    '南通市','连云港市','盐城市',
    '营口市','锦州市','大连市','丹东市',
    '青岛市','烟台市','东营市','威海市','日照市',
    '宁波市','舟山市','台州市','温州市'
]

cities["highlight"] = cities["name"].isin(highlight_cities)
print(cities["name"])

fig, ax = plt.subplots(figsize=(6.5,5.5))
ax.set_aspect("equal")

china.plot(
    ax=ax,
    facecolor="#F0F8FF",
    edgecolor="white",
    linewidth=0.35,
    zorder=1
)

china.dropna(subset=["Average_total"]).plot(
    ax=ax,
    facecolor="#ADD8E6",
    edgecolor="#6495ED",
    linewidth=0.2,
    zorder=2
)

cities.plot(
    ax=ax,
    facecolor="none",
    edgecolor="gray",
    linewidth=0.15,
    zorder=3
)

highlight_city_gdf = cities[cities["highlight"]].copy()

points = highlight_city_gdf.representative_point()

ax.scatter(
    points.x,
    points.y,
    s=50,
    marker="v",
    color="#E76F51",
    edgecolors="black",
    linewidths=0.3,
    zorder=6
)

china.boundary.plot(
    ax=ax,
    edgecolor="#555555",
    linewidth=0.5,
    zorder=5
)

ax.set_xlim(100,130)
ax.set_ylim(16,46)

ax.set_xticks(np.arange(100,131,10))
ax.set_yticks(np.arange(16,47,10))

ax.tick_params(
    axis='both',
    which='both',
    labelbottom=False,
    labelleft=False
)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("black")
    spine.set_linewidth(0.8)

plt.tight_layout()

plt.savefig(
    "china_province_city_map.png",
    dpi=300,
    transparent=True,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.show()