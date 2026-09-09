import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# -----------------------------
china = gpd.read_file("china.json")

provinces_with_data = [
    '山东省', '广西壮族自治区', '福建省', '海南省', '广东省',
    '天津市', '浙江省', '上海市', '辽宁省', '河北省', '江苏省'
]

province_avg = pd.DataFrame({
    'Province': provinces_with_data,
    'Average_total': [1] * len(provinces_with_data)
})


# -----------------------------
china_with_data = china.merge(province_avg, left_on='name', right_on='Province', how='left')

fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
ax.set_aspect('equal')

china.plot(
    facecolor="#F0F8FF",
    edgecolor="white",
    linewidth=0.35,
    ax=ax,
    zorder=1
)

china_with_data.dropna(subset=['Average_total']).plot(
    facecolor='#ADD8E6',  
    edgecolor='#6495ED', 
    linewidth=0.2,
    ax=ax
)

china.boundary.plot(
    edgecolor='#555555', 
    linewidth=0.5,
    ax=ax,
    zorder=5
)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("black")
    spine.set_linewidth(0.8)
ax.set_xticks(np.arange(100, 131, 10))   # 100,110,120,130
ax.set_yticks(np.arange(16, 47, 10))     # 20,30,40
ax.tick_params(axis='both', which='both',
               labelbottom=False,
               labelleft=False)
plt.tight_layout()
ax.set_xlim(100, 130)
ax.set_ylim(16, 46)

plt.savefig(
    "china_province_city_map_shellfish.png",
    dpi=300,
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.05
)
plt.show()
