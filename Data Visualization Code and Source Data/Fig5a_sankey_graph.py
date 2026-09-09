import pandas as pd
import plotly.graph_objects as go

path = r"C:\Users\13709\Desktop\论文撰写四\sankey\sankey_final_label.csv"

df = pd.read_csv(path, low_memory=False)

# ======================
# 数据划分
# ======================

seafood = df[df["Seafood_Label"] == 1].copy()
plasma = df[df["Plasma_Label"] == 1].copy()

shared = df[(df["Seafood_Label"] == 1) & (df["Plasma_Label"] == 1)]
seafood_specific = df[(df["Seafood_Label"] == 1) & (df["Plasma_Label"] == 0)]
plasma_specific = df[(df["Seafood_Label"] == 0) & (df["Plasma_Label"] == 1)]

print("Total:", len(df))
print("Shared:", len(shared))
print("Seafood-specific:", len(seafood_specific))
print("Plasma-specific:", len(plasma_specific))


# ======================
# 节点
# ======================

middle_nodes = [
    "Seafood-specific PBTs",
    "Shared PBTs",
    "Plasma-specific PBTs"
]

type_order = [
    "Industry and consumer goods",
    "Pharmaceuticals",
    "Transformation and metabolic products",
    "Pesticides",
    "Food and other consumption",
    "Personal care products"
]

type_nodes = [
    x for x in type_order
    if x in df["Pollutant_Type"].dropna().unique()
]


nodes = ["Total PBTs"] + middle_nodes + type_nodes

node_dict = {n:i for i,n in enumerate(nodes)}

sources = []
targets = []
values = []


# ======================
# 第一层 -> 第二层
# ======================

sources.extend([
    node_dict["Total PBTs"],
    node_dict["Total PBTs"],
    node_dict["Total PBTs"]
])

targets.extend([
    node_dict["Seafood-specific PBTs"],
    node_dict["Shared PBTs"],
    node_dict["Plasma-specific PBTs"]
])

values.extend([
    len(seafood_specific),
    len(shared),
    len(plasma_specific)
])


# ======================
# 第二层 -> 第三层
# ======================

groups = {
    "Seafood-specific PBTs": seafood_specific,
    "Shared PBTs": shared,
    "Plasma-specific PBTs": plasma_specific
}


for name, temp in groups.items():

    for pollutant in type_order:

        count = (temp["Pollutant_Type"] == pollutant).sum()

        if count > 0:
            sources.append(node_dict[name])
            targets.append(node_dict[pollutant])
            values.append(count)



# ======================
# 分类颜色
# ======================

category_colors = {
    "Industry and consumer goods": "#ff7f0e",
    "Pharmaceuticals": "#1f77b4",
    "Transformation and metabolic products": "#2ca02c",
    "Pesticides": "#9467bd",
    "Food and other consumption": "#d62728",
    "Personal care products": "#8c564b"
}


def hex_to_rgba(hex_color, alpha=0.25):

    hex_color = hex_color.replace("#","")

    r = int(hex_color[0:2],16)
    g = int(hex_color[2:4],16)
    b = int(hex_color[4:6],16)

    return f"rgba({r},{g},{b},{alpha})"



# ======================
# 节点颜色
# ======================

node_colors = []

for n in nodes:

    if n in category_colors:
        node_colors.append(
            hex_to_rgba(category_colors[n],0.5)
        )

    elif n == "Total PBTs":
        node_colors.append("#4C78A8")

    elif n == "Shared PBTs":
        node_colors.append("#BDBDBD")

    elif n == "Seafood-specific PBTs":
        node_colors.append("#A9CDEB")

    elif n == "Plasma-specific PBTs":
        node_colors.append("#F4B6B6")

    else:
        node_colors.append("#CCCCCC")



# ======================
# 连线颜色
# ======================

link_colors = []

for s,t in zip(sources,targets):

    source_name = nodes[s]
    target_name = nodes[t]


    # Total -> Seafood-specific
    if target_name == "Seafood-specific PBTs":

        link_colors.append(
            "rgba(169,205,235,0.25)"
        )


    # Total -> Shared

    elif target_name == "Shared PBTs":

        link_colors.append(
            "rgba(189,189,189,0.25)"
        )


    # Total -> Plasma-specific

    elif target_name == "Plasma-specific PBTs":

        link_colors.append(
            "rgba(244,182,182,0.25)"
        )


    # 分类

    elif target_name in category_colors:

        link_colors.append(
            hex_to_rgba(category_colors[target_name],0.25)
        )


    else:

        link_colors.append(
            "rgba(150,150,150,0.2)"
        )



# ======================
# Sankey
# ======================

fig = go.Figure(
    go.Sankey(
        arrangement="snap",

        node=dict(
            label=[""]*len(nodes),
            pad=20,
            thickness=25,
            color=node_colors,
            hovertemplate=""
        ),

        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate=""
        )
    )
)


fig.update_layout(
    width=1100,
    height=800,
    margin=dict(l=20,r=20,t=20,b=20)
)


fig.show()