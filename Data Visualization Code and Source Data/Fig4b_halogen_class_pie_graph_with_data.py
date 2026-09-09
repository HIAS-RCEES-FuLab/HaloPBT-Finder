import matplotlib.pyplot as plt

# 数据
values = [4929, 361, 136]

# 突出显示所有切片
explode = [0.04, 0.04, 0.04]  # 每个切片都略微突出

# 颜色（柔和渐变）
colors = ['#FFE699', '#FFD29B', '#C5E0B4']  # 绿-浅橙-黄

plt.figure(figsize=(6,6))
ax = plt.gca()

# 创建饼图
ax.pie(
    values,
    startangle=90,
    colors=colors,
    explode=explode,
    shadow=True,
    labels=None,
    autopct=None
)

# 保持圆形
plt.axis('equal')

# 保存为透明 PNG
plt.tight_layout()
plt.savefig("pie_chart_IDresult.png", dpi=300, transparent=True)

plt.show()