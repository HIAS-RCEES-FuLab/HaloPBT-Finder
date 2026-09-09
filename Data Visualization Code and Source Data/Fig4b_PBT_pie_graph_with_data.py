import matplotlib.pyplot as plt

# 数据
values = [5007, 419]

# 突出显示（略微分离）
explode = [0.04, 0.04]

colors = ['#C6C7E2', '#E89A9A']

plt.figure(figsize=(6,6))
ax = plt.gca()

# 创建饼图（不显示文字）
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
plt.savefig("pie_chart_PBT_brown.png", dpi=300, transparent=True)

plt.show()