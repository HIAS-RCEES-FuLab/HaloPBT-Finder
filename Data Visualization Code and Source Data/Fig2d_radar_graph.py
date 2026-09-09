import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 全局字体设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'

# 模型顺序
models = ['transformer','MLP','RF','SVM','KNN','LG']

# 第一组数据
data1 = pd.DataFrame({
    'Model': models,
    'test_acc_all': [0.8928,0.8458,0.8561,0.8579,0.8003,0.8182],
    'test_precision_all': [0.8451,0.7923,0.8648,0.6912,0.6066,0.6848],
    'test_recall_all': [0.9215,0.8879,0.9281,0.8964,0.7996,0.8097],
})

# 第二组数据
data2 = pd.DataFrame({
    'Model': models,
    'test_acc_all': [0.9985,0.9947,0.9182,0.7496,0.6093,0.7487],
    'test_precision_all': [0.9999,0.9999,0.9898,0.9235,0.8169,0.9405],
    'test_recall_all': [1,1,0.9971,0.95526,0.9064,0.9867],
})

# 合并成 8 维度
combined = pd.DataFrame()
for i, model in enumerate(models):
    combined_row = list(data1.loc[i, data1.columns[1:]]) + list(data2.loc[i, data2.columns[1:]])
    combined = pd.concat([combined, pd.DataFrame([combined_row])], ignore_index=True)

# 设置列名
combined.columns = [
    'acc1','pr1','roc1',
    'acc2','pr2','roc2'
]

# 雷达图设置
num_vars = len(combined.columns)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
ax.set_ylim(0.5, 1.0)

# 颜色
colors = ['#D3544E', '#7FB3D5', '#82E0AA', '#F7DC6F', '#F5CBA7', '#BB8FCE']

# ==============================
# ⭐ 先画其他模型
# ==============================
for j, model in enumerate(models):
    if model == 'transformer':
        continue

    values = combined.loc[j].tolist()
    values += values[:1]

    ax.plot(
        angles, values,
        linewidth=2.5,
        color=colors[j],
        alpha=0.8
    )

# ==============================
# ⭐ 最后画 transformer（自动在最上层）
# ==============================
i = models.index('transformer')
values = combined.loc[i].tolist()
values += values[:1]

ax.plot(
    angles, values,
    linewidth=3,   # 稍微加粗一点更突出
    color=colors[i],
    alpha=0.8,
    label='transformer'
)

# 去掉刻度标签，只保留刻度线（可选）
ax.set_xticks(angles[:-1])
ax.set_xticklabels([])  # 不显示角度文字
ax.set_yticks(np.linspace(0.4, 1, 4))
ax.set_yticklabels([])  # 不显示半径文字

# 背景透明
plt.gcf().patch.set_alpha(0.0)
plt.gca().patch.set_alpha(0.0)

plt.tight_layout()
plt.savefig('model_8dim_radar_no_labels.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
