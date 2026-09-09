import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =====================================================
# This script visualizes the test accuracy comparison between
# four different model types (bin_0.01, bin_0.1, bin_1, Transformer)
# for Cl/Br prediction (left) and PBT prediction (right).
# Each bar represents the average accuracy over five cross-validation folds.
# The dashed line separates the two prediction tasks for clarity.
# =====================================================

# -----------------------------
# Load data
# -----------------------------
cl_br_file_path = r"C:\Users\13709\Desktop\论文撰写四\模型\测试\Cl_Br.xlsx"
pbt_file_path = r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PBT.xlsx"

cl_br_df = pd.read_excel(cl_br_file_path)
pbt_df = pd.read_excel(pbt_file_path)

# Rows to extract
rows_to_extract = [
    'bin_0.01_1', 'bin_0.01_2', 'bin_0.01_3', 'bin_0.01_4', 'bin_0.01_5',
    'bin_0.1_1', 'bin_0.1_2', 'bin_0.1_3', 'bin_0.1_4', 'bin_0.1_5',
    'bin_1_1', 'bin_1_2', 'bin_1_3', 'bin_1_4', 'bin_1_5',
    'transformer_1', 'transformer_2', 'transformer_3', 'transformer_4', 'transformer_5'
]

# Set index
cl_br_df.set_index(cl_br_df.columns[0], inplace=True)
pbt_df.set_index(pbt_df.columns[0], inplace=True)

# Extract test accuracy
cl_br_data = cl_br_df.loc[rows_to_extract, ['test_acc_all']]
pbt_data = pbt_df.loc[rows_to_extract, ['test_acc_all']]

# -----------------------------
# Compute 5-fold average per model type
# -----------------------------
def calculate_model_means(data, model_prefix):
    model_rows = [row for row in data.index if row.startswith(model_prefix)]
    return data.loc[model_rows, 'test_acc_all'].mean()

# Cl/Br
clbr_values = [
    calculate_model_means(cl_br_data, 'bin_0.01'),
    calculate_model_means(cl_br_data, 'bin_0.1'),
    calculate_model_means(cl_br_data, 'bin_1'),
    calculate_model_means(cl_br_data, 'transformer')
]

# PBT
pbt_values = [
    calculate_model_means(pbt_data, 'bin_0.01'),
    calculate_model_means(pbt_data, 'bin_0.1'),
    calculate_model_means(pbt_data, 'bin_1'),
    calculate_model_means(pbt_data, 'transformer')
]

# -----------------------------
# Prepare plot
# -----------------------------
x_clbr = np.arange(4)
x_pbt = np.arange(4) + 5
width = 0.6

color_map = {"ClBr": "#80b1d3", "PBT": "#fb8072"}

fig, ax_clbr = plt.subplots(figsize=(8, 6))
ax_pbt = ax_clbr.twinx()

# Draw Cl/Br bars
for i, val in enumerate(clbr_values):
    ax_clbr.bar(x_clbr[i], val, width, color=color_map["ClBr"], edgecolor='black', alpha=0.8)

# Draw PBT bars
for i, val in enumerate(pbt_values):
    ax_pbt.bar(x_pbt[i], val, width, color=color_map["PBT"], edgecolor='black', alpha=0.8)

# X-axis ticks
all_x_ticks = list(x_clbr) + list(x_pbt)
ax_clbr.set_xticks(all_x_ticks)
ax_clbr.set_xticklabels([''] * len(all_x_ticks))  # hide all labels

# Area separator
mid_point = (x_clbr[-1] + x_pbt[0]) / 2
ax_clbr.axvline(x=mid_point, color='gray', linestyle='--', linewidth=1, alpha=0.8)

# Y-axis settings
ax_clbr.set_ylim(0.9, 1)
ax_pbt.set_ylim(0.6, 0.9)

# Hide y-axis ticks
ax_clbr.set_yticklabels([])
ax_pbt.set_yticklabels([])

# Borders style (publication style)
for spine in ['top']:
    ax_clbr.spines[spine].set_visible(False)
    ax_pbt.spines[spine].set_visible(False)
ax_clbr.spines['left'].set_linewidth(1)
ax_clbr.spines['left'].set_color(color_map["ClBr"])
ax_clbr.spines['bottom'].set_linewidth(1)
ax_pbt.spines['right'].set_linewidth(1)
ax_pbt.spines['right'].set_color(color_map["PBT"])

# -----------------------------
# Add model type labels (English)
# -----------------------------
model_labels = ['bin_0.01', 'bin_0.1', 'bin_1', 'Transformer']
for i, label in enumerate(model_labels):
    ax_clbr.text(x_clbr[i], 0.895, label, ha='center', va='top', fontsize=10)
    ax_clbr.text(x_pbt[i], 0.595, label, ha='center', va='top', fontsize=10)

# Add region annotations
ax_clbr.text((x_clbr[0]+x_clbr[-1])/2, 1.01, 'Cl/Br Prediction', ha='center', va='bottom', fontsize=11, fontweight='bold', color=color_map["ClBr"])
ax_clbr.text((x_pbt[0]+x_pbt[-1])/2, 0.92, 'PBT Prediction', ha='center', va='bottom', fontsize=11, fontweight='bold', color=color_map["PBT"])

# Adjust layout
plt.tight_layout()

# Save figure
output_path = r"C:\Users\13709\Desktop\论文撰写四\模型\测试\ClBr_PBT_comparison_english.png"
plt.savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')
print(f"Figure saved at: {output_path}")

plt.show()