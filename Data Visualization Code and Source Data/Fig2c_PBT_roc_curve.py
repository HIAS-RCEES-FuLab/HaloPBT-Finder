import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================================
# This script visualizes the ROC curves for PBT prediction
# using four different models: MLP (bin=1, 0.1, 0.01) and
# Transformer (encoded features).
# The script reads precomputed ROC curves from CSV files for
# five cross-validation folds, computes mean ROC curves,
# AUC statistics (mean, std, min, max), and plots the full
# ROC curves with a publication-style format.
# It saves the figure as a transparent PNG for inclusion in papers.
# =====================================================

# ==============================
# Global font settings
# ==============================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# Model file paths and patterns
# ==============================
model_paths = {
    'MLP_bin_1_PBT': r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PR_ROC_PBT_bin\1",
    'MLP_bin_0.1_PBT': r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PR_ROC_PBT_bin\0.1",
    'MLP_bin_0.01_PBT': r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PR_ROC_PBT_bin\0.01",
    'Transformer_encode_PBT': r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PR_ROC_PBT_encode"
}

file_patterns = {
    'MLP_bin_1_PBT': "multi_PBT_1_0122_all_fold{}_roc_curve.csv",
    'MLP_bin_0.1_PBT': "multi_PBT_0.1_0122_all_fold{}_roc_curve.csv",
    'MLP_bin_0.01_PBT': "multi_PBT_0.01_0122_all_fold{}_roc_curve.csv",
    'Transformer_encode_PBT': "multi_PBT_transformer_0.01_0119_all_fold{}_roc_curve.csv"
}

# ==============================
# Store all model data
# ==============================
all_models_data = {}

# Color and linestyle settings
model_colors = {
    'MLP_bin_1_PBT': '#B39ED5',
    'MLP_bin_0.1_PBT': '#FBD49C',
    'MLP_bin_0.01_PBT': '#FDB7AE',
    'Transformer_encode_PBT': '#87BBDE'
}

model_linestyles = {
    'MLP_bin_1_PBT': '-',
    'MLP_bin_0.1_PBT': '-',
    'MLP_bin_0.01_PBT': '-',
    'Transformer_encode_PBT': '-'
}

model_display_names = {
    'MLP_bin_1_PBT': 'MLP (bin=1)',
    'MLP_bin_0.1_PBT': 'MLP (bin=0.1)',
    'MLP_bin_0.01_PBT': 'MLP (bin=0.01)',
    'Transformer_encode_PBT': 'Transformer (encode)'
}

# ==============================
# Process each model and compute mean ROC
# ==============================
for model_name, base_dir in model_paths.items():
    print(f"\n{'='*60}")
    print(f"Processing model: {model_display_names[model_name]}")
    print(f"Path: {base_dir}")
    print('='*60)

    file_pattern = file_patterns[model_name]
    roc_files = []

    # Find ROC curve files for 5 folds
    for fold in range(1, 6):
        file_path = os.path.join(base_dir, file_pattern.format(fold))
        if os.path.exists(file_path):
            roc_files.append((fold, file_path))

    if not roc_files:
        print(f"Warning: No ROC curve files found for {model_display_names[model_name]}")
        continue

    all_fprs, all_tprs, fold_aucs = [], [], []
    for fold, file_path in roc_files:
        df = pd.read_csv(file_path)
        if 'fpr' in df.columns and 'tpr' in df.columns:
            fpr = df['fpr'].values
            tpr = df['tpr'].values
            sorted_idx = np.argsort(fpr)
            fpr_sorted = fpr[sorted_idx]
            tpr_sorted = tpr[sorted_idx]
            auc_value = df['auc'].iloc[0] if 'auc' in df.columns else np.trapz(tpr_sorted, fpr_sorted)
            fold_aucs.append(auc_value)
            all_fprs.append(fpr_sorted)
            all_tprs.append(tpr_sorted)

    if not fold_aucs:
        continue

    fpr_grid = np.linspace(0, 1, 101)
    interpolated_tprs = []
    for fpr_curve, tpr_curve in zip(all_fprs, all_tprs):
        fpr_ext = np.concatenate([[0], fpr_curve]) if fpr_curve[0]>0 else fpr_curve
        tpr_ext = np.concatenate([[0], tpr_curve]) if fpr_curve[0]>0 else tpr_curve
        if fpr_ext[-1]<1.0:
            fpr_ext = np.append(fpr_ext,1.0)
            tpr_ext = np.append(tpr_ext,1.0)
        interpolated_tprs.append(np.interp(fpr_grid, fpr_ext, tpr_ext, left=0.0, right=1.0))
    interpolated_tprs = np.array(interpolated_tprs)

    all_models_data[model_name] = {
        'fpr_grid': fpr_grid,
        'mean_tpr': np.mean(interpolated_tprs, axis=0),
        'avg_auc': np.mean(fold_aucs),
        'std_auc': np.std(fold_aucs),
        'min_auc': np.min(fold_aucs),
        'max_auc': np.max(fold_aucs),
        'fold_aucs': fold_aucs,
        'num_folds': len(fold_aucs),
        'display_name': model_display_names[model_name]
    }

# ==============================
# Plot full ROC curves (publication-style)
# ==============================
if all_models_data:
    draw_order = ['MLP_bin_0.01_PBT', 'MLP_bin_0.1_PBT', 'MLP_bin_1_PBT', 'Transformer_encode_PBT']
    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot([0,1],[0,1], color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    for model_name in draw_order:
        if model_name not in all_models_data:
            continue
        data = all_models_data[model_name]
        ax.plot(data['fpr_grid'], data['mean_tpr'], color=model_colors[model_name],
                linestyle=model_linestyles[model_name], linewidth=2.5, alpha=1.0)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()

    output_dir = r"C:\Users\13709\Desktop\论文撰写四\模型\测试"
    output_file = os.path.join(output_dir, "pbt_all_models_roc_curves_full_view_no_offset.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Full ROC curve figure saved to: {output_file}")
    plt.show()
else:
    print("No valid model data found.")