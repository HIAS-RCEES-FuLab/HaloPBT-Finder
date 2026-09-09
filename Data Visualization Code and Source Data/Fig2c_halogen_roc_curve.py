import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================================
# This script visualizes the ROC curves for Cl/Br prediction
# using four different models: MLP (bin=1, 0.1, 0.01) and
# Transformer (encoded features).
# The script reads precomputed ROC curves from CSV files for
# five cross-validation folds, computes mean ROC curves,
# AUC statistics (mean, std, min, max), and plots both the
# full ROC curves and zoomed-in ROC curves at low FPR/high TPR.
# It also saves all statistics to a CSV file for comparison.
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
    'MLP_bin_1': r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PR_ROC_Cl_Br_bin\1",
    'MLP_bin_0.1': r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PR_ROC_Cl_Br_bin\0.1",
    'MLP_bin_0.01': r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PR_ROC_Cl_Br_bin\0.01",
    'Transformer_encode': r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PR_ROC_Cl_Br_encode"
}

file_patterns = {
    'MLP_bin_1': "multi_cl_br_1_0122_fold{}_combined_roc_curves.csv",
    'MLP_bin_0.1': "multi_cl_br_0.1_0118_fold{}_combined_roc_curves.csv",
    'MLP_bin_0.01': "multi_cl_br_0.01_0120_fold{}_combined_roc_curves.csv",
    'Transformer_encode': "multi_cl_br_transformer_0.01_0122_fold{}_combined_roc_curves.csv"
}

# ==============================
# Store all model data
# ==============================
all_models_data = {}

# Color and linestyle settings
model_colors = {
    'MLP_bin_1': '#87BBDE',
    'MLP_bin_0.1': '#B39ED5',
    'MLP_bin_0.01': '#FBD49C',
    'Transformer_encode': '#FDB7AE'
}

model_linestyles = {
    'MLP_bin_1': '-',
    'MLP_bin_0.1': '-',
    'MLP_bin_0.01': '-',
    'Transformer_encode': '-'
}

model_display_names = {
    'MLP_bin_1': 'MLP (bin=1)',
    'MLP_bin_0.1': 'MLP (bin=0.1)',
    'MLP_bin_0.01': 'MLP (bin=0.01)',
    'Transformer_encode': 'Transformer (encode)'
}

# ==============================
# Process each model
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
        else:
            # fallback: search all CSV files
            actual_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
            for f in actual_files:
                if f'fold{fold}' in f and '_combined_roc_curves.csv' in f:
                    file_path = os.path.join(base_dir, f)
                    roc_files.append((fold, file_path))

    if not roc_files:
        print(f"Warning: No ROC curve files found for {model_display_names[model_name]}")
        continue

    # Read and process each fold
    all_fprs, all_tprs, fold_aucs = [], [], []
    for fold, file_path in roc_files:
        df = pd.read_csv(file_path)
        class_df = df[df['class'].isin(['overall','combined_overall','combined'])] if 'class' in df.columns else df

        if 'fpr' in class_df.columns and 'tpr' in class_df.columns:
            fpr = np.array(class_df['fpr'])
            tpr = np.array(class_df['tpr'])
            sorted_idx = np.argsort(fpr)
            fpr_sorted = fpr[sorted_idx]
            tpr_sorted = tpr[sorted_idx]

            auc_value = class_df['auc'].iloc[0] if 'auc' in class_df.columns else np.trapz(tpr_sorted, fpr_sorted)
            fold_aucs.append(auc_value)
            all_fprs.append(fpr_sorted)
            all_tprs.append(tpr_sorted)

    if not fold_aucs:
        continue

    # Compute mean TPR with interpolation
    fpr_grid = np.linspace(0, 1, 101)
    interpolated_tprs = []
    for fpr_curve, tpr_curve in zip(all_fprs, all_tprs):
        fpr_ext = np.concatenate([[0], fpr_curve]) if fpr_curve[0] > 0 else fpr_curve
        tpr_ext = np.concatenate([[0], tpr_curve]) if fpr_curve[0] > 0 else tpr_curve
        if fpr_ext[-1] < 1.0:
            fpr_ext = np.append(fpr_ext, 1.0)
            tpr_ext = np.append(tpr_ext, 1.0)
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
# Plot full and zoomed ROC curves
# ==============================
if all_models_data:
    # Full ROC
    fig1, ax1 = plt.subplots(figsize=(3,3))
    ax1.plot([0,1],[0,1], color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random (AUC = 0.500)')

    draw_order = ['Transformer_encode','MLP_bin_0.01','MLP_bin_0.1','MLP_bin_1']
    for model_name in draw_order:
        if model_name in all_models_data:
            data = all_models_data[model_name]
            ax1.plot(data['fpr_grid'], data['mean_tpr'], color=model_colors[model_name], linestyle=model_linestyles[model_name], linewidth=2.5,
                     label=f"{data['display_name']} (AUC={data['avg_auc']:.6f})")

    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(os.path.join(r"C:\Users\13709\Desktop\论文撰写四\模型\测试","roc_curves_full_view_no_offset.png"), dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

    # Zoomed ROC (low FPR)
    fig2, ax2 = plt.subplots(figsize=(2,4))
    zoom_xlim = [0.0,0.02]; zoom_ylim = [0.75,1.0]
    ax2.plot(zoom_xlim, zoom_xlim, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    for model_name in draw_order:
        if model_name in all_models_data:
            data = all_models_data[model_name]
            mask = (data['fpr_grid']>=zoom_xlim[0]) & (data['fpr_grid']<=zoom_xlim[1])
            ax2.plot(data['fpr_grid'][mask], data['mean_tpr'][mask], color=model_colors[model_name], linestyle=model_linestyles[model_name], linewidth=3.0,
                     label=f"{data['display_name']} (AUC={data['avg_auc']:.6f})")
    ax2.set_xlim(zoom_xlim); ax2.set_ylim(zoom_ylim)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5); ax2.spines['bottom'].set_linewidth(1.5)
    ax2.set_xticks([zoom_xlim[0], zoom_xlim[1]]); ax2.set_yticks([zoom_ylim[0], zoom_ylim[1]])
    ax2.set_xticklabels([]); ax2.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(os.path.join(r"C:\Users\13709\Desktop\论文撰写四\模型\测试","roc_curves_zoomed_view_no_offset.png"), dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

# ==============================
# Save summary statistics
# ==============================
comparison_stats = []
for model_name, data in all_models_data.items():
    comparison_stats.append({
        'Model': data['display_name'],
        'Mean_AUC': data['avg_auc'],
        'Std_AUC': data['std_auc'],
        'Min_AUC': data['min_auc'],
        'Max_AUC': data['max_auc'],
        'Range': f"{data['min_auc']:.6f} - {data['max_auc']:.6f}",
        'Num_Folds': data['num_folds']
    })
comparison_df = pd.DataFrame(comparison_stats)
comparison_file = os.path.join(r"C:\Users\13709\Desktop\论文撰写四\模型\测试","all_models_roc_comparison_no_offset.csv")
comparison_df.to_csv(comparison_file, index=False, float_format='%.6f')
print(f"All ROC comparison statistics saved to: {comparison_file}")