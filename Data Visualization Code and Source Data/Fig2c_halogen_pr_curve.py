import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================================
# This script visualizes the Precision-Recall (PR) curves
# for Cl/Br prediction using four different models:
# MLP (bin=1, 0.1, 0.01) and Transformer (encoded features).
# The script reads precomputed PR curves from CSV files for
# five cross-validation folds, computes mean PR curves,
# average precision (AP) statistics (mean, std, min, max),
# plots full PR curves and zoomed-in high-precision regions,
# and saves the results as publication-ready figures.
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
    'MLP_bin_1': "multi_cl_br_1_0122_fold{}_combined_pr_curves.csv",
    'MLP_bin_0.1': "multi_cl_br_0.1_0118_fold{}_combined_pr_curves.csv",
    'MLP_bin_0.01': "multi_cl_br_0.01_0120_fold{}_combined_pr_curves.csv",
    'Transformer_encode': "multi_cl_br_transformer_0.01_0122_fold{}_combined_pr_curves.csv"
}

# ==============================
# Store all model data
# ==============================
all_models_data = {}

model_colors = {
    'MLP_bin_1': '#B39ED5',
    'MLP_bin_0.1': '#FBD49C',
    'MLP_bin_0.01': '#FDB7AE',
    'Transformer_encode': '#87BBDE'
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
# Process each model and compute mean PR curves
# ==============================
for model_name, base_dir in model_paths.items():
    print(f"\n{'='*60}")
    print(f"Processing model: {model_display_names[model_name]}")
    print(f"Path: {base_dir}")
    print('='*60)

    file_pattern = file_patterns[model_name]
    pr_files = []

    # Find PR curve files for 5 folds
    for fold in range(1, 6):
        file_path = os.path.join(base_dir, file_pattern.format(fold))
        if os.path.exists(file_path):
            pr_files.append((fold, file_path))

    if not pr_files:
        print(f"Warning: No PR curve files found for {model_display_names[model_name]}")
        continue

    all_recalls, all_precisions, fold_aprs = [], [], []

    for fold, file_path in pr_files:
        try:
            df = pd.read_csv(file_path)

            # Filter for overall or combined_overall if available
            if 'class' in df.columns:
                for class_name in ['overall', 'combined_overall', 'combined']:
                    class_df = df[df['class'] == class_name]
                    if not class_df.empty:
                        break
                if class_df.empty:
                    class_df = df
            else:
                class_df = df

            if 'recall' in class_df.columns and 'precision' in class_df.columns:
                recall = class_df['recall'].values
                precision = class_df['precision'].values
                sorted_idx = np.argsort(recall)
                recall_sorted = recall[sorted_idx]
                precision_sorted = precision[sorted_idx]

                # Get AP (average precision)
                if 'ap' in class_df.columns and len(class_df['ap'].unique()) > 0:
                    apr_value = class_df['ap'].iloc[0]
                elif 'auc' in class_df.columns and len(class_df['auc'].unique()) > 0:
                    apr_value = class_df['auc'].iloc[0]
                else:
                    apr_value = np.trapz(precision_sorted, recall_sorted)

                fold_aprs.append(apr_value)
                all_recalls.append(recall_sorted)
                all_precisions.append(precision_sorted)
            else:
                print(f"  Warning: Fold {fold} missing recall or precision columns")
        except Exception as e:
            print(f"  Error processing fold {fold}: {e}")

    if not fold_aprs:
        continue

    recall_grid = np.linspace(0,1,101)
    interpolated_precisions = []

    for recall_curve, precision_curve in zip(all_recalls, all_precisions):
        recall_ext = np.concatenate([[0], recall_curve]) if recall_curve[0]>0 else recall_curve
        precision_ext = np.concatenate([[1], precision_curve]) if recall_curve[0]>0 else precision_curve
        if recall_ext[-1]<1.0:
            recall_ext = np.append(recall_ext,1.0)
            precision_ext = np.append(precision_ext, precision_ext[-1])
        precision_interp = np.interp(recall_grid, recall_ext, precision_ext, left=1.0, right=precision_ext[-1])
        interpolated_precisions.append(precision_interp)

    interpolated_precisions = np.array(interpolated_precisions)
    all_models_data[model_name] = {
        'recall_grid': recall_grid,
        'mean_precision': np.mean(interpolated_precisions, axis=0),
        'avg_apr': np.mean(fold_aprs),
        'std_apr': np.std(fold_aprs),
        'min_apr': np.min(fold_aprs),
        'max_apr': np.max(fold_aprs),
        'fold_aprs': fold_aprs,
        'num_folds': len(fold_aprs),
        'display_name': model_display_names[model_name]
    }

# ==============================
# Plot full PR curves (publication style, no offset)
# ==============================
if all_models_data:
    draw_order = ['MLP_bin_0.01', 'MLP_bin_0.1', 'MLP_bin_1', 'Transformer_encode']

    # Full PR curve
    fig1, ax1 = plt.subplots(figsize=(3,3))
    for model_name in draw_order:
        if model_name not in all_models_data:
            continue
        data = all_models_data[model_name]
        ax1.plot(data['recall_grid'], data['mean_precision'],
                 color=model_colors[model_name],
                 linestyle=model_linestyles[model_name],
                 linewidth=2.5, alpha=1.0)

    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.tight_layout()

    output_dir = r"C:\Users\13709\Desktop\论文撰写四\模型\测试"
    output_file1 = os.path.join(output_dir, "pr_curves_full_view_no_offset.png")
    plt.savefig(output_file1, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

    # Zoomed-in PR curve (high precision region)
    zoom_xlim = [0.97,1.0]
    zoom_ylim = [0.97,1.0]
    fig2, ax2 = plt.subplots(figsize=(2,2))
    for model_name in draw_order:
        if model_name not in all_models_data:
            continue
        data = all_models_data[model_name]
        mask = (data['recall_grid']>=zoom_xlim[0]) & (data['recall_grid']<=zoom_xlim[1])
        ax2.plot(data['recall_grid'][mask], data['mean_precision'][mask],
                 color=model_colors[model_name],
                 linestyle=model_linestyles[model_name],
                 linewidth=3.0)
    ax2.set_xlim(zoom_xlim)
    ax2.set_ylim(zoom_ylim)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    plt.tight_layout()
    output_file2 = os.path.join(output_dir, "pr_curves_zoomed_view_no_offset.png")
    plt.savefig(output_file2, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

    # Save detailed statistics
    comparison_stats = []
    for model_name in draw_order:
        if model_name not in all_models_data: continue
        data = all_models_data[model_name]
        comparison_stats.append({
            'Model': data['display_name'],
            'Mean_AP': data['avg_apr'],
            'Std_AP': data['std_apr'],
            'Min_AP': data['min_apr'],
            'Max_AP': data['max_apr'],
            'Range': f"{data['min_apr']:.6f} - {data['max_apr']:.6f}",
            'Num_Folds': data['num_folds']
        })
    comparison_df = pd.DataFrame(comparison_stats)
    comparison_file = os.path.join(output_dir, "all_models_pr_comparison_no_offset.csv")
    comparison_df.to_csv(comparison_file, index=False, float_format='%.6f')

    # Print detailed comparison
    print("\n" + "="*80)
    print("Model performance comparison (original AP, no offset)")
    print("="*80)
    sorted_stats = sorted(comparison_stats, key=lambda x: x['Mean_AP'], reverse=True)
    for i, data in enumerate(sorted_stats):
        print(f"{data['Model']:<20} Mean: {data['Mean_AP']:.6f}  Std: {data['Std_AP']:.6f}  Range: {data['Range']}  Rank: #{i+1}")
    best_model = sorted_stats[0]
    print(f"\nBest model: {best_model['Model']}, AP={best_model['Mean_AP']:.6f}")

else:
    print("No valid model data found.")