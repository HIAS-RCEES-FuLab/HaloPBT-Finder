import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================================
# This script visualizes the Precision-Recall (PR) curves
# for PBT prediction using four different models:
# MLP (bin=1, 0.1, 0.01) and Transformer (encoded features).
# The script reads precomputed PR curves from CSV files for
# five cross-validation folds, computes mean PR curves,
# average precision (AP) statistics (mean, std, min, max),
# plots full PR curves and zoomed-in high-precision regions,
# and saves the results as publication-ready figures.
# =====================================================

# Set global font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Define model data directories
model_paths = {
    'MLP_bin_1_PBT': r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PR_ROC_PBT_bin\1",
    'MLP_bin_0.1_PBT': r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PR_ROC_PBT_bin\0.1",
    'MLP_bin_0.01_PBT': r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PR_ROC_PBT_bin\0.01",
    'Transformer_encode_PBT': r"C:\Users\13709\Desktop\论文撰写四\模型\测试\PR_ROC_PBT_encode"
}

# File name patterns for PR curves
file_patterns = {
    'MLP_bin_1_PBT': "multi_PBT_1_0122_all_fold{}_pr_curve.csv",
    'MLP_bin_0.1_PBT': "multi_PBT_0.1_0122_all_fold{}_pr_curve.csv",
    'MLP_bin_0.01_PBT': "multi_PBT_0.01_0122_all_fold{}_pr_curve.csv",
    'Transformer_encode_PBT': "multi_PBT_transformer_0.01_0119_all_fold{}_pr_curve.csv"
}

# Storage for all model data
all_models_data = {}

# Color settings for plots
model_colors = {
    'MLP_bin_1_PBT': '#B39ED5',
    'MLP_bin_0.1_PBT': '#FBD49C',
    'MLP_bin_0.01_PBT': '#FDB7AE',
    'Transformer_encode_PBT': '#87BBDE'
}

# Line style settings
model_linestyles = {
    'MLP_bin_1_PBT': '-',
    'MLP_bin_0.1_PBT': '-',
    'MLP_bin_0.01_PBT': '-',
    'Transformer_encode_PBT': '-'
}

# Model display names
model_display_names = {
    'MLP_bin_1_PBT': 'MLP (bin=1)',
    'MLP_bin_0.1_PBT': 'MLP (bin=0.1)',
    'MLP_bin_0.01_PBT': 'MLP (bin=0.01)',
    'Transformer_encode_PBT': 'Transformer (encode)'
}

# =========================
# Process each model
# =========================
for model_name, base_dir in model_paths.items():
    print(f"\n{'=' * 60}")
    print(f"Processing model: {model_display_names[model_name]}")
    print(f"Directory: {base_dir}")
    print('=' * 60)

    file_pattern = file_patterns[model_name]
    pr_files = []

    # Search PR curve files for all folds
    for fold in range(1, 6):
        file_path = os.path.join(base_dir, file_pattern.format(fold))
        if os.path.exists(file_path):
            pr_files.append((fold, file_path))
            print(f"  Found Fold {fold} file: {os.path.basename(file_path)}")
        else:
            # Try to find CSV files in the directory
            actual_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
            for f in actual_files:
                if f'fold{fold}' in f and '_pr_curve.csv' in f:
                    file_path = os.path.join(base_dir, f)
                    pr_files.append((fold, file_path))
                    print(f"  Found matching file: {f}")

    if not pr_files:
        print(f"Warning: {model_display_names[model_name]} has no PR curve files")
        continue

    # Read and process all fold data
    all_recalls = []
    all_precisions = []
    fold_aprs = []

    for fold, file_path in pr_files:
        try:
            df = pd.read_csv(file_path)

            if 'recall' in df.columns and 'precision' in df.columns:
                recall = df['recall'].values
                precision = df['precision'].values

                # Sort by recall
                sorted_indices = np.argsort(recall)
                recall_sorted = recall[sorted_indices]
                precision_sorted = precision[sorted_indices]

                # Get AP from file if available
                if 'ap' in df.columns and len(df['ap'].unique()) > 0:
                    apr_value = df['ap'].iloc[0]
                elif 'auc' in df.columns and len(df['auc'].unique()) > 0:
                    apr_value = df['auc'].iloc[0]
                else:
                    # Compute AP using trapezoidal rule
                    apr_value = np.trapz(precision_sorted, recall_sorted)

                fold_aprs.append(apr_value)
                all_recalls.append(recall_sorted)
                all_precisions.append(precision_sorted)
                print(f"  Fold {fold}: AP = {apr_value:.6f}")
            else:
                print(f"  Warning: Fold {fold} file missing 'recall' or 'precision' columns")
        except Exception as e:
            print(f"  Error processing Fold {fold} file: {e}")

    if not fold_aprs:
        print(f"Warning: {model_display_names[model_name]} has no valid AP data")
        continue

    # Compute average and stats
    apr_values = fold_aprs
    avg_apr = np.mean(apr_values)
    std_apr = np.std(apr_values)
    min_apr = np.min(apr_values)
    max_apr = np.max(apr_values)

    print(f"\n  {model_display_names[model_name]} statistics:")
    print(f"    Average AP: {avg_apr:.6f} ± {std_apr:.6f}")
    print(f"    Range: {min_apr:.6f} - {max_apr:.6f}")

    # Interpolate curves to a common recall grid
    recall_grid = np.linspace(0, 1, 101)
    interpolated_precisions = []

    for recall_curve, precision_curve in zip(all_recalls, all_precisions):
        if len(recall_curve) == 0 or len(precision_curve) == 0:
            continue

        if recall_curve[0] > 0:
            recall_extended = np.concatenate([[0], recall_curve])
            precision_extended = np.concatenate([[1], precision_curve])
        else:
            recall_extended = recall_curve
            precision_extended = precision_curve

        if recall_extended[-1] < 1.0:
            recall_extended = np.concatenate([recall_extended, [1.0]])
            precision_extended = np.concatenate([precision_extended, [precision_extended[-1]]])

        precision_interp = np.interp(
            recall_grid,
            recall_extended,
            precision_extended,
            left=1.0,
            right=precision_extended[-1]
        )
        interpolated_precisions.append(precision_interp)

    if not interpolated_precisions:
        print(f"Warning: {model_display_names[model_name]} could not compute interpolated curves")
        continue

    interpolated_precisions = np.array(interpolated_precisions)
    mean_precision = np.mean(interpolated_precisions, axis=0)
    std_precision = np.std(interpolated_precisions, axis=0)

    # Save model data
    all_models_data[model_name] = {
        'recall_grid': recall_grid,
        'mean_precision': mean_precision,
        'avg_apr': avg_apr,
        'std_apr': std_apr,
        'min_apr': min_apr,
        'max_apr': max_apr,
        'fold_aprs': apr_values,
        'num_folds': len(pr_files),
        'display_name': model_display_names[model_name]
    }

# =========================
# Plot full PR curves (no offset)
# =========================
if all_models_data:
    draw_order = ['MLP_bin_0.01_PBT', 'MLP_bin_0.1_PBT', 'MLP_bin_1_PBT', 'Transformer_encode_PBT']

    fig1, ax1 = plt.subplots(figsize=(3, 3))

    for model_name in draw_order:
        if model_name not in all_models_data:
            continue
        model_data = all_models_data[model_name]
        ax1.plot(model_data['recall_grid'], model_data['mean_precision'],
                 color=model_colors.get(model_name, '#000000'),
                 linestyle=model_linestyles.get(model_name, '-'),
                 linewidth=2.5, alpha=1.0)

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    plt.tight_layout()

    output_dir = r"C:\Users\13709\Desktop\论文撰写四\模型\测试"
    output_file1 = os.path.join(output_dir, "pbt_all_models_pr_curves_full_view_no_offset.png")
    plt.savefig(output_file1, dpi=300, bbox_inches='tight', transparent=True)
    print(f"\n{'=' * 60}")
    print(f"Full PR curves for all PBT models saved at: {output_file1}")
    print('=' * 60)

    plt.show()
else:
    print("No valid model data found")