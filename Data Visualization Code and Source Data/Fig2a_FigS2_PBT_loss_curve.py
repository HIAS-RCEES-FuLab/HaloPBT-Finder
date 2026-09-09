import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# This script visualizes training and validation loss curves for PBT model.
# The CSV file already contains preprocessed training/validation metrics
# =====================================================

# Set global font to Arial
plt.rcParams['font.family'] = 'Arial'

# Load CSV file
file_path = r"C:\Users\13709\Desktop\论文撰写四\模型\验证\PBT_bin_0.1_trainval_results.csv"
df = pd.read_csv(file_path)

# Check required columns
required_cols = ['fold', 'epoch', 'train_loss', 'val_loss']
for col in required_cols:
    if col not in df.columns:
        print(f"Error: '{col}' column is missing")
        exit()

# Determine maximum fold number
max_fold = df['fold'].max()
print(f"\nMaximum fold number: {max_fold}")

# Create subplots for each fold
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
max_epochs_per_fold = []

for fold in range(1, max_fold + 1):
    fold_data = df[df['fold'] == fold].sort_values('epoch')
    if fold_data.empty:
        print(f"Warning: Fold {fold} has no data")
        continue

    ax = axes[fold - 1]
    epochs = fold_data['epoch'].values
    train_losses = fold_data['train_loss'].values
    val_losses = fold_data['val_loss'].values
    max_epoch = max(epochs)
    max_epochs_per_fold.append(max_epoch)

    print(f"Fold {fold}: Max epoch = {max_epoch}, Data points = {len(epochs)}")

    # Plot training and validation loss
    ax.plot(epochs, train_losses, linewidth=2, alpha=0.8, color='#80b1d3', label='Training Loss')
    ax.plot(epochs, val_losses, linewidth=2, alpha=0.8, color='#fb8072', label='Validation Loss')

    # Extend curves with dashed lines if early stopping occurred
    if max_epoch < 50:
        extended_epochs = np.arange(max_epoch, 51)
        ax.plot(extended_epochs, [train_losses[-1]] * len(extended_epochs),
                'b--', linewidth=1.5, alpha=0.5, color='#80b1d3')
        ax.plot(extended_epochs, [val_losses[-1]] * len(extended_epochs),
                'r--', linewidth=1.5, alpha=0.5, color='#fb8072')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlim(0, 50)

    # Set consistent y-axis across all folds
    all_losses = pd.concat([df['train_loss'], df['val_loss']])
    y_min = all_losses.min() * 0.9
    y_max = all_losses.max() * 1.05
    ax.set_ylim(y_min, y_max)

plt.tight_layout()

# Save fold-level loss figure
output_path = r"C:\Users\13709\Desktop\论文撰写四\模型\验证\loss_curves_by_fold.png"
plt.savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')
print(f"\nLoss curves by fold saved to: {output_path}")
plt.show()

# =============================
# Average loss across folds
# =============================
fig2, ax = plt.subplots(figsize=(6, 3))
max_epoch_all = max(max_epochs_per_fold)

train_loss_matrix = []
val_loss_matrix = []

for epoch in range(1, max_epoch_all + 1):
    train_losses_at_epoch = []
    val_losses_at_epoch = []

    for fold in range(1, max_fold + 1):
        fold_data = df[df['fold'] == fold].sort_values('epoch')
        if not fold_data.empty and epoch <= len(fold_data):
            train_losses_at_epoch.append(fold_data.iloc[epoch - 1]['train_loss'])
            val_losses_at_epoch.append(fold_data.iloc[epoch - 1]['val_loss'])
        elif not fold_data.empty:
            # Use last value if early stopping
            train_losses_at_epoch.append(fold_data['train_loss'].iloc[-1])
            val_losses_at_epoch.append(fold_data['val_loss'].iloc[-1])

    if train_losses_at_epoch:
        train_loss_matrix.append(train_losses_at_epoch)
        val_loss_matrix.append(val_losses_at_epoch)

train_loss_matrix = np.array(train_loss_matrix)
val_loss_matrix = np.array(val_loss_matrix)

# Compute mean and standard deviation
train_mean = np.mean(train_loss_matrix, axis=1)
train_std = np.std(train_loss_matrix, axis=1)
val_mean = np.mean(val_loss_matrix, axis=1)
val_std = np.std(val_loss_matrix, axis=1)

epochs_range = np.arange(1, len(train_mean) + 1)

# Plot mean curves with shaded std
ax.plot(epochs_range, train_mean, linewidth=2, color='#80b1d3', label='Training Loss Mean')
ax.fill_between(epochs_range, train_mean - train_std, train_mean + train_std, alpha=0.3, color='#80b1d3')
ax.plot(epochs_range, val_mean, linewidth=2, color='#fb8072', label='Validation Loss Mean')
ax.fill_between(epochs_range, val_mean - val_std, val_mean + val_std, alpha=0.3, color='#fb8072')

ax.set_xlim(0, 18)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

# Save average loss figure
avg_output_path = r"C:\Users\13709\Desktop\论文撰写四\模型\验证\average_loss_curves.png"
plt.savefig(avg_output_path, dpi=300, transparent=True, bbox_inches='tight')
print(f"Average loss curves saved to: {avg_output_path}")
plt.show()