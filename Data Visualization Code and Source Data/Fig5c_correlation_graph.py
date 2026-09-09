import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, t

base_dir = r"C:\Users\13709\Desktop\论文撰写四\ZJ_YUHUAN_PBT"
pbt_file = os.path.join(base_dir, "Shared_PBT_information_matched.csv")
plot_file = os.path.join(base_dir, "Seafood_Plasma_DF_correlation.png")

df = pd.read_csv(pbt_file, low_memory=False)
required_cols = ["Seafood_Detection_Rate", "Plasma_Detection_Rate"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

data = df[required_cols].copy()
data[required_cols] = data[required_cols].apply(pd.to_numeric, errors="coerce")
data = data.dropna()

x = data["Seafood_Detection_Rate"].values
y = data["Plasma_Detection_Rate"].values

# Pearson correlation
pearson_r, pearson_p = pearsonr(x, y)

print("=" * 60)
print(f"Number of compounds: {len(data)}")
print(f"Pearson r = {pearson_r:.4f}")
print(f"Pearson p = {pearson_p:.4e}")
print("=" * 60)

# Linear regression forced through the origin: y = kx
k = np.sum(x * y) / np.sum(x ** 2)
y_fit = k * x

n = len(x)
residuals = y - y_fit
SSE = np.sum(residuals ** 2)

# Error estimation for regression through the origin
MSE = SSE / (n - 1)
Sxx_origin = np.sum(x ** 2)
se_k = np.sqrt(MSE / Sxx_origin)

# 95% confidence interval for the slope
t_critical = t.ppf(0.975, n - 1)
k_lower = k - t_critical * se_k
k_upper = k + t_critical * se_k

print(f"Slope = {k:.6f}")
print(f"Slope 95% CI = [{k_lower:.6f}, {k_upper:.6f}]")
print("=" * 60)

# Regression line and 95% confidence interval
x_line = np.linspace(0, max(x.max(), y.max()), 200)
y_line = k * x_line

se_mean = np.sqrt(MSE * (x_line ** 2) / Sxx_origin)
y_lower = y_line - t_critical * se_mean
y_upper = y_line + t_critical * se_mean

# Plot
fig, ax = plt.subplots(figsize=(4, 3))

# 95% confidence interval
ax.fill_between(
    x_line,
    y_lower,
    y_upper,
    color="lightgray",
    alpha=0.5,
    zorder=1
)

# Scatter plot
ax.scatter(
    x,
    y,
    s=35,
    color="#1f77b4",
    alpha=0.8,
    zorder=3
)

# Regression line forced through the origin
ax.plot(
    x_line,
    y_line,
    linewidth=2,
    zorder=4
)

max_value = max(x.max(), y.max())
ax.set_xlim(0, max_value * 1.05)
ax.set_ylim(0, max_value * 1.05)

# Hide tick labels
ax.tick_params(
    axis="both",
    which="both",
    labelbottom=False,
    labelleft=False
)

# Hide top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    plot_file,
    dpi=600,
    bbox_inches="tight",
    transparent=True
)
plt.show()