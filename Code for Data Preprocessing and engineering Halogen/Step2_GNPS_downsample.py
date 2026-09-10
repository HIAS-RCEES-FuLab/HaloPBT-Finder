import pandas as pd
import os

# ============================
# Input/Output directories
# ============================
input_dir = r"D:\MSMS\MSMS_ClBr_H+H-\GNPS"
output_dir = os.path.join(input_dir, "sampled")
os.makedirs(output_dir, exist_ok=True)

# ============================
# CSV file paths
# ============================
files = {
    "ClBr_POS": os.path.join(input_dir, "filtered_gnps_POS_with_ClBr.csv"),
    "ClBr_NEG": os.path.join(input_dir, "filtered_gnps_NEG_with_ClBr.csv"),
    "noClBr_POS": os.path.join(input_dir, "filtered_gnps_POS_no_ClBr.csv"),
    "noClBr_NEG": os.path.join(input_dir, "filtered_gnps_NEG_no_ClBr.csv"),
}

# ============================
# Load datasets
# ============================
datasets = {name: pd.read_csv(path) for name, path in files.items()}

# ============================
# Sampling function
# ============================
def sample_no_clbr(noclbr_df, clbr_df, ratio=2, max_peaks=500, random_state=42):
    filtered = noclbr_df[noclbr_df['Num_peaks'] < max_peaks]
    target_count = len(clbr_df) * ratio
    if len(filtered) > target_count:
        return filtered.sample(n=target_count, random_state=random_state)
    return filtered

# ============================
# Apply sampling
# ============================
sampled_noClBr_POS = sample_no_clbr(datasets["noClBr_POS"], datasets["ClBr_POS"])
sampled_noClBr_NEG = sample_no_clbr(datasets["noClBr_NEG"], datasets["ClBr_NEG"])

# ============================
# Save sampled datasets
# ============================
sampled_noClBr_POS.to_csv(os.path.join(output_dir, "filtered_gnps_POS_no_ClBr_sampled.csv"), index=False)
sampled_noClBr_NEG.to_csv(os.path.join(output_dir, "filtered_gnps_NEG_no_ClBr_sampled.csv"), index=False)

# ============================
# Print summary statistics
# ============================
summary = pd.DataFrame({
    "Dataset": ["ClBr_POS", "ClBr_NEG", "noClBr_POS_sampled", "noClBr_NEG_sampled"],
    "Total_spectra": [
        len(datasets["ClBr_POS"]),
        len(datasets["ClBr_NEG"]),
        len(sampled_noClBr_POS),
        len(sampled_noClBr_NEG)
    ],
    "Unique_compounds": [
        datasets["ClBr_POS"]["Name"].nunique(),
        datasets["ClBr_NEG"]["Name"].nunique(),
        sampled_noClBr_POS["Name"].nunique(),
        sampled_noClBr_NEG["Name"].nunique()
    ]
})

print("\n===== Sampling Summary =====")
print(summary)