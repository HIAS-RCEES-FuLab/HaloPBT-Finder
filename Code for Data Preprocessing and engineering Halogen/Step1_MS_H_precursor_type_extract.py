import pandas as pd
import os
import ast

# -----------------------------
# Data directories
# -----------------------------
data_dir = r"D:\MSMS\MSMS_ClBr"       # Input directory
output_dir = r"D:\MSMS\MSMS_ClBr_H+H-" # Output directory
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# CSV file lists per database
# -----------------------------
files = {
    "NIST": [
        "NIST/filtered_nist_NEG_with_ClBr.csv",
        "NIST/filtered_nist_POS_with_ClBr.csv",
        "NIST/filtered_nist_NEG_no_ClBr.csv",
        "NIST/filtered_nist_POS_no_ClBr.csv"
    ],
    "MoNA": [
        "MoNA/filtered_mona_NEG_with_ClBr.csv",
        "MoNA/filtered_mona_POS_with_ClBr.csv",
        "MoNA/filtered_mona_NEG_no_ClBr.csv",
        "MoNA/filtered_mona_POS_no_ClBr.csv"
    ],
    "GNPS": [
        "GNPS/filtered_gnps_NEG_with_ClBr.csv",
        "GNPS/filtered_gnps_POS_with_ClBr.csv",
        "GNPS/filtered_gnps_NEG_no_ClBr.csv",
        "GNPS/filtered_gnps_POS_no_ClBr.csv"
    ],
}

# -----------------------------
# Configuration
# -----------------------------
mode_map = {"[M-H]-": "NEG", "[M+H]+": "POS"}
columns_to_keep = [
    "Name", "Precursor_type", "Precursor_mz", "Ion_mode",
    "SMILES", "Formula", "Num_peaks", "Mass_spectral"
]
H_mass = 1.007276

# -----------------------------
# Check if Mass_spectral has more than 2 decimal places in m/z
# -----------------------------
def has_more_than_2_decimal(ms_string):
    try:
        ms_list = ast.literal_eval(ms_string)
        for mz, intensity in ms_list:
            decimal_part = str(mz).split(".")[1] if "." in str(mz) else "0"
            if len(decimal_part) > 2:
                return True
        return False
    except:
        return True

# -----------------------------
# Summary storage
# -----------------------------
summary_all = []

# -----------------------------
# Process each database and file
# -----------------------------
for source, file_list in files.items():
    os.makedirs(output_dir, exist_ok=True)

    compounds_pos = set()
    compounds_neg = set()

    for fname in file_list:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            continue

        df = pd.read_csv(fpath)
        if "Precursor_type" not in df.columns:
            continue
        df["Precursor_type"] = df["Precursor_type"].fillna("Unknown")

        # Keep only specified precursor types
        df = df[df["Precursor_type"].isin(mode_map.keys())].copy()
        if df.empty:
            continue

        for precursor, mode in mode_map.items():
            df_filtered = df[df["Precursor_type"] == precursor].copy()
            total_rows = len(df_filtered)
            if total_rows == 0:
                continue

            # Keep only selected columns
            df_filtered = df_filtered[[col for col in columns_to_keep if col in df_filtered.columns]].copy()

            # Discard rows with m/z decimal ≤2
            if "Mass_spectral" in df_filtered.columns:
                mask = df_filtered["Mass_spectral"].apply(has_more_than_2_decimal)
                df_final = df_filtered[mask].copy()
                discarded_due_to_precision = total_rows - len(df_final)
            else:
                df_final = df_filtered.copy()
                discarded_due_to_precision = 0

            # Convert Precursor_mz to float and calculate Exact_mass
            if "Precursor_mz" in df_final.columns:
                df_final.loc[:, "Precursor_mz"] = pd.to_numeric(df_final["Precursor_mz"], errors="coerce")

                def calc_exact_mass(row):
                    mz = row["Precursor_mz"]
                    if pd.isna(mz):
                        return None
                    if row["Precursor_type"] == "[M+H]+":
                        return mz - H_mass
                    elif row["Precursor_type"] == "[M-H]-":
                        return mz + H_mass
                    else:
                        return mz

                df_final.loc[:, "Exact_mass"] = df_final.apply(calc_exact_mass, axis=1)

            saved_rows = len(df_final)
            compound_count = df_final["Name"].nunique() if "Name" in df_final.columns else 0

            # Save filtered CSV
            if not df_final.empty:
                out_fname = f"{os.path.splitext(fname)[0]}_{mode}.csv"
                out_path = os.path.join(output_dir, out_fname)
                df_final.to_csv(out_path, index=False)

            print(
                f"File '{fname}' | Precursor '{precursor}': "
                f"Original {total_rows} rows, Discarded {discarded_due_to_precision} rows, "
                f"Saved {saved_rows} rows, Unique compounds {compound_count}"
            )

            # Update compound sets
            if "Name" in df_final.columns:
                if mode == "POS":
                    compounds_pos.update(df_final["Name"])
                elif mode == "NEG":
                    compounds_neg.update(df_final["Name"])

    # Summary statistics for the database
    shared_compounds = compounds_pos & compounds_neg
    total_unique_compounds = compounds_pos | compounds_neg

    summary_all.append({
        "Database": source,
        "POS compounds": len(compounds_pos),
        "NEG compounds": len(compounds_neg),
        "Shared compounds (POS ∩ NEG)": len(shared_compounds),
        "Total unique compounds (POS ∪ NEG)": len(total_unique_compounds)
    })

# -----------------------------
# Output summary
# -----------------------------
summary_df = pd.DataFrame(summary_all)
print("\nDatabase summary:")
print(summary_df)