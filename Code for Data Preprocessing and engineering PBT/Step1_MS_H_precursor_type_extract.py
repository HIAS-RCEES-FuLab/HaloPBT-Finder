import os
import pandas as pd
import ast
from rdkit import Chem

def to_canonical_smiles(smiles):
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

# --- Directories ---
data_dir = r"D:\MSMS\MSMS_PBT\MS2_PBT_database"
output_dir = r"D:\MSMS\MSMS_PBT\MS2_PBT_database_H+H-"
os.makedirs(output_dir, exist_ok=True)

# --- Input CSV files ---
files = {
    "NIST": ["NIST/NIST_NEG_PBT.csv", "NIST/NIST_POS_PBT.csv",
             "NIST/NIST_NEG_noPBT.csv", "NIST/NIST_POS_noPBT.csv"],
    "MoNA": ["MoNA/MONA_NEG_PBT.csv", "MoNA/MONA_POS_PBT.csv",
             "MoNA/MONA_NEG_noPBT.csv", "MoNA/MONA_POS_noPBT.csv"],
    "GNPS": ["GNPS/GNPS_NEG_PBT.csv", "GNPS/GNPS_POS_PBT.csv",
             "GNPS/GNPS_NEG_noPBT.csv", "GNPS/GNPS_POS_noPBT.csv"],
}

# --- Configuration ---
ADDUCT_MAP = {
    "M+H": "[M+H]+", "[M+H]": "[M+H]+", "[M+H]+": "[M+H]+",
    "M-H": "[M-H]-", "[M-H]": "[M-H]-", "[M-H]-": "[M-H]-"
}
MODE_MAP = {"[M+H]+": "POS", "[M-H]-": "NEG"}
columns_to_keep = ["Name", "Precursor_type", "Precursor_mz", "Ion_mode",
                   "SMILES", "Formula", "Num_peaks", "Mass_spectral",
                   "PBT_label", "Table2_PBT", "Overlap_PBT"]

H_mass = 1.007276

# --- Check if Mass_spectral m/z has >2 decimal places ---
def has_more_than_2_decimal(ms_string):
    try:
        ms_list = ast.literal_eval(ms_string)
        for mz, intensity in ms_list:
            decimal_part = str(mz).split(".")[1] if "." in str(mz) else "0"
            if len(decimal_part) > 2:
                return True
        return False
    except:
        return True  # keep if parsing fails

# --- Summary collection ---
summary_all = []

for source, file_list in files.items():
    source_dir = os.path.join(output_dir, source)
    os.makedirs(source_dir, exist_ok=True)

    class_counts = {"POS_PBT": set(), "NEG_PBT": set(),
                    "POS_nonPBT": set(), "NEG_nonPBT": set()}

    for fname in file_list:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            continue

        df = pd.read_csv(fpath)
        if "Precursor_type" not in df.columns or "SMILES" not in df.columns:
            continue

        # Standardize adducts
        df["Precursor_type"] = df["Precursor_type"].astype(str).str.strip()
        df["Precursor_type"] = df["Precursor_type"].map(ADDUCT_MAP)
        df = df[df["Precursor_type"].notna()].copy()
        if df.empty:
            continue

        # Canonicalize SMILES
        df["Canonical_smiles"] = df["SMILES"].apply(to_canonical_smiles)
        df = df.dropna(subset=["Canonical_smiles"])
        if df.empty:
            continue

        for precursor, mode in MODE_MAP.items():
            df_mode = df[df["Precursor_type"] == precursor].copy()
            if df_mode.empty:
                continue

            # Keep required columns
            keep_cols = [col for col in columns_to_keep if col in df_mode.columns]
            if "Canonical_smiles" not in keep_cols:
                keep_cols.append("Canonical_smiles")
            df_mode = df_mode[keep_cols]

            # Filter low-precision m/z
            if "Mass_spectral" in df_mode.columns:
                mask = df_mode["Mass_spectral"].apply(has_more_than_2_decimal)
                df_final = df_mode[mask].copy()
            else:
                df_final = df_mode.copy()
            if df_final.empty:
                continue

            # Update class counts
            for _, row in df_final.iterrows():
                key = f"{mode}_PBT" if row.get("PBT_label", 0) == 1 else f"{mode}_nonPBT"
                class_counts[key].add(row["Canonical_smiles"])

            # Save PBT / non-PBT files
            for pbt_flag, label in [(1, "PBT"), (0, "nonPBT")]:
                df_to_save = df_final[df_final["PBT_label"] == pbt_flag]
                if df_to_save.empty:
                    continue
                out_fname = f"{os.path.splitext(os.path.basename(fname))[0]}_{mode}_{label}.csv"
                out_path = os.path.join(source_dir, out_fname)
                df_to_save.to_csv(out_path, index=False)
                print(f"Saved: {out_fname} ({len(df_to_save)} rows)")

    # Summary per database
    summary_all.append({
        "Database": source,
        "POS_PBT": len(class_counts["POS_PBT"]),
        "NEG_PBT": len(class_counts["NEG_PBT"]),
        "POS_nonPBT": len(class_counts["POS_nonPBT"]),
        "NEG_nonPBT": len(class_counts["NEG_nonPBT"])
    })

# Save overall summary
summary_df = pd.DataFrame(summary_all)
print("\nDatabase summary (classified by PBT & mode):")
print(summary_df)
summary_df.to_csv(os.path.join(output_dir, "Database_summary_by_PBT_class.csv"), index=False)