import pandas as pd
import glob
import os

def load_input_smiles(input_csv):
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    pbt_1 = (df["PBT_label"] == 1).sum()
    pbt_0 = (df["PBT_label"] == 0).sum()
    canonical_set = set(df["Canonical_smiles"].dropna())
    info = {
        "total_rows": total_rows,
        "PBT_1": pbt_1,
        "PBT_0": pbt_0
    }
    return canonical_set, df, info

def filter_compounds(file_pattern, input_csv):
    target_smiles_set, input_df, input_info = load_input_smiles(input_csv)
    matched_rows = []
    files = sorted(glob.glob(file_pattern))
    print(f">>> Detected {len(files)} files")

    total_rows = 0
    for file in files:
        print(f"Processing file: {file}")
        df = pd.read_csv(file)
        total_rows += len(df)
        if "SMILES" not in df.columns:
            print(f"File {file} has no SMILES column, skipped")
            continue
        matched = df[df["SMILES"].isin(target_smiles_set)].copy()
        matched_rows.append(matched)

    matched_df = pd.concat(matched_rows, ignore_index=True) if matched_rows else pd.DataFrame()
    matched_df = matched_df.merge(
        input_df[["Canonical_smiles", "PBT_label", "Table2_PBT", "Overlap_PBT", "Source_files"]],
        left_on="SMILES",
        right_on="Canonical_smiles",
        how="left"
    )
    matched_df.drop(columns=["Canonical_smiles"], inplace=True)

    pbt_1_rows = (matched_df["PBT_label"] == 1).sum()
    pbt_0_rows = (matched_df["PBT_label"] == 0).sum()

    unique_df = matched_df.drop_duplicates(subset=["SMILES"])
    unique_compounds = len(unique_df)
    unique_pbt_1 = (unique_df["PBT_label"] == 1).sum()
    unique_pbt_0 = (unique_df["PBT_label"] == 0).sum()

    match_info = {
        "files": len(files),
        "total_rows": total_rows,
        "matched_rows": len(matched_df),
        "PBT_1_rows": pbt_1_rows,
        "PBT_0_rows": pbt_0_rows,
        "unique_compounds": unique_compounds,
        "unique_PBT_1": unique_pbt_1,
        "unique_PBT_0": unique_pbt_0
    }
    return matched_df, input_info, match_info

pos_pattern = os.path.join("MSMS_extract", "mona_data_POS_*.csv")
neg_pattern = os.path.join("MSMS_extract", "mona_data_NEG_*.csv")
# Collected list of PBT and non-PBT compounds
input_csv = "halo_Si_merged_final.csv"

pos_match, input_info, pos_info = filter_compounds(pos_pattern, input_csv)
neg_match, _, neg_info = filter_compounds(neg_pattern, input_csv)

pos_PBT = pos_match[pos_match["PBT_label"] == 1]
pos_noPBT = pos_match[pos_match["PBT_label"] == 0]
pos_PBT.to_csv("MoNA_POS_PBT.csv", index=False)
pos_noPBT.to_csv("MoNA_POS_noPBT.csv", index=False)

neg_PBT = neg_match[neg_match["PBT_label"] == 1]
neg_noPBT = neg_match[neg_match["PBT_label"] == 0]
neg_PBT.to_csv("MoNA_NEG_PBT.csv", index=False)
neg_noPBT.to_csv("MoNA_NEG_noPBT.csv", index=False)

print("\n================= Matching Report =================")
print(f">>> Input PBT table")
print(f"Total compounds: {input_info['total_rows']}")
print(f"PBT=1: {input_info['PBT_1']}")
print(f"PBT=0: {input_info['PBT_0']}")

print(f"\n>>> POS match results")
print(f"Files: {pos_info['files']}")
print(f"Total rows: {pos_info['total_rows']}")
print(f"Matched rows (including duplicates): {pos_info['matched_rows']}")
print(f"PBT=1 (rows): {pos_info['PBT_1_rows']}")
print(f"PBT=0 (rows): {pos_info['PBT_0_rows']}")
print(f"Unique compounds: {pos_info['unique_compounds']}")
print(f"Unique PBT=1: {pos_info['unique_PBT_1']}")
print(f"Unique PBT=0: {pos_info['unique_PBT_0']}")

print(f"\n>>> NEG match results")
print(f"Files: {neg_info['files']}")
print(f"Total rows: {neg_info['total_rows']}")
print(f"Matched rows (including duplicates): {neg_info['matched_rows']}")
print(f"PBT=1 (rows): {neg_info['PBT_1_rows']}")
print(f"PBT=0 (rows): {neg_info['PBT_0_rows']}")
print(f"Unique compounds: {neg_info['unique_compounds']}")
print(f"Unique PBT=1: {neg_info['unique_PBT_1']}")
print(f"Unique PBT=0: {neg_info['unique_PBT_0']}")

print("\n>>> Matching completed. MoNA PBT / noPBT files successfully saved!")