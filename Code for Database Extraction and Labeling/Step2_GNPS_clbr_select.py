import pandas as pd
import glob
import os
import re

ALLOWED_ELEMENTS = {"C", "H", "O", "N", "P", "S", "F", "Cl", "Br", "I", "Si"}

def parse_molecular_formula(formula):
    if pd.isna(formula):
        return None
    if '+' in formula or '-' in formula:
        return None

    element_counts = {}
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

    for element, count in matches:
        count = int(count) if count else 1
        element_counts[element] = element_counts.get(element, 0) + count

    return element_counts

def only_allowed_elements(mol_elements):
    if mol_elements is None:
        return False
    return set(mol_elements.keys()).issubset(ALLOWED_ELEMENTS)

def filter_compounds(file_pattern):
    files = sorted(glob.glob(file_pattern))
    rows_kept = []

    for file in files:
        df = pd.read_csv(file)
        df = df.dropna(subset=['Formula'])

        df['ElementCounts'] = df['Formula'].apply(parse_molecular_formula)
        df = df[df['ElementCounts'].notna()]

        df = df[df['ElementCounts'].apply(only_allowed_elements)]
        df = df[df['SMILES'].notna() & (df['SMILES'] != "")]

        rows_kept.append(df)

    kept = pd.concat(rows_kept, ignore_index=True) if rows_kept else pd.DataFrame()
    return kept

pos_pattern = os.path.join('MSMS_extract', 'gnps_data_POS_*.csv')
neg_pattern = os.path.join('MSMS_extract', 'gnps_data_NEG_*.csv')

pos_kept = filter_compounds(pos_pattern)
neg_kept = filter_compounds(neg_pattern)

pos_kept.to_csv("filtered_gnps_POS_allowed_elements.csv", index=False)
neg_kept.to_csv("filtered_gnps_NEG_allowed_elements.csv", index=False)

print("GNPS POS kept count:", len(pos_kept))
print("GNPS NEG kept count:", len(neg_kept))