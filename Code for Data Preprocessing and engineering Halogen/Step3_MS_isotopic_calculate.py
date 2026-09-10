import os
import pandas as pd
import re
from itertools import product

# -----------------------------
# Isotope patterns and abundances
# -----------------------------
isotopes_ratios = {
    'C': [(12.0000000, 0.9893), (13.0033548378, 0.0107)],
    'Cl': [(34.968852682, 0.7576), (36.965902602, 0.2424)],
    'Br': [(78.91834, 0.5069), (80.91629, 0.4931)],
}

# -----------------------------
# Count atoms in formula
# -----------------------------
def count_atom(formula, atom):
    if pd.isna(formula):
        return 0
    matches = re.findall(f"{atom}(\\d*)", formula)
    return sum(int(x) if x else 1 for x in matches) if matches else 0

# -----------------------------
# Calculate isotopic pattern
# -----------------------------
def calculate_isotopic_pattern(element_counts, abundance_threshold=0.001, max_m_plus_n=7):
    """
    element_counts: dict, e.g., {'C': 6, 'Cl': 1, 'Br': 1}
    Returns:
      filtered_abundance: list of relative abundances >= threshold
      m_plus_n_abundance: list of abundances for M0 ~ M6
    """
    isotopic_pattern = [(0, 1.0)]  # initial pattern

    for element, count in element_counts.items():
        if element not in isotopes_ratios or count == 0:
            continue
        element_isotopes = isotopes_ratios[element]
        for _ in range(count):
            new_pattern = [
                (mass1 + mass2, abun1 * abun2)
                for mass1, abun1 in isotopic_pattern
                for mass2, abun2 in element_isotopes
                if abun1 * abun2 > abundance_threshold
            ]
            merged = {}
            for mass, abun in new_pattern:
                key = round(mass, 6)
                merged[key] = merged.get(key, 0) + abun
            isotopic_pattern = list(merged.items())

    isotopic_pattern.sort()
    max_abun = max(abun for mass, abun in isotopic_pattern)
    normalized_pattern = [(mass, (abun / max_abun) * 100) for mass, abun in isotopic_pattern]
    filtered_pattern = [(mass, abun) for mass, abun in normalized_pattern if abun >= 0.001]
    filtered_abundance = [abun for mass, abun in filtered_pattern]

    m_plus_n = [(0.0, 0.0)] * max_m_plus_n
    if filtered_pattern:
        base_mass = filtered_pattern[0][0]
        for mass, abun in filtered_pattern:
            index = int(round(mass - base_mass))
            if 0 <= index < max_m_plus_n:
                combined_abun = m_plus_n[index][1] + abun
                m_plus_n[index] = (mass, combined_abun)

    m_plus_n_abundance = [abun for mass, abun in m_plus_n]
    return filtered_abundance, m_plus_n_abundance

# -----------------------------
# Update isotopic patterns in CSV files
# -----------------------------
base_dir = r"D:\MSMS\MSMS_ClBr_H+H-\Train_data"

for fname in os.listdir(base_dir):
    if not fname.endswith(".csv"):
        continue
    fpath = os.path.join(base_dir, fname)
    try:
        df = pd.read_csv(fpath)
        if "Formula" not in df.columns:
            print(f"Skipped {fpath} (no 'Formula' column)")
            continue

        df['C_count'] = df['Formula'].apply(lambda f: count_atom(f, 'C'))
        df['Cl_count'] = df['Formula'].apply(lambda f: count_atom(f, 'Cl'))
        df['Br_count'] = df['Formula'].apply(lambda f: count_atom(f, 'Br'))

        def calc_pattern(row):
            element_counts = {'C': row['C_count'], 'Cl': row['Cl_count'], 'Br': row['Br_count']}
            _, m_plus_n = calculate_isotopic_pattern(element_counts)
            return m_plus_n

        df['isotope_pattern_M0_M6'] = df.apply(calc_pattern, axis=1)
        df.to_csv(fpath, index=False)
        print(f"✅ Updated isotopic pattern: {fpath}")

    except Exception as e:
        print(f"❌ Failed to process {fpath}: {e}")