import os
import pandas as pd
import re

# --- Atomic exact masses ---
isotope_mass = {
    'H': 1.0078250322, 'C': 12.0, 'N': 14.0030740048,
    'O': 15.9949146196, 'F': 18.998403, 'P': 30.9737619985,
    'S': 31.9720711744, 'Cl': 34.968852682, 'Br': 78.91834,
    'I': 126.904473
}

def calc_formula_mass(formula):
    """Calculate exact mass from a chemical formula."""
    mass = 0.0
    pattern = r'([A-Z][a-z]*)(\d*)'
    for elem, count in re.findall(pattern, formula):
        count = int(count) if count else 1
        if elem not in isotope_mass:
            print(f"⚠ Warning: Unknown element '{elem}' in formula '{formula}' — ignored")
            continue
        mass += isotope_mass[elem] * count
    return mass

def kendrick_mass_defect(exact_mass, unit_str="CF2"):
    """Compute Kendrick Mass (KM) and Kendrick Mass Defect (KMD)."""
    exact_unit_mass = calc_formula_mass(unit_str)
    nominal_unit_mass = round(exact_unit_mass)

    km = exact_mass * nominal_unit_mass / exact_unit_mass
    kmd = round(km) - km

    return {
        'Exact_mass': exact_mass,
        'KM': km,
        'KMD': kmd,
        'Exact_unit_mass': exact_unit_mass,
        'Nominal_unit_mass': nominal_unit_mass
    }

# --- Directories ---
base_dir = r"D:\MSMS\MSMS_ClBr_H+H-\Train_data"
output_dir = os.path.join(base_dir, "KMD_updated")
os.makedirs(output_dir, exist_ok=True)

# --- KMD units to calculate ---
kmd_units = ["Cl", "Br"]

# --- Process each CSV ---
for fname in os.listdir(base_dir):
    if not fname.endswith(".csv"):
        continue

    fpath = os.path.join(base_dir, fname)
    try:
        df = pd.read_csv(fpath)

        if "Exact_mass" not in df.columns:
            print(f"Skipped {fpath} (no Exact_mass column)")
            continue

        # --- Compute KMD for each unit ---
        for unit in kmd_units:
            col_name = f"KMD_{unit}"
            df[col_name] = df["Exact_mass"].apply(
                lambda m: kendrick_mass_defect(m, unit_str=unit)["KMD"] if pd.notna(m) else None
            )

        # --- Count Cl and Br atoms in Formula ---
        if "Formula" in df.columns:
            def count_atom(formula, atom):
                if pd.isna(formula):
                    return 0
                matches = re.findall(f"{atom}(\\d*)", formula)
                return sum(int(x) if x else 1 for x in matches) if matches else 0

            df["Cl_count"] = df["Formula"].apply(lambda f: count_atom(f, "Cl"))
            df["Br_count"] = df["Formula"].apply(lambda f: count_atom(f, "Br"))

        # --- Save updated CSV ---
        out_path = os.path.join(output_dir, fname)
        df.to_csv(out_path, index=False)
        print(f"✅ Updated KMD and Cl/Br counts saved: {out_path}")

    except Exception as e:
        print(f"❌ Failed to process {fpath}: {e}")