import os
import pandas as pd
import numpy as np
import ast
import re

# --- Atomic exact masses ---
isotope_mass = {
    'H': 1.0078250322, 'C': 12.0, 'N': 14.0030740048,
    'O': 15.9949146196, 'F': 18.998403, 'P': 30.9737619985,
    'S': 31.9720711744, 'Cl': 34.968852682, 'Br': 78.91834,
    'I': 126.904473
}

def calc_formula_mass(formula):
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
    exact_unit_mass = calc_formula_mass(unit_str)
    nominal_unit_mass = round(exact_unit_mass)
    km = exact_mass * nominal_unit_mass / exact_unit_mass
    kmd = round(km) - km
    return {'Exact_mass': exact_mass, 'KM': km, 'KMD': kmd}

# --- Configuration ---
base_dir = r"D:\MSMS\MSMS_ClBr_H+H-\Train_data"
mz_lower, mz_upper = 50, 1000
mz_bin_width = 0.01

# --- Global bin range ---
db_min_bin, db_max_bin = float("inf"), -float("inf")

# --- Process CSV files ---
for fname in os.listdir(base_dir):
    if not fname.endswith(".csv"):
        continue
    fpath = os.path.join(base_dir, fname)
    try:
        df = pd.read_csv(fpath)
        if "Mass_spectral" not in df.columns:
            print(f"⚠️ Skip {fpath}, no 'Mass_spectral' column.")
            continue

        spectra_features = []

        for spec_str in df["Mass_spectral"].dropna():
            try:
                spectrum = ast.literal_eval(spec_str)
                # Filter valid peaks
                spectrum_checked = [(float(mz), float(inten)) for mz, inten in spectrum
                                    if isinstance(mz, (int, float)) and isinstance(inten, (int, float))]
                if not spectrum_checked:
                    spectra_features.append("[]")
                    continue

                # Take top 50 by intensity
                spectrum_checked.sort(key=lambda x: x[1], reverse=True)
                top_peaks = spectrum_checked[:50]
                max_intensity = max(inten for _, inten in top_peaks)

                # Bin peaks
                binned = {}
                for mz, inten in top_peaks:
                    if mz_lower <= mz <= mz_upper:
                        idx = int((mz - mz_lower) / mz_bin_width)
                        binned.setdefault(idx, []).append((mz, inten))

                if not binned:
                    spectra_features.append("[]")
                    continue

                # Update global bin range
                min_bin, max_bin = min(binned.keys()), max(binned.keys())
                db_min_bin, db_max_bin = min(db_min_bin, min_bin), max(db_max_bin, max_bin)

                # Generate features: (bin_index, mz, norm_intensity, KMD_Cl, KMD_Br)
                feature_list = []
                for idx in sorted(binned.keys()):
                    for mz_val, inten in binned[idx]:
                        norm_inten = round(inten / max_intensity, 4) if max_intensity > 0 else 0
                        kmd_cl = round(kendrick_mass_defect(mz_val, "Cl")["KMD"], 6)
                        kmd_br = round(kendrick_mass_defect(mz_val, "Br")["KMD"], 6)
                        feature_list.append((idx, mz_val, norm_inten, kmd_cl, kmd_br))

                spectra_features.append(str(feature_list))

            except Exception as e:
                print(f"❌ Parse error in {fpath}: {e}")
                spectra_features.append(spec_str)

        # Add new column and save
        df["Mass_spectral_features"] = spectra_features
        df.to_csv(fpath, index=False)
        print(f"✅ Updated file: {fpath}")

    except Exception as e:
        print(f"❌ Failed to process {fpath}: {e}")

# Print global bin index range
if db_min_bin != float("inf"):
    print(f"📌 Global bin index range across all files: {db_min_bin} - {db_max_bin}")