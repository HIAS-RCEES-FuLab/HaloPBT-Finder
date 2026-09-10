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

# --- 处理目录 ---
base_dir = r"D:\MSMS\MSMS_PBT\MS2_PBT_database_H+H-\Train_data"

# 设置质荷比范围和 bin
mz_lower = 50
mz_upper = 1000
mz_bin_width = 0.01
mz_bins = np.arange(mz_lower, mz_upper + mz_bin_width, mz_bin_width)

db_min_bin = float("inf")
db_max_bin = -float("inf")

for fname in os.listdir(base_dir):
    if not fname.endswith(".csv"):
        continue

    fpath = os.path.join(base_dir, fname)
    df = pd.read_csv(fpath)

    if "Mass_spectral" not in df.columns:
        print(f"⚠️ Skip {fpath}, no 'Mass_spectral' column.")
        continue

    new_spectra_full = []

    for spec_str in df["Mass_spectral"].dropna():
        try:
            spectrum = ast.literal_eval(spec_str)
            spectrum_checked = [
                (float(mz), float(i)) for mz, i in spectrum
                if isinstance(mz, (int, float)) and isinstance(i, (int, float))
            ]

            if not spectrum_checked:
                new_spectra_full.append("[]")
                continue

            # --- 只取强度前50个峰 ---
            spectrum_checked.sort(key=lambda x: x[1], reverse=True)
            spectrum_top = spectrum_checked[:50]
            global_max_intensity = max(inten for _, inten in spectrum_top)

            # --- binning ---
            binned = {}
            for mz, inten in spectrum_top:
                if mz_lower <= mz <= mz_upper:
                    idx = int((mz - mz_lower) / mz_bin_width)
                    if idx not in binned:
                        binned[idx] = []
                    binned[idx].append((mz, inten))

            if not binned:
                new_spectra_full.append("[]")
                continue

            # 更新全局 bin 范围
            min_bin = min(binned.keys())
            max_bin = max(binned.keys())
            db_min_bin = min(db_min_bin, min_bin)
            db_max_bin = max(db_max_bin, max_bin)

            # --- 生成 (bin_index, m/z, normalized_intensity, KMD_Cl, KMD_Br) ---
            full_features = []
            for idx in sorted(binned.keys()):
                for mz_val, inten in binned[idx]:
                    norm_inten = round(inten / global_max_intensity, 4) if global_max_intensity > 0 else 0
                    kmd_cl = round(kendrick_mass_defect(mz_val, "Cl")["KMD"], 6)
                    kmd_br = round(kendrick_mass_defect(mz_val, "Br")["KMD"], 6)
                    full_features.append((idx, mz_val, norm_inten, kmd_cl, kmd_br))

            new_spectra_full.append(str(full_features))

        except Exception as e:
            print(f"❌ Parse error in {fpath}: {e}")
            new_spectra_full.append(spec_str)

    # 保存新列
    df["Mass_spectral_features"] = new_spectra_full
    df.to_csv(fpath, index=False)
    print(f"✅ Updated file: {fpath}")

# 打印统一的 bin 范围
if db_min_bin != float("inf"):
    print(f"📌 全部文件中出现的 bin index 范围: {db_min_bin} - {db_max_bin}")
