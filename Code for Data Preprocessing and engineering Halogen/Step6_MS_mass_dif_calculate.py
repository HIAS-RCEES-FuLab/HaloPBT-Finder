import os
import pandas as pd
import ast
import itertools

base_dir = r"D:\MSMS\MSMS_ClBr_H+H-\Train_data"
TOP_N = 10
BIN_START = 0.0
BIN_END = 500.0
BIN_STEP = 0.01

def get_bin_index(mz):
    if mz < BIN_START or mz >= BIN_END:
        return None
    return int((mz - BIN_START) / BIN_STEP)

def compute_neutral_losses(spectrum, top_n=TOP_N):
    if len(spectrum) < 2:
        return []
    top_peaks = sorted(spectrum, key=lambda x: x[1], reverse=True)[:top_n]
    mz_values = sorted([mz for mz, _ in top_peaks], reverse=True)
    losses = [round(m1 - m2, 4)
              for i, m1 in enumerate(mz_values)
              for m2 in mz_values[i+1:]
              if 0 < m1 - m2 <= BIN_END]
    return losses

def bin_neutral_losses(losses):
    bin_dict = {}
    for l in losses:
        idx = get_bin_index(l)
        if idx is not None:
            bin_dict[idx] = l
    return sorted(bin_dict.items(), key=lambda x: x[0])

for fname in os.listdir(base_dir):
    if not fname.endswith(".csv"):
        continue

    fpath = os.path.join(base_dir, fname)
    df = pd.read_csv(fpath)

    if "Mass_spectral" not in df.columns:
        print(f"Skip {fpath}, no 'Mass_spectral' column.")
        continue

    binned_nl_list = []

    for idx, spec_str in enumerate(df["Mass_spectral"].dropna()):
        try:
            spectrum = ast.literal_eval(spec_str)
            spectrum_checked = [
                (float(mz), float(inten)) for mz, inten in spectrum
                if isinstance(mz, (int, float)) and isinstance(inten, (int, float))
            ]
            if not spectrum_checked:
                binned_nl_list.append("[]")
                continue

            losses = compute_neutral_losses(spectrum_checked, top_n=TOP_N)
            binned = bin_neutral_losses(losses)
            binned_nl_list.append(str(binned))

        except Exception as e:
            print(f"Parse error in {fpath} at row {idx}: {e}")
            binned_nl_list.append("[]")

    df["Neutral_losses_binned"] = binned_nl_list
    df.to_csv(fpath, index=False)
    print(f"Updated Neutral_losses_binned in file: {fpath}")