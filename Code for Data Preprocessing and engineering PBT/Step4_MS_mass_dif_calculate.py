import os
import pandas as pd
import ast
import itertools

base_dir = r"D:\MSMS\MSMS_PBT\MS2_PBT_database_H+H-\Train_data"
TOP_N = 10
BIN_START = 0.0
BIN_END = 500.0
BIN_STEP = 0.01

def get_bin_index(mz):
    if mz < BIN_START or mz >= BIN_END:
        return None
    return int((mz - BIN_START) / BIN_STEP)

def compute_neutral_losses(spectrum):
    if len(spectrum) < 2:
        return []
    mz_values = sorted([mz for mz, _ in spectrum], reverse=True)
    losses = []
    for mz1, mz2 in itertools.combinations(mz_values, 2):
        loss = mz1 - mz2
        if 0 < loss <= BIN_END:
            losses.append(round(loss, 4))
    return losses

for fname in os.listdir(base_dir):
    if not fname.endswith(".csv"):
        continue

    fpath = os.path.join(base_dir, fname)
    df = pd.read_csv(fpath)

    if "Mass_spectral" not in df.columns:
        print(f"Skip {fpath}, no 'Mass_spectral' column.")
        continue

    binned_nl_list = []

    for spec_str in df["Mass_spectral"].dropna():
        try:
            spectrum = ast.literal_eval(spec_str)
            spectrum_checked = [
                (float(mz), float(inten))
                for mz, inten in spectrum
                if isinstance(mz, (int, float)) and isinstance(inten, (int, float))
            ]
            if not spectrum_checked:
                binned_nl_list.append("[]")
                continue

            spectrum_top = sorted(spectrum_checked, key=lambda x: x[1], reverse=True)[:TOP_N]
            losses = compute_neutral_losses(spectrum_top)

            bin_dict = {}
            for l in losses:
                bin_idx = get_bin_index(l)
                if bin_idx is not None:
                    bin_dict[bin_idx] = l

            binned = sorted(bin_dict.items(), key=lambda x: x[0])
            binned_nl_list.append(str(binned))

        except Exception as e:
            print(f"Parse error in {fpath}: {e}")
            binned_nl_list.append("[]")

    df["Neutral_losses_binned"] = binned_nl_list
    df.to_csv(fpath, index=False)
    print(f"Updated Neutral_losses_binned in file: {fpath}")