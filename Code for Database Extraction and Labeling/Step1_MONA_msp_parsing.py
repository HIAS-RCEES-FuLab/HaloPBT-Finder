import re
import pandas as pd

# Set the input file path and output directory
msp_file = "MSMS_rawdata/mb_na_msms.msp"
output_csv = "MSMS_extract/mona_data_"

# Define regular expression extraction rules for each object (metadata)
patterns = {
    "DB_ID": r"DB#:\s*(.*?)\s*(?=\n|$)",
    "Name": r"Name:\s*(.*?)\s*(?=\n|$)",
    "Precursor_type": r"Precursor_type:\s*(.*?)\s*(?=\n|$)",
    "Precursor_mz": r"PrecursorMZ:\s*(.*?)\s*(?=\n|$)",
    "Formula": r"Formula:\s*(.*?)\s*(?=\n|$)",
    "InChIKey": r"InChIKey:\s*(.*?)\s*(?=\n|$)",
    "MW": r"MW:\s*(.*?)\s*(?=\n|$)",
    "Exact_mass": r"ExactMass:\s*(.*?)\s*(?=\n|$)",
    "Instrument_type": r"Instrument_type:\s*(.*?)\s*(?=\n|$)",
    "Instrument": r"Instrument:\s*(.*?)\s*(?=\n|$)",
    "Collision_energy": r"Collision_energy:\s*(.*?)\s*(?=\n|$)",
    "Ion_mode": r"Ion_mode:\s*(.*?)\s*(?=\n|$)",
    "Comments": r"Comments:\s*(.*?)\s*(?=\n|$)",
    "Num_peaks": r"Num Peaks:\s*(\d+)\s*(?=\n|$)"
}

# Define target column order (metadata storage order)
new_column_order = [
    'ID', 'Name', 'Precursor_type', 'Precursor_mz', 'Instrument_type',
    'Instrument', 'Ionization', 'Collision_energy', 'Ion_mode',
    'InChIKey', 'SMILES', 'Pubchem_cid', 'Formula', 'MW', 'Exact_mass',
    'CAS_Num', 'DB_ID',
    'Num_peaks', 'Mass_spectral',
]

# Parameter settings and initialization
pos_data = []
neg_data = []
pos_count = 0
neg_count = 0
batch_size = 10000
pos_file_count = 1
neg_file_count = 1
ID=0

# Read the MSP file line by line for processing
with open(msp_file, 'r', encoding='utf-8') as file:
    spectrum = ""
    spectrum_data = {}
    empty_lines_count = 0
    for line in file:
        line = line.strip()

        if not line:
            empty_lines_count += 1

            if empty_lines_count >= 2:
                ID += 1
                if spectrum:
                    spectrum_data["ID"] = ID
                    for pattern_name, pattern in patterns.items():
                        matches = re.findall(pattern, spectrum, re.DOTALL)
                        if matches:
                            spectrum_data[pattern_name] = matches[0].strip()
                        else:
                            spectrum_data[pattern_name] = ""

                    ion_mode = spectrum_data.get("Ion_mode", "").strip().upper()

                    mass_spectral_peaks_pattern = r"(\d+\.\d+)\s+(\d+\.\d+)"
                    peaks_data = re.findall(mass_spectral_peaks_pattern, spectrum)

                    num_peaks = int(spectrum_data.get("Num_peaks", 0))
                    peaks_list = [(float(mz), round(float(intensity), 1)) for mz, intensity in peaks_data[:num_peaks] if
                                  float(intensity) >= 1.0]

                    spectrum_data["Mass_spectral"] = peaks_list

                    comments_text = spectrum_data.get("Comments", 0)
                    cas_pattern = r'cas="?([0-9\-]+)"?'
                    ionization_pattern = r'ionization="?([A-Za-z]+)"?'
                    smiles_pattern = r'SMILES="?([^\s";]+)"?'
                    cid_pattern = r'pubchem cid="?(\d+)"?'
                    cas_match = re.search(cas_pattern, comments_text)
                    ionization_match = re.search(ionization_pattern, comments_text)
                    smiles_match = re.search(smiles_pattern, comments_text)
                    cid_match = re.search(cid_pattern, comments_text)
                    cas_value = cas_match.group(1) if cas_match else None
                    ionization_value = ionization_match.group(1) if ionization_match else None
                    smiles_value = smiles_match.group(1) if smiles_match else None
                    cid_value = cid_match.group(1) if cid_match else None
                    spectrum_data["CAS_Num"] = cas_value.replace("-", "") if cas_value else None
                    spectrum_data["Ionization"] = ionization_value
                    spectrum_data["SMILES"] = smiles_value
                    spectrum_data["Pubchem_cid"] = cid_value

                    if ion_mode == "P":
                        pos_data.append(spectrum_data)
                        pos_count += 1
                        if pos_count >= batch_size:
                            current_output_csv = f"{output_csv}POS_{pos_file_count}.csv"
                            df = pd.DataFrame(pos_data)
                            df = df[new_column_order]
                            df.to_csv(current_output_csv, index=False, encoding='utf-8')
                            pos_data = []
                            pos_count = 0
                            pos_file_count += 1

                    elif ion_mode == "N":
                        neg_data.append(spectrum_data)
                        neg_count += 1
                        if neg_count >= batch_size:
                            current_output_csv = f"{output_csv}NEG_{neg_file_count}.csv"
                            df = pd.DataFrame(neg_data)
                            df = df[new_column_order]
                            df.to_csv(current_output_csv, index=False, encoding='utf-8')
                            neg_data = []
                            neg_count = 0
                            neg_file_count += 1

                spectrum = ""
                spectrum_data = {}
                empty_lines_count = 0

        else:
            empty_lines_count = 0
            spectrum += line + "\n"

if pos_data:
    current_output_csv = f"{output_csv}POS_{pos_file_count}.csv"
    df = pd.DataFrame(pos_data)
    df = df[new_column_order]
    df.to_csv(current_output_csv, index=False, encoding='utf-8')

if neg_data:
    current_output_csv = f"{output_csv}NEG_{neg_file_count}.csv"
    df = pd.DataFrame(neg_data)
    df = df[new_column_order]
    df.to_csv(current_output_csv, index=False, encoding='utf-8')