import re
import pandas as pd
from rdkit import Chem

# Set the input file path and output directory
sdf_file = "MSMS_rawdata/hr_msms_nist.SDF"
output_csv = "MSMS_extract/nist_data_test_"

# Define regular expression extraction rules for each object (metadata)
patterns = {
    "ID": r"<ID>\s*(.*?)\s*>",
    "Name": r"<NAME>\s*(.*?)\s*>",
    "Precursor_type": r"<PRECURSOR TYPE>\s*(.*?)\s*>",
    "Precursor_mz": r"<PRECURSOR M/Z>\s*(.*?)\s*>",
    "Instrument_type": r"<INSTRUMENT TYPE>\s*(.*?)\s*>",
    "Instrument": r"<INSTRUMENT>\s*(.*?)\s*>",
    "Ionization": r"<IONIZATION>\s*(.*?)\s*>",
    "Collision_energy": r"<COLLISION ENERGY>\s*(.*?)\s*>",
    "Ion_mode": r"<ION MODE>\s*(.*?)\s*>",
    "InChIKey": r"<INCHIKEY>\s*(.*?)\s*>",
    "SMILES": r"<SMILES>\s*(.*?)\s*>",
    "Pubchem_cid": r"<PUBCHEM CID>\s*(.*?)\s*>",
    "Formula": r"<FORMULA>\s*(.*?)\s*>",
    "MW": r"<MW>\s*(.*?)\s*>",
    "Exact_mass": r"<EXACT MASS>\s*(.*?)\s*>",
    "CAS_Num": r"<CASNO>\s*(.*?)\s*>",
    "DB_ID": r"<NISTNO>\s*(.*?)\s*>",
    "Num_peaks": r"<NUM PEAKS>\s*(.*?)\s*>"
}

# Parameter settings and initialization
pos_data = []
neg_data = []
pos_count = 0
neg_count = 0
batch_size = 10000
pos_file_count = 1
neg_file_count = 1

# Read the SDF file line by line for processing
with open(sdf_file, 'r', encoding='utf-8') as file:
    spectrum = ""
    for line in file:
        if line.strip() == "$$$$":
            if spectrum:
                spectrum_data = {}
                for pattern_name, pattern in patterns.items():
                    matches = re.findall(pattern, spectrum, re.DOTALL)
                    if matches:
                        spectrum_data[pattern_name] = matches[0].strip()
                    else:
                        spectrum_data[pattern_name] = ""
                mol_block_match = re.search(r"^(.*?^\s*M\s+END\s*)", spectrum, re.MULTILINE | re.DOTALL)
                if mol_block_match:
                    mol_block = mol_block_match.group(1)
                    try:
                        mol = Chem.MolFromMolBlock(mol_block, sanitize=False)
                        if mol:
                            Chem.SanitizeMol(mol)
                            smiles = Chem.MolToSmiles(mol)
                        else:
                            smiles = ""
                    except Exception:
                        smiles = ""
                else:
                    smiles = ""
                spectrum_data["SMILES"] = smiles

                ion_mode = spectrum_data.get("Ion_mode", "").strip().upper()

                mass_spectral_peaks_pattern = r"<MASS SPECTRAL PEAKS>(.*)"
                mass_spectral_peaks_match = re.search(mass_spectral_peaks_pattern, spectrum, re.DOTALL)
                if mass_spectral_peaks_match:
                    peaks_data = mass_spectral_peaks_match.group(1).strip().splitlines()
                    peaks_list = []
                    num_peaks = int(spectrum_data.get("Num_peaks", 0))
                    for peak in peaks_data[:num_peaks]:
                        parts=peak.split()
                        if len(parts) >= 2:
                            mz = float(parts[0])
                            intensity = float(parts[1]) / 9.99
                            intensity = round(intensity, 1)

                            if intensity >= 1:
                                peaks_list.append((mz, intensity))

                    spectrum_data["Mass_spectral"] = peaks_list

                if ion_mode == "P":
                    pos_data.append(spectrum_data)
                    pos_count += 1
                    if pos_count >= batch_size:
                        current_output_csv = f"{output_csv}POS_{pos_file_count}.csv"
                        df = pd.DataFrame(pos_data)
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
                        df.to_csv(current_output_csv, index=False, encoding='utf-8')
                        neg_data = []
                        neg_count = 0
                        neg_file_count += 1

            spectrum = ""
        else:
            spectrum += line

if pos_data:
    current_output_csv = f"{output_csv}POS_{pos_file_count}.csv"
    df = pd.DataFrame(pos_data)
    df.to_csv(current_output_csv, index=False, encoding='utf-8')

if neg_data:
    current_output_csv = f"{output_csv}NEG_{neg_file_count}.csv"
    df = pd.DataFrame(neg_data)
    df.to_csv(current_output_csv, index=False, encoding='utf-8')