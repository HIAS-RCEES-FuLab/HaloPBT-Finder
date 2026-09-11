# HaloPBT-Finder

HaloPBT-Finder is an integrated platform for screening of chlorinated and brominated organic compounds (Cl/Br-HOCs), comprising an interpretable multimodal neural network and downstream structural annotation.

This repository provides the implementation of the HaloPBT-Finder model and the executable platform, including data preprocessing, feature engineering, model training, model interpretation, and result visualization. It also provides the source data and visualization scripts used to generate the results presented in the associated study.

<img width="1914" height="1331" alt="图片1" src="https://github.com/user-attachments/assets/aab50385-6c3a-47c8-9467-b355eb43b3df" />

---

## Spectral Libraries

The spectral databases used in this project include:

- **NIST 2020 MS/MS Library**
  - The library can be exported by following the MassFormer instructions:
  - https://github.com/Roestlab/massformer?tab=readme-ov-file#exporting-the-nist-data

- **GNPS Spectral Libraries**
  - Available at:
  - https://external.gnps2.org/gnpslibrary

- **MoNA (MassBank of North America)**
  - Available at:
  - https://mona.fiehnlab.ucdavis.edu/downloads

## Hardware Requirements

HaloPBT-Finder was developed and tested on a personal computer with the following configuration:

- Operating system: Windows 11 64-bit
- Processor: Intel(R) Core(TM) Ultra 7 258V (2.20 GHz), x64-based processor
- Memory: 32.0 GB RAM (31.5 GB available)
- GPU: Not required. The current version supports CPU-based inference.
- Storage: Approximately 2 GB of free storage space is sufficient for the executable program, models, and example files.

This configuration is sufficient for HaloPBT-Finder development and routine Cl/Br-HOC screening workflows. Additional storage space may be required when processing user-provided MS files or large spectral libraries.

## Software Requirements

HaloPBT-Finder was developed using PyCharm 2024.3.4 with Python 3.9. The required Python dependencies are provided in the `requirements.txt` file.

The dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Model Reproduction Workflow

The model reproduction workflow is organized into three main folders:

1. **Code for Database Extraction and Labeling**
   
   This folder contains the scripts for extracting data from different databases, standardizing metadata, and merging the processed datasets. The internal steps are explicitly labeled. Users can execute the scripts sequentially according to the provided step numbers and modify the data paths according to their local environment.

2. **Code for Data Preprocessing and Engineering**
   
   This folder contains two subfolders for the preprocessing and feature engineering of PBT and halogen data, respectively. The internal steps in each subfolder are explicitly numbered and should be executed sequentially after completing the database extraction and labeling. Users only need to update the relevant data paths according to their local environment before running the scripts.

3. **Code for Model Training and Interpretability Analysis**
   
   This folder contains the scripts for model development and evaluation, including training and testing of the proposed model, baseline models, and ablation models, as well as downstream model interpretability analysis.

By following the three folders in order, users can reproduce the complete model development workflow, from database extraction and compound labeling, through data preprocessing and feature engineering for PBT and halogen classification, to model training, evaluation, and interpretability analysis.

---

## Platform

The graphical user interface of HaloPBT-Finder is shown below.

<img width="725" height="662" alt="fig" src="https://github.com/user-attachments/assets/0d623412-8676-43af-8e10-01cf83b05d9e" />

The platform consists of two main modules: **Peak Feature Mining** and **Identification**.

### Peak Feature Mining Parameter Settings

This module is used for MS feature extraction and preprocessing.

- **Scan mode**
  - **DDA** is the recommended acquisition mode because each MS/MS spectrum is directly associated with its precursor ion.
  - **DIA** and **Full Scan** data are also supported. For DIA data, HaloPBT-Finder performs model prediction using the acquired window-based MS/MS spectra because precursor-specific MS/MS spectra are unavailable. For Full Scan data, where no MS/MS spectra are acquired, the model utilizes the acquired full-scan MS spectra and potential in-source fragmentation ions for prediction.

- **Ion mode**
  - Select Positive or Negative according to the ionization mode of the experimental data.

- **Noise threshold**
  - Defines the intensity threshold for noise removal.
  - This parameter should be adjusted according to the performance and noise characteristics of the mass spectrometer used.

- **Extract MS/MS**
  - Extracts MS/MS spectra associated with detected features for downstream analysis.

- **Single-trace filtering**
  - Uses the OpenMS single-trace filtering algorithm to remove low-confidence features and reduce potential false-positive peaks.

- **Adduct annotation**
  - Annotates supported adduct ions according to predefined adduct rules.

- **Feature filtering**
  - Uses machine learning models to simultaneously predict halogen content and PBT properties.

### Identification Parameter Settings

This module performs hierarchical compound identification.

1. **Exact mass and isotope pattern matching**.
2. **Theoretical fragment calculation**. The fragmentation tree depth determines the level of theoretical fragmentation. Increasing the tree depth produces more predicted fragments for matching.
3. **MS/MS spectral matching** against reference spectra.

All matching parameters can be customized according to the analytical requirements.

### Input and Output Settings

This module is used to configure the input mass spectrometry files and the output directory.

- **File input**
  - HaloPBT-Finder currently supports standardized **mzML** format files.
  - Click **Browse** to navigate to the directory containing the target MS files.
  - Click **Add File** to add the selected file(s) to the processing queue.

- **Output path**
  - Click **Browse** to specify the directory for saving analysis results.

- **Analysis processing**
  - After configuring all parameters, click **Start** to begin processing. The platform will automatically execute the complete workflow, including feature mining, feature filtering, compound identification, and result generation.
