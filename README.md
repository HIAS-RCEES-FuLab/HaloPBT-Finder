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
