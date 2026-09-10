HaloPBT-Finder

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
