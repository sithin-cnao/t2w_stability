# Exploring the Impact of T2-weighted MRI Fat-Saturation on Radiomics Stability for Brain Radionecrosis Prediction after Skull-Base Proton Therapy: A Pilot Study

Original repository supporting the article submitted to Cancers [MDPI] (@Citation TBA)

### **Repository structure**
#### **Overview:**

* Automated skullstripping (brain extraction) and tissue segmentation methods
* Rigid registration of segmentation using SimpleITK
* Radiomics Feature Extraction using Pyradiomics
* Image-level Processing Methods: N4 BFC, Normalization
* Feature-level Processing Method: Combat
* Optimal Processing Config Analysis
* Stability analysis
* BRN Prediction Analysis
* Statistical Analysis
* Visualizations
* Jupyter notebooks
* Python Scripts

#### **Contents:**
```
t2w_stability
├── src/                                   
│    └── data_curation/                                         # Contains files associated with the data curation for the stability study
|          └── mask_curation.ipynb                              # Generating T2w-nfs masks, rigid-registration of T2w-nfs to T2w-fs images
|          └── tissue_segmentation_methods.ipynb                # All automatic brain, tissue segmentation methods explored;
|          └── gui.py, registration_gui.py                      # Utility files
│    └── paramSetting/
|          └── StudySettings3D.yaml                             # Radiomics feature extraction settings for both stability study and BRN prediction modelling
│    └── stability_analysis/
|          └── feature_extraction3D.ipynb                       # Radiomics feature extraction from tissues
|          └── stability_analysis.ipynb                         # Stability quantification using CCC
|          └── config_analysis.py                               # Exploring 40 processing configurations, Uncertainty using Statistical Analysis
|          └── utils.py, gui.py, registration_gui.py            # Utility files
│    └── brn_prediction/
|          └── data_prep.py                                     # Data preparation for BRN predictive modeling
|          └── mask_gen.py                                      # Tissue mask generation independently for T2w-nfs and T2w-fs
|          └── feature_gen.py                                   # Radiomics feature extraction from gwm tissue
|          └── analysis.py                                      # 5-times repeated 5-fold cross-validation
|          └── utils.py                                         # Utilities
├── LICENCE                                                     # GNU General Public License v3.0
├── README.md

```
