# Evaluating the impact of fat-saturation on radiomics feature stability in brain T2-weighted Magnetic Resonance Imaging

Original repository supporting the article submitted to Physics and Imaging in Radiation Oncology journal (@Citation TBA)

### **Repository structure**
#### **Overview:**

* Automated skullstripping (brain extraction) and tissue segmentation methods
* Rigid registration of segmentation using SimpleITK
* Image-level Processing Methods: N4 BFC, Normalization
* Radiomics Feature Extraction using Pyradiomics
* ComBat harmonization
* Stability analysis
* Statistical Analysis
* Visualizations
* Jupyter notebooks with usage examples

#### **Contents:**
```
t2w_stability
├── curation                                  # Contains automatic brain, tissue segmentation methods explored; segmented mask coregistration >< T2w-fs and T2w-nfs
│    └── tissue_segmentation_methods.ipynb    # Jupyter notebook demonstrating various combinations of automated brain extraction and tissue segmentation methods on T2w-nfs images  
│    └── mask_curation.ipynb                  # Coregistration of segmented tissues to tumor-focalized T2w-fs images; Adjusting T2w-nfs images to match T2w-fs images
│    └── registration_gui.py, gui.py          # Utilities for image visualization copied from https://github.com/SimpleITK/TUTORIAL/blob/main/05_basic_registration.ipynb

├── analysis                                  # Radiomics Feature Extraction + Combat harmonization + CCC analysis
│    └── paramSettings                        # Pyradiomics feature extraction setting file, originally copied from https://github.com/AIM-Harvard/pyradiomics/tree/master/examples/exampleSettings
|          └── StudySettings3D.yaml           # This is the actual configuration file we used for feature extraction
|          └── StudySettings3DVoxel.yaml      # NA
|          └── exampleMR_3mm.yaml             # NA
│    └── feature_extraction3D.ipynb           # Jupyter notebook demonstrating feature extraction after subjecting T2w-fs/nfs image to image-level processing methods: N4 BFC, Normalization
│    └── stability_analysis.ipynb             # Jupyter notebook presenting ComBat harmonization and CCC estimation of features
│    └── statistical_analysis.ipynb           # Jupyter notebook presenting MC Power Analysis, Impact of TR, and regression analysis
│    └── registration_gui.py, gui.py          # Utilities for image visualization copied from https://github.com/SimpleITK/TUTORIAL/blob/main/05_basic_registration.ipynb

├── LICENCE                                   # GNU General Public License v3.0
├── README.md

```
