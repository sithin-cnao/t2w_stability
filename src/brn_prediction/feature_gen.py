
#%%
import SimpleITK as sitk
import pandas as pd
import os
from tqdm import tqdm

from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor, as_completed
NUM_WORKERS = 16

DATA_DIR = r"/home/thulasiseetha/research/dataset/curated/ChordomaMR/data"
DB_FILE = r"/home/thulasiseetha/research/dataset/curated/ChordomaMR/db_df.csv"
PARAM_FILE = "/home/thulasiseetha/research/sithin_projects/t2w_stability/src/paramSetting/StudySettings3D.yaml"
OUT_DIR = r"/home/thulasiseetha/research/dataset/curated/ChordomaMR"

SOI = ["t2w_fs", "t2w_nfs"]
ROI = {
    'csf':[1],
    'gm':[2], #select 2 and do not include anything else; 2 = Gray Matter
    'wm':[3], #select 3 and do not include anything else; 3 = White Matter
    'gwm':[2,3], 
    'brain':[1,2,3,4] #In FAST FSL, if you are segmenting T2-weighted images, you may need to select 4 classes so that dark non-brain matter is processed correctly
}
TISSUE = "gwm"


def extract_features(db_row):

    row_dict = db_row.to_dict()

    featureVectors = []

    pid = row_dict["pid"]
    sequence = row_dict["sequence"]
    
    data_dir = os.path.join(DATA_DIR, pid, sequence)
    if os.path.exists(data_dir):
        sitk_img = sitk.ReadImage(os.path.join(data_dir, "img.nii.gz"))
        sitk_mask = sitk.ReadImage(os.path.join(data_dir, "mask_tissue_seg.nii.gz"))
        tissue_mask = sum([sitk.BinaryThreshold(sitk_mask, label, label, insideValue=True, outsideValue=False) for label in ROI[TISSUE]])
        tissue_mask = sitk.Cast(tissue_mask, sitk.sitkUInt8)

        extractor = featureextractor.RadiomicsFeatureExtractor(PARAM_FILE)
        featureVector = extractor.execute(sitk_img, tissue_mask, label=1)
        featureVector = {**row_dict, **featureVector}
        featureVectors.append(featureVector)

    return featureVectors



if __name__=="__main__":

    db = pd.read_csv(DB_FILE)

    radiomics_df = []

    with ProcessPoolExecutor(NUM_WORKERS) as e:
        futures = [e.submit(extract_features, row) for idx, row in db.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features", position=0):
            featureVectors = future.result()
            radiomics_df.extend(featureVectors)

    radiomics_df = pd.DataFrame(radiomics_df)
    radiomics_df.to_csv(os.path.join(OUT_DIR, "radiomicsFeatures3D.csv"), index=False)
    
