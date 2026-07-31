#%%
import os
import SimpleITK as sitk
from tqdm.auto import tqdm
import shutil

import subprocess
import pandas as pd

from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor, as_completed
NUM_WORKERS = 16

#%%
# tissue segmentation
DATA_DIR = r"/home/thulasiseetha/research/dataset/curated/BrainMR"
OUT_DIR = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/data_curation/reg_analysis/data"

os.makedirs(OUT_DIR, exist_ok=True)
# we already have the registered images, 
# Skull stripping, Tissue segmentation and radiomics feature extraction
def execute_command(command):

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return (command, output, error)

def process(src_dir, dest_dir = None):

    if not dest_dir:
        dest_dir = src_dir
    
    img_file = os.path.join(src_dir, "img.nii.gz")
    reg_mask_file = os.path.join(src_dir, "mask.nii.gz")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    shutil.copy(img_file, os.path.join(dest_dir, "img.nii.gz"))
    shutil.copy(reg_mask_file, os.path.join(dest_dir, "reg_mask_tissue_seg.nii.gz"))

    brain_mask_file = os.path.join(dest_dir, "mask_brain.nii.gz")
    brain_masked_img_file = os.path.join(dest_dir, "img_brain.nii.gz")
    
    command = f"mri_synthstrip -i {img_file} -m {brain_mask_file} -o {brain_masked_img_file}"
    
    command, output, error = execute_command(command)
    
    if not error:
        
        #Also, if you are segmenting T2-weighted images, you may need to select 4 classes so that dark non-brain matter is processed correctly (this is not a problem with T1-weighted as CSF and dark non-brain matter look similar).
    
        tissue_prefix = os.path.join(dest_dir, "mask_tissue")
        command = f"fast -t 2 -o {tissue_prefix} -n 4 {brain_masked_img_file}"

        command, output, error = execute_command(command)
        
    if error:
        print(f"Error executing command '{command}': {error.decode()} for {src_dir}")
        

if __name__ == "__main__":

    pids = os.listdir(os.path.join(DATA_DIR, "t2w_fs"))
    print("total subjects = ", len(pids))

    src_dirs = [os.path.join(DATA_DIR, "t2w_fs", pid) for pid in pids]
    dest_dirs = [os.path.join(OUT_DIR, pid) for pid in pids]
    print("total mr data found = ", len(src_dirs))

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as e:
        futures = [e.submit(process, src_dir, dest_dir) for src_dir, dest_dir in zip(src_dirs, dest_dirs)]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating tissue segmentation masks", position=0):
            res = future.result()

#%%
# reg_fs_tissue_mask vs independent_fs_tissue_mask
DATA_DIR = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/data_curation/reg_analysis/data"
PARAM_FILE = "/home/thulasiseetha/research/sithin_projects/t2w_stability/src/paramSetting/StudySettings3D.yaml"
OUT_DIR = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/data_curation/reg_analysis"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

MASKS = {"registered_mask":"reg_mask_tissue_seg.nii.gz", "independent_mask":"mask_tissue_seg.nii.gz"}

ROI = {
    'csf':[1],
    'gm':[2], #select 2 and do not include anything else; 2 = Gray Matter
    'wm':[3], #select 3 and do not include anything else; 3 = White Matter
    'gwm':[2,3], 
    'brain':[1,2,3,4] #In FAST FSL, if you are segmenting T2-weighted images, you may need to select 4 classes so that dark non-brain matter is processed correctly
}
TISSUE = "gwm"

#%%
# quality check
reg_quality_df = {"pid":[], "dice":[], "hausdorff":[]}


def calculate_metrics(pid):

    reg_mask_file = os.path.join(DATA_DIR, pid, "reg_mask_tissue_seg.nii.gz")
    independent_mask_file = os.path.join(DATA_DIR, pid, "mask_tissue_seg.nii.gz")

    reg_mask = sitk.ReadImage(reg_mask_file)
    independent_mask = sitk.ReadImage(independent_mask_file)

    ind_tissue_mask = sum([sitk.BinaryThreshold(independent_mask, label, label, insideValue=True, outsideValue=False) for label in ROI[TISSUE]])
    ind_tissue_mask = sitk.Cast(ind_tissue_mask, sitk.sitkUInt8)

    reg_tissue_mask = sum([sitk.BinaryThreshold(reg_mask, label, label, insideValue=True, outsideValue=False) for label in ROI[TISSUE]])
    reg_tissue_mask = sitk.Cast(reg_tissue_mask, sitk.sitkUInt8)


    dice_score = sitk.LabelOverlapMeasuresImageFilter()
    dice_score.Execute(ind_tissue_mask, reg_tissue_mask)
    dice = dice_score.GetDiceCoefficient()

    hausdorff_distance = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance.Execute(ind_tissue_mask, reg_tissue_mask)
    hausdorff = hausdorff_distance.GetHausdorffDistance()

    return {"pid":pid, "dice":dice, "hausdorff":hausdorff}

if __name__ == "__main__":

    pids = os.listdir(DATA_DIR)

    reg_quality = []

    with ProcessPoolExecutor(NUM_WORKERS) as e:
        futures = [e.submit(calculate_metrics, pid) for pid in pids]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating metrics", position=0):
            metrics = future.result()
            reg_quality.append(metrics)

    reg_quality = pd.DataFrame(reg_quality)
    reg_quality.to_csv(os.path.join(OUT_DIR, "reg_quality.csv"), index=False)

#%%
# radiomics feature extraction
def extract_features(pid):

    featureVectors = []

    for mask_type, mask_fname in MASKS.items():

        sitk_img = sitk.ReadImage(os.path.join(DATA_DIR, pid, "img.nii.gz"))
        sitk_mask = sitk.ReadImage(os.path.join(DATA_DIR, pid, mask_fname))

        tissue_mask = sum([sitk.BinaryThreshold(sitk_mask, label, label, insideValue=True, outsideValue=False) for label in ROI[TISSUE]])
        tissue_mask = sitk.Cast(tissue_mask, sitk.sitkUInt8)

        extractor = featureextractor.RadiomicsFeatureExtractor(PARAM_FILE)
        featureVector = extractor.execute(sitk_img, tissue_mask, label=1)
        featureVector = {**{'patient_id':pid, 'mask_type':mask_type}, **featureVector}
        featureVectors.append(featureVector)

    return featureVectors



if __name__=="__main__":

    pids = os.listdir(DATA_DIR)

    radiomics_df = []

    with ProcessPoolExecutor(NUM_WORKERS) as e:
        futures = [e.submit(extract_features, pid) for pid in pids]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features", position=0):
            featureVectors = future.result()
            radiomics_df.extend(featureVectors)

    radiomics_df = pd.DataFrame(radiomics_df)
    radiomics_df.to_csv(os.path.join(OUT_DIR, "radiomicsFeatures3D.csv"), index=False)
    
#%%
# ccc between reg vs. independent mask

