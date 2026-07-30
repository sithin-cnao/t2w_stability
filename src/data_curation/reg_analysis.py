#%%
import os
import SimpleITK as sitk
from tqdm.notebook import tqdm

import subprocess
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
NUM_WORKERS = 16


DATA_DIR = r"/home/sithints/research/datasets/curated/BrainMR"
OUT_DIR = r"/home/sithints/research/projects/t2w_stability/outputs/data_curation"

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

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
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

def parse_nifty(data_dir):
    nifty_dirs = []
    for root,_,files in os.walk(data_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                nifty_dirs.append(root)
                break;

    return nifty_dirs
        

if __name__ == "__main__":

    src_dirs = parse_nifty(DATA_DIR)
    print("total mr data found = ", len(src_dirs))

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as e:
        futures = [e.submit(process, src_dir) for src_dir in src_dirs]

        for future in tqdm(as_completed(futures), total=len(src_dirs), desc="Generating tissue segmentation masks", position=0):
            res = future.result()