#%%
import SimpleITK as sitk    
import pydicom as pydicom
import json

import pandas as pd
import os
import numpy as np

import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
NUM_WORKERS = os.cpu_count() // 2

from tqdm import tqdm

DATA_DIR = os.path.join("..", "Simulation_T2")
DB_FILE = os.path.join("..", "Acquisition_Pts_Data_GF.xlsx")


#%%
OUT_DIR = "output"


def parse_dcms(root):
    dcm_paths = set()
    for root, _, files in tqdm(os.walk(root), desc="Finding dcm  files"):
        for file in files:
            if pydicom.misc.is_dicom(os.path.join(root, file)):
                dcm_paths.add(root)
                break

    return list(dcm_paths)    

def extract_mr(path):

    mr_record = {}
    
    series_ids = sitk.ImageSeriesReader().GetGDCMSeriesIDs(path)

    if series_ids:
        
        assert len(series_ids)==1, f"multiple MR series found at path: {path}"

        series_id = series_ids[0]
        dcm_series = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(path, series_id)
        dcm_series = sorted(dcm_series, key=lambda x : pydicom.dcmread(x, stop_before_pixels=True).SliceLocation)
        if dcm_series:

            pid = path.split(os.path.sep)[2]
            dcm = pydicom.dcmread(dcm_series[0])
            sitk_img = sitk.ReadImage(dcm_series)
            sequence = "t2w_fs" if "fs" in dcm.SeriesDescription.lower() else "t2w_nfs"
            mr_record = {"pid":pid, "sequence":sequence, "dcm_series":dcm_series, "sitk_img":sitk_img}
            mr_record["metadata"] = {
                "gender": getattr(dcm, "PatientSex", None),
                "age": getattr(dcm, "PatientAge", None),
                "slice_thickness": getattr(dcm, 'SliceThickness', None),
                "spacing_btw_slices": getattr(dcm, 'SpacingBetweenSlices', None),
                "voxel_spacing": list(sitk_img.GetSpacing()),
                "img_matrix": list(sitk_img.GetSize()),
                "echo_train_length": getattr(dcm, 'EchoTrainLength', None),
                "num_of_avgs": getattr(dcm, 'NumberOfAverages', None),
                "flip_angle": getattr(dcm, 'FlipAngle', None),
                "TR": getattr(dcm, 'RepetitionTime', None),
                "TE": getattr(dcm, 'EchoTime', None)
            }
    return mr_record

def export_mr(mr_record, out_dir):

    is_success = False

    try:

        pid, sequence, sitk_img, metadata = mr_record["pid"], mr_record["sequence"], mr_record["sitk_img"], mr_record["metadata"]

        out_dir = os.path.join(out_dir, "data", pid, sequence)
        os.makedirs(out_dir, exist_ok=True)
        
        sitk.WriteImage(sitk_img, os.path.join(out_dir, "img.nii.gz"))
        
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        is_success = True

    except Exception as e:
        print(f"Error exporting MR data for {pid}: {e}")
        pass
    
    return is_success


def parse_and_export(data_dir, out_dir):

    mr_records = {}

    if not os.path.exists(os.path.join(out_dir, "dcm_paths.npy")):
        os.makedirs(out_dir, exist_ok=True)
        dcm_paths = parse_dcms(data_dir)
        np.save(os.path.join(out_dir, "dcm_paths.npy"), dcm_paths)

    else:
        dcm_paths = np.load(os.path.join(out_dir, "dcm_paths.npy"))
       
    dcm_paths = dcm_paths[:20]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as e:
        futures = [e.submit(extract_mr, path) for path in dcm_paths]

        for future in tqdm(as_completed(futures), total=len(dcm_paths), desc="Extracting MR Data", position=0):
            mr_data = future.result()
            if mr_data:
                mr_records[mr_data["pid"]] = {**mr_records.get(mr_data["pid"], {}), mr_data["sequence"]:mr_data["dcm_series"]}
                export_mr(mr_data, out_dir)

    np.save(os.path.join(out_dir,"mr_records.npy"), mr_records)

    return mr_records


if __name__ == "__main__":

    df = pd.read_excel(DB_FILE)
    df.pid = df.pid.str.strip()

    mr_records = parse_and_export(DATA_DIR, OUT_DIR)
    print("total mr data loaded = ", len(mr_records))

    available_pairs = set()
    for pid, seq_dict in mr_records.items():
        available_pairs |= {(pid, seq) for seq in seq_dict}

    all_pairs = list(zip(df["pid"], df["sequence"]))
    mask = [pair in available_pairs for pair in all_pairs]
    sub_df = df[mask].reset_index(drop=True)
    sub_df_grp = sub_df.groupby(by="pid").agg(list).reset_index()
    display(sub_df_grp)
    


