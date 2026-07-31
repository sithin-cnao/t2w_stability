#%%

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from utils import compute_ccc, combat_correction, bca_bootstrap_ci

BIAS_CORRECTION = ["none", "default", "brain", "gwm"]
ZNORM_ROI = ["none", "brain", "gwm", "gm", "wm"]
FEAT_FAMILIES = ["firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"]
TISSUE = "gwm"

NUM_BOOTSTRAPS = 2000
SEED = 42

XL_RADIOMICS = r"/home/sithints/research/projects/t2w_stability/outputs/stability/radiomicsFeatures3D.csv"
OUT_DIR = r"/home/sithints/research/projects/t2w_stability/outputs/stability/config_analysis"

os.makedirs(OUT_DIR, exist_ok=True)

radiomics_df = pd.read_csv(XL_RADIOMICS, index_col=0)
features = [feat for feat in radiomics_df.columns for feat_family in FEAT_FAMILIES if feat_family in feat.lower()]
pids = radiomics_df["id"].unique()

#%%

def estimator(fs_df, nfs_df):
    ccc_df = compute_ccc(fs_df, nfs_df)
    return ccc_df.ccc.median()

#%%
if __name__=="__main__":

    observed_median_ccc = {}
    bootstrap_median_ccc = {}
    observed_ccc_dfs = {}
    bootstrap_ccc_dfs = {}
    for bc in BIAS_CORRECTION:
        for roi in ZNORM_ROI:
            print(f"Configuration: Bias Correction = {bc}, Z-Norm ROI = {roi}")

            sub_df = radiomics_df[(radiomics_df.bias_correction == bc) & (radiomics_df.norm_roi == roi) & (radiomics_df.tissue == TISSUE)].reset_index(drop=True)
            combat_sub_df = sub_df.copy()
            combat_sub_df[features] = combat_correction(sub_df[features], sub_df[["sequence"]], batch_col="sequence")
            
            fs_df = sub_df[sub_df.sequence=="t2w_fs"].sort_values(by="id").reset_index(drop=True)
            nfs_df = sub_df[sub_df.sequence=="t2w_nfs"].sort_values(by="id").reset_index(drop=True)
            
            combat_fs_df = combat_sub_df[sub_df.sequence=="t2w_fs"].sort_values(by="id").reset_index(drop=True)
            combat_nfs_df = combat_sub_df[sub_df.sequence=="t2w_nfs"].sort_values(by="id").reset_index(drop=True)
            
            ccc_df = compute_ccc(fs_df[features], nfs_df[features])
            combat_ccc_df = compute_ccc(combat_fs_df[features], combat_nfs_df[features])

            observed_ccc_dfs[(bc, roi, "normal")] = ccc_df
            observed_ccc_dfs[(bc, roi, "combat")] = combat_ccc_df

            observed_median_ccc[(bc, roi, "normal")] = (ccc_df.ccc).median()
            observed_median_ccc[(bc, roi, "combat")] = (combat_ccc_df.ccc).median()

            np.random.seed(SEED)

            for i in tqdm(range(NUM_BOOTSTRAPS), position=0, desc=f"Bootstrapping configuration: Bias correction = {bc}, Z-Norm ROI = {roi}"):
                
                bootstrap_pids = np.random.choice(pids, size=len(pids), replace=True)
                bootstrap_idx = np.array([np.argwhere(sub_df.id.values==pid) for pid in bootstrap_pids]).ravel()

                bootstrap_sub_df = sub_df.iloc[bootstrap_idx].reset_index(drop=True)
                bootstrap_combat_sub_df = bootstrap_sub_df.copy()
                bootstrap_combat_sub_df[features] = combat_correction(bootstrap_sub_df[features], bootstrap_sub_df[["sequence"]], batch_col="sequence")

                bootstrap_fs_df = bootstrap_sub_df[bootstrap_sub_df.sequence=="t2w_fs"].sort_values(by="id").reset_index(drop=True)
                bootstrap_nfs_df = bootstrap_sub_df[bootstrap_sub_df.sequence=="t2w_nfs"].sort_values(by="id").reset_index(drop=True)

                bootstrap_combat_fs_df = bootstrap_combat_sub_df[bootstrap_combat_sub_df.sequence=="t2w_fs"].sort_values(by="id").reset_index(drop=True)
                bootstrap_combat_nfs_df = bootstrap_combat_sub_df[bootstrap_combat_sub_df.sequence=="t2w_nfs"].sort_values(by="id").reset_index(drop=True)

                bootstrap_ccc_df = compute_ccc(bootstrap_fs_df[features], bootstrap_nfs_df[features])
                bootstrap_combat_ccc_df = compute_ccc(bootstrap_combat_fs_df[features], bootstrap_combat_nfs_df[features])

                bootstrap_ccc_dfs.setdefault((bc, roi, "normal"), [])
                bootstrap_ccc_dfs[(bc, roi, "normal")].append(bootstrap_ccc_df)

                bootstrap_ccc_dfs.setdefault((bc, roi, "combat"), [])
                bootstrap_ccc_dfs[(bc, roi, "combat")].append(bootstrap_combat_ccc_df)

                bootstrap_median_ccc.setdefault((bc, roi, "normal"), [])
                bootstrap_median_ccc[(bc, roi, "normal")].append((bootstrap_ccc_df.ccc).median())

                bootstrap_median_ccc.setdefault((bc, roi, "combat"), [])
                bootstrap_median_ccc[(bc, roi, "combat")].append((bootstrap_combat_ccc_df.ccc).median())
            
    np.savez_compressed(os.path.join(OUT_DIR, "bootstrap_config_results.npz"), bootstrap_median_ccc = bootstrap_median_ccc, observed_median_ccc = observed_median_ccc, observed_ccc_dfs = observed_ccc_dfs, bootstrap_ccc_dfs = bootstrap_ccc_dfs)

#%%
# analyze results
bootstrap_config_results = np.load(os.path.join(OUT_DIR, "bootstrap_config_results.npz"), allow_pickle=True)
bootstrap_median_ccc = bootstrap_config_results['bootstrap_median_ccc'].item()
observed_median_ccc = bootstrap_config_results['observed_median_ccc'].item()

bootstrap_ccc_dfs = bootstrap_config_results['bootstrap_ccc_dfs'].item()
observed_ccc_dfs = bootstrap_config_results['observed_ccc_dfs'].item()

configs = list(itertools.product(BIAS_CORRECTION, ZNORM_ROI, ["normal", "combat"]))
bootstrap_matrix = np.array([bootstrap_median_ccc[config] for config in configs])
true_matrix = np.array([observed_median_ccc[config] for config in configs])

best_config = None
max_value = -np.inf

config_stats = {"config": [], "observed_median_ccc": [], "bootstrap_median_ccc": [], "bias": [], "95% CI":[], "95% BCa CI": []}
for bc in BIAS_CORRECTION:
    for roi in ZNORM_ROI:

        sub_df = radiomics_df[(radiomics_df.bias_correction == bc) & (radiomics_df.norm_roi == roi) & (radiomics_df.tissue == TISSUE)].reset_index(drop=True)
        combat_sub_df = sub_df.copy()
        combat_sub_df[features] = combat_correction(sub_df[features], sub_df[["sequence"]], batch_col="sequence")
        
        fs_df = sub_df[sub_df.sequence=="t2w_fs"].sort_values(by="id").reset_index(drop=True)
        nfs_df = sub_df[sub_df.sequence=="t2w_nfs"].sort_values(by="id").reset_index(drop=True)
        
        combat_fs_df = combat_sub_df[sub_df.sequence=="t2w_fs"].sort_values(by="id").reset_index(drop=True)
        combat_nfs_df = combat_sub_df[sub_df.sequence=="t2w_nfs"].sort_values(by="id").reset_index(drop=True)

        for data_type, dfs in {"normal":(fs_df, nfs_df), "combat":(combat_fs_df, combat_nfs_df)}.items():
            df1, df2 = dfs
            print("config=", (bc, roi, data_type))
            estimate = observed_median_ccc[(bc, roi, data_type)]
            bootstrap_estimates = bootstrap_median_ccc[(bc, roi, data_type)]
            
            bootstrap_mean = np.mean(bootstrap_estimates)
            bootstrap_ci = np.quantile(bootstrap_estimates, [0.025, 0.975])
            bias = bootstrap_mean - estimate
            
            corr_ci = bca_bootstrap_ci(estimator, df1[features], df2[features], estimate, bootstrap_estimates)
        
            
            print(f"Sample Estimate: {estimate:.3f}")
            print(f"Bootstrap Mean: {bootstrap_mean:.3f}")
            print(f"95% CI: {bootstrap_ci}")
            print(f"Bias: {bias:.3f}")
            print(f"95% BCa CI: {corr_ci}")

            config_stats["config"].append((bc, roi, data_type))
            config_stats["observed_median_ccc"].append(estimate)
            config_stats["bootstrap_median_ccc"].append(bootstrap_mean)
            config_stats["bias"].append(bias)
            config_stats["95% CI"].append(bootstrap_ci)
            config_stats["95% BCa CI"].append(corr_ci)

            if estimate>max_value:
                max_value = estimate
                best_config = (bc, roi, data_type)


config_stats_df = pd.DataFrame(config_stats)
print(config_stats_df)
config_stats_df.to_csv(os.path.join(OUT_DIR, "config_stats.csv"), index=False)


#%%
config_pvalues = {"config":[], "p-value":[]}
for config in configs:

    if config == best_config:
        continue

    best_config_df = observed_ccc_dfs[best_config].sort_values(by="feature").reset_index(drop=True)
    config_df = observed_ccc_dfs[config].sort_values(by="feature").reset_index(drop=True)
    
    statistic, p_value = mannwhitneyu(config_df.ccc, best_config_df.ccc)
    print(f"For config: {config}, The p-value from mann-whitney u test with best_config: {best_config} is {p_value}")

    config_pvalues['config'].append(config)
    config_pvalues['p-value'].append(p_value)

config_pvalues_df = pd.DataFrame(config_pvalues)

_, bonf_pvals_corrected, _, _ = multipletests(
    config_pvalues_df['p-value'], 
    alpha=0.05, 
    method='bonferroni'
)
config_pvalues_df['bonf_corrected_p-value'] = bonf_pvals_corrected

_, bh_pvals_corrected, _, _ = multipletests(
    config_pvalues_df['p-value'], 
    alpha=0.05, 
    method='fdr_bh'
)
config_pvalues_df['bh_corrected_p-value'] = bh_pvals_corrected

config_pvalues_df.to_csv(os.path.join(OUT_DIR, "config_pvalues.csv"), index=False)
print(config_pvalues_df)

                