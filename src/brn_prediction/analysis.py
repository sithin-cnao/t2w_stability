#%%
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SequentialFeatureSelector
from boruta import BorutaPy
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
# import neuroCombat as neuroCombat
from neurocombat_sklearn import CombatModel

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import os

import pandas as pd
import numpy as np

from MLstatkit import Delong_test

#%%

FUP_RADIOMICS_FILE = r"/home/sithints/research/projects/t2w_stability/outputs/stability/radiomicsFeatures3D copy.csv"
FUP_STABILITY_FILE = r"/home/sithints/research/projects/t2w_stability/outputs/stability/stability_df copy.csv"

BRN_RADIOMICS_FILE = r"/home/sithints/research/projects/t2w_stability/outputs/brn_prediction/radiomicsFeatures3D copy.csv"
BRN_DB_FILE = r"/home/sithints/research/projects/t2w_stability/outputs/brn_prediction/db copy.xlsx"
OUTDIR = r"/home/sithints/research/projects/t2w_stability/outputs/brn_prediction/analysis_5CV"

os.makedirs(OUTDIR, exist_ok=True)

FEAT_FAMILIES = ["firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"]

BIAS = "none"
ZNORM_ROI = "none"
TISSUE = "gwm"
TYPE = "combat"
CCC_THRESHOLDS = {"baseline":-np.inf, ">=good":0.70, "excellent":0.85}
TARGET_LABEL = "CTCAE_GRADE_NECROSIS>=1"
BATCH_COL = "sequence"

N_SPLITS = 5
RANDOM_STATE = 42

def filter_near_zero(df, threshold = 1e-6, verbose=False): #1e-6 and 1e-3 works
    feats = df.columns.to_list()
    feats_var = df.var()
    mask_feats = feats_var[feats_var<=threshold].index.to_list()
    selected_feats = [feat for feat in feats if feat not in mask_feats]
    if verbose:
        print(f"Deleted {len(mask_feats)}/{len(feats)} near zero features, remaining {len(selected_feats)} features")
    return selected_feats

def filter_high_corr(df, threshold=0.85, verbose=False):

    corr_matrix = df.corr(method='spearman').abs()
    mean_corr = corr_matrix.mean()
    ordered_feats = mean_corr.sort_values(ascending=True).index.to_list()

    corr_matrix = df[ordered_feats].corr(method='spearman').abs()
    up_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    mask_feats = [column for column in up_tri.columns if any(up_tri[column]>=threshold)]
    selected_feats = [feat for feat in ordered_feats if feat not in mask_feats]
    if verbose:
        print(f"Deleted {len(mask_feats)}/{len(ordered_feats)} highly corr features, remaining {len(selected_feats)} features")
    
    return selected_feats

def select_topk_utest(X, y, k=5):

    scores = []
    for i in range(X.shape[1]):
        pvalue = mannwhitneyu(X[y==0, i], X[y==1, i], alternative = 'two-sided').pvalue
        scores.append(pvalue)

    selected_idxs = np.argsort(scores)[:k]

    return selected_idxs

def select_topk_boruta(X, y, k=5):
    rf = RandomForestClassifier(n_jobs=-1, max_depth=5, class_weight="balanced")
    
    boruta = BorutaPy(estimator=rf, n_estimators='auto', verbose=0, random_state=42)
    boruta.fit(X, y)

    importance_history = boruta.importance_history_
    importance_history = np.nan_to_num(importance_history, 0.0)
    importance = np.mean(importance_history, axis=0)

    selected_idxs = np.argsort(importance)[::-1][:k]

    return selected_idxs

def mean_weighted_matching_correlation(df_A, df_B, method = 'pearson'):
    """
    Computes the Mean Weighted Matching Correlation (MWMC) between two signatures (feature sets + predictions).
    It solves the linear assignment problem on the cost matrix C = 1 - |corr(A_i, B_j)|
    and returns the mean absolute correlation of the optimal matching.
    """
    # Calculate the pairwise absolute correlation matrix
    combined = pd.concat([df_A, df_B], axis=1)
    corr = combined.corr(method=method).abs()
    
    # Extract the cross-correlation submatrix (A vs B)
    cols_A = df_A.columns
    cols_B = df_B.columns
    cross_corr = corr.loc[cols_A, cols_B].values
    
    # Cost matrix is 1 - absolute correlation
    cost_matrix = 1.0 - cross_corr
    
    # Find the optimal bipartite matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Return the mean matched correlation (1 - cost)
    return cross_corr[row_ind, col_ind].mean()

def visualize_corr_matrix(corr_matrix, title, figsize=(10,8)):
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=0, 
        vmax=1, 
        square=True,
        cbar_kws={"shrink": .8}
    )
    plt.title(title, fontsize=12, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    plt.savefig(os.path.join(OUTDIR, f"{title}.tiff"), dpi=600, bbox_inches='tight')
    plt.show()

def cross_val(model, fs_method, num_folds = 5, n_repeats=5, random_state = 42):

    def call_fn(X_df, y_df):
        cv = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=n_repeats, random_state=random_state) # LeaveOneOut() 
        y_preds, y_trues = [], []

        X, y = X_df.values, y_df.values
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            train_selected_idxs = fs_method(X_train, y_train, k=k)
            X_train_selected = X_train[:, train_selected_idxs]
            X_test_selected = X_test[:, train_selected_idxs]

            y_pred = clone(model).fit(X_train_selected, y_train).predict_proba(X_test_selected)[:, 1]
            y_preds.append(y_pred)
            y_trues.append(y_test)

        y_true = np.concatenate(y_trues)
        y_pred = np.concatenate(y_preds)

        return y_true, y_pred

    return call_fn

combined_signatures = {}

# %%
# Heterogeneous (Mixed-FS) Subgroups Analysis 
if __name__=="__main__":

    fs_method = select_topk_utest

    fup_radiomics_df = pd.read_csv(FUP_RADIOMICS_FILE, index_col=0)
    fup_radiomics_df = fup_radiomics_df[(fup_radiomics_df.tissue==TISSUE)&(fup_radiomics_df.bias_correction==BIAS)&(fup_radiomics_df.norm_roi==ZNORM_ROI)].copy().reset_index(drop=True)
    features = [feat for feat in fup_radiomics_df.columns for family in FEAT_FAMILIES if family in feat.lower()]

    # Combat parameters are learned from follow-up data used for stability study
    combat = CombatModel()

    variances = fup_radiomics_df[features].copy().var()
    zero_var_feats = variances[variances == 0].index

    nzvar_features = [feat for feat in features if feat not in zero_var_feats]
    combat.fit(data = fup_radiomics_df[nzvar_features].values, sites = fup_radiomics_df[[BATCH_COL]].apply(lambda x: x.astype('category').cat.codes).values)

    radiomics_df = pd.read_csv(BRN_RADIOMICS_FILE)[["pid", "sequence", "exclude"]+features]
    db = pd.read_excel(BRN_DB_FILE)[["ID", "CTCAE GRADE NECROSIS2"]]
    db = db.rename(columns={"ID":"pid"})
    radiomics_df = radiomics_df.merge(db, on="pid").reset_index(drop=True)
    radiomics_df["CTCAE_GRADE_NECROSIS"] = radiomics_df["CTCAE GRADE NECROSIS2"].fillna(0)
    radiomics_df[TARGET_LABEL] = (radiomics_df["CTCAE_GRADE_NECROSIS"]>=1).astype(int)
    radiomics_df = radiomics_df.dropna(subset=features, inplace=False)
    radiomics_df = radiomics_df[radiomics_df.exclude==0].reset_index(drop=True)

    stability_df = pd.read_csv(FUP_STABILITY_FILE)
    stability_df = stability_df[(stability_df.bias_correction==BIAS)&(stability_df.normalization==ZNORM_ROI)&(stability_df.tissue==TISSUE)&(stability_df.type==TYPE)]

    ## Performance Analysis on the heterogenous data
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=np.inf,  random_state = 42)) #no penalty

    k = 4
    
    outputs = {}

    print(f"\nTarget label: {TARGET_LABEL} (prevalance= {len(radiomics_df[radiomics_df[TARGET_LABEL]==1])} / {len(radiomics_df)} = {radiomics_df[TARGET_LABEL].mean():.3f})")
    

    for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():

        print(f"\tCCC threshold: {ccc_stability}")

        combat_data_df = radiomics_df.copy().reset_index(drop=True)
        # transforming mixed-FS baseline features using Combat parameters learned from the stability study
        combat_data_df[nzvar_features] = combat.transform(data = combat_data_df[nzvar_features].values, sites = combat_data_df[[BATCH_COL]].apply(lambda x: x.astype('category').cat.codes).values)
            
        columns = stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()
        stable_features = [feat for feat in columns if feat in features]

        print(f"\t\t# stable features = {len(stable_features)}")
        filtered_features = filter_near_zero(combat_data_df[stable_features])
        filtered_features = filter_high_corr(combat_data_df[filtered_features])
        
        print(f"\t\t# filtered features = {len(filtered_features)}")

        X_df = combat_data_df[filtered_features].copy().reset_index(drop=True)
        y_df = combat_data_df[TARGET_LABEL].copy().reset_index(drop=True)

        selected_idxs = fs_method(X_df.values, y_df.values, k=k)
        selected_features = X_df.iloc[:, selected_idxs].columns.to_list()

        combined_signatures[("fs + non-fs", ccc_stability)] = selected_features

        y_trues, y_preds = cross_val(estimator, fs_method, N_SPLITS, RANDOM_STATE)(X_df, y_df)
        _, _, ci, _, auc, _, info = Delong_test(y_trues, y_preds, y_preds, return_ci=True, return_auc=True, verbose = 0)
        
        utest_p = mannwhitneyu(y_preds[y_trues==0], y_preds[y_trues==1], alternative = 'two-sided').pvalue
        print(f"\t\t\t {estimator[-1].__class__.__name__}: AUC={auc:.3f} [{ci[0]:.3f}-{ci[1]:.3f}], p = {utest_p:.3f}")
        
        outputs[ccc_stability] = {"y_trues":y_trues, "y_preds":y_preds}

    # Delong's test between baseline vs. excellent features; good vs. excellent features

    y_trues = outputs[">=good"]["y_trues"]
    y_preds_baseline = outputs["baseline"]["y_preds"]
    y_preds_good = outputs[">=good"]["y_preds"]
    y_preds_excellent = outputs["excellent"]["y_preds"]

    _, p_value = Delong_test(y_trues, y_preds_baseline, y_preds_excellent, return_ci=False, return_auc=False, verbose = 0)
    print(f"Delong's test between baseline and excellent features: p_value={p_value:.3f}")
    
    _, p_value = Delong_test(y_trues, y_preds_good, y_preds_excellent, return_ci=False, return_auc=False, verbose = 0)
    print(f"Delong's test between >=good and excellent features: p_value={p_value:.3f}")

#%%

# Homogeneous subgroups (FS-only vs. non-FS only) analysis
if __name__=="__main__":

    fs_method = select_topk_utest

    fup_radiomics_df = pd.read_csv(FUP_RADIOMICS_FILE, index_col=0)
    fup_radiomics_df = fup_radiomics_df[(fup_radiomics_df.tissue==TISSUE)&(fup_radiomics_df.bias_correction==BIAS)&(fup_radiomics_df.norm_roi==ZNORM_ROI)].reset_index(drop=True)
    features = [feat for feat in fup_radiomics_df.columns for family in FEAT_FAMILIES if family in feat.lower()]
   
    radiomics_df = pd.read_csv(BRN_RADIOMICS_FILE)[["pid", "sequence", "exclude"]+features]
    db = pd.read_excel(BRN_DB_FILE)[["ID", "CTCAE GRADE NECROSIS2"]]
    db = db.rename(columns={"ID":"pid"})
    radiomics_df = radiomics_df.merge(db, on="pid").reset_index(drop=True)
    radiomics_df["CTCAE_GRADE_NECROSIS"] = radiomics_df["CTCAE GRADE NECROSIS2"].fillna(0)
    radiomics_df[TARGET_LABEL] = (radiomics_df["CTCAE_GRADE_NECROSIS"]>=1).astype(int)
    radiomics_df = radiomics_df.dropna(subset=features, inplace=False)

    grp_radiomics_df = radiomics_df.groupby(by=["pid"]).agg(list).reset_index()
    pids_with_both_sequences = grp_radiomics_df[grp_radiomics_df.sequence.apply(lambda x: len(x)==2)].pid.to_list()
    radiomics_df = radiomics_df[radiomics_df.pid.isin(pids_with_both_sequences)].reset_index(drop=True)

    stability_df = pd.read_csv(FUP_STABILITY_FILE)
    stability_df = stability_df[(stability_df.bias_correction==BIAS)&(stability_df.normalization==ZNORM_ROI)&(stability_df.tissue==TISSUE)&(stability_df.type==TYPE)]

    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=np.inf,  random_state = 42)) #no penalty

    k = 2
    
    outputs = {}
    for sequence in ["t2w_fs", "t2w_nfs"]:

        print(f"\n**Homogeneous Subgroups: Sequence = {sequence}**")

        data_df = radiomics_df[(radiomics_df["sequence"]==sequence)].copy().reset_index(drop=True)

        print(f"\nTarget label: {TARGET_LABEL} (prevalance= {len(data_df[data_df[TARGET_LABEL]==1])} / {len(data_df)} = {data_df[TARGET_LABEL].mean():.3f})")

        for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():

            print(f"\tCCC threshold: {ccc_stability}")

            columns = stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()
            stable_features = [feat for feat in columns if feat in features]

            print(f"\t\t# stable features = {len(stable_features)}")
            filtered_features = filter_near_zero(data_df[stable_features])
            filtered_features = filter_high_corr(data_df[filtered_features])
            
            print(f"\t\t# filtered features = {len(filtered_features)}")

            X_df = data_df[filtered_features].copy().reset_index(drop=True)
            y_df = data_df[TARGET_LABEL].copy().reset_index(drop=True)

            selected_idxs = fs_method(X_df.values, y_df.values, k=k)
            selected_features = X_df.iloc[:, selected_idxs].columns.to_list()

            combined_signatures[(sequence, ccc_stability)] = selected_features

            y_trues, y_preds = cross_val(estimator, fs_method, N_SPLITS, RANDOM_STATE)(X_df, y_df)
            _, _, ci, _, auc, _, info = Delong_test(y_trues, y_preds, y_preds, return_ci=True, return_auc=True, verbose = 0)
            
            utest_p = mannwhitneyu(y_preds[y_trues==0], y_preds[y_trues==1], alternative = 'two-sided').pvalue
            print(f"\t\t\t {estimator[-1].__class__.__name__}: AUC={auc:.3f} [{ci[0]:.3f}-{ci[1]:.3f}], p = {utest_p:.3f}")

            outputs[(sequence, ccc_stability)] = {"y_trues":y_trues, "y_preds":y_preds}

        y_trues = outputs[(sequence, ">=good")]["y_trues"]
        y_preds_baseline = outputs[(sequence, "baseline")]["y_preds"]
        y_preds_good = outputs[(sequence, ">=good")]["y_preds"]
        y_preds_excellent = outputs[(sequence, "excellent")]["y_preds"]

        _, p_value = Delong_test(y_trues, y_preds_baseline, y_preds_excellent, return_ci=False, return_auc=False, verbose = 0)
        print(f"\t\tDelong's test between baseline and excellent features: p_value={p_value:.3f}")

        _, p_value = Delong_test(y_trues, y_preds_good, y_preds_excellent, return_ci=False, return_auc=False, verbose = 0)
        print(f"\t\tDelong's test between >=good and excellent features: p_value={p_value:.3f}")
    
    y_trues = outputs[("t2w_fs", "excellent")]["y_trues"]
    y_preds_fs = outputs[("t2w_fs", "excellent")]["y_preds"]
    y_preds_nfs = outputs[("t2w_nfs", "excellent")]["y_preds"]

    _, p_value = Delong_test(y_trues, y_preds_fs, y_preds_nfs, return_ci=False, return_auc=False, verbose = 0)

    print(f"Delong's test between excellent fs and nfs features: p_value={p_value:.3f}")

    config_names = list(combined_signatures.keys())
    n_configs = len(config_names)
    pearson_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)
    spearman_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)

    for i in range(n_configs):
        for j in range(n_configs):
            name_A = config_names[i]
            name_B = config_names[j]
            
            pearson_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(radiomics_df[combined_signatures[name_A]], radiomics_df[combined_signatures[name_B]], method='pearson')
            spearman_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(radiomics_df[combined_signatures[name_A]], radiomics_df[combined_signatures[name_B]], method='spearman')


    for method, matrix in {"pearson":pearson_mwmc_matrix, "spearman":spearman_mwmc_matrix}.items():
        visualize_corr_matrix(matrix, method.capitalize() + " Correlation Matrix")



# %%
print(combined_signatures)

'''
{('t2w_fs', 'baseline'): ['wavelet-HLH_glcm_InverseVariance',
  'lbp-3D-k_firstorder_10Percentile'],
 ('t2w_fs', '>=good'): ['lbp-3D-k_firstorder_10Percentile',
  'wavelet-LHL_ngtdm_Strength'],
 ('t2w_fs', 'excellent'): ['lbp-3D-k_firstorder_10Percentile',
  'wavelet-LHL_ngtdm_Strength'],
 ('t2w_nfs', 'baseline'): ['wavelet-LLL_glszm_LargeAreaHighGrayLevelEmphasis',
  'lbp-3D-k_firstorder_10Percentile'],
 ('t2w_nfs', '>=good'): ['original_glszm_LargeAreaHighGrayLevelEmphasis',
  'lbp-3D-k_firstorder_10Percentile'],
 ('t2w_nfs', 'excellent'): ['wavelet-LLH_gldm_GrayLevelNonUniformity',
  'original_glszm_ZoneVariance']}

'''