#%%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SequentialFeatureSelector
from boruta import BorutaPy
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from neuroCombat import neuroCombat
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import os

import pandas as pd
import numpy as np

STABILITY_FILE = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/stability_df.csv"
RADIOMICS_FILE = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/radiomicsFeatures3D.csv"
OUTDIR = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/figures"

BIAS = "none"
NORMALIZATION = "none"
TISSUE = "gwm"
TYPE = "combat"
CCC_THRESHOLDS = {"baseline":-np.inf, ">=good":0.70, "excellent":0.85}
METADATA_COLS = ["pid", "sequence", "exclude", "brn_grade", "brn_grade>=g1", "slice_thickness", "spacing_btw_slices", ]
TARGET_LABEL = "brn_grade>=g1"

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

def combat_correction(X_df, covars, batch_col):
    features = X_df.columns.to_list()

    data = X_df.copy()
    covars = pd.DataFrame({col:covars[col].astype("category").cat.codes for col in covars.columns})

    combat_X = neuroCombat(dat=data.T, covars = covars, batch_col=batch_col)["data"].T
    combat_X_df = pd.DataFrame(combat_X, columns = features)
    
    return combat_X_df


def select_topk_boruta(X, y, k=5):
    rf = RandomForestClassifier(n_jobs=-1)
    
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

global_signatures = {}


#%%

stability_df = pd.read_csv(STABILITY_FILE)

bias_correction = "none"
normalization = "none"
tissue = "gwm"


best_df = stability_df[(stability_df.type=="combat")&(stability_df.bias_correction==bias_correction)&(stability_df.normalization==normalization)&(stability_df.tissue==tissue)]
features = [feat for feat in best_df.feature.to_list() if "shape" not in feat]
best_df = best_df[best_df.feature.isin(features)]
best_df = best_df[best_df.ccc>=0.70]
best_df = best_df[["feature", "ccc"]].reset_index(drop=True)
best_df["ccc_level"] = pd.NA
best_df.loc[best_df.ccc>=0.85, "ccc_level"] = "excellent"
best_df.loc[best_df.ccc<0.85, "ccc_level"] = "good"
best_df.head()
best_df.to_excel(os.path.join(OUTDIR, "best_features.xlsx"), index=False)

#%%
improvements = []
for bias_correction in ["none", "default", "brain", "gwm"]:

    for normalization in ["none", "brain", "gwm", "gm", "wm"]:

        print(bias_correction, normalization)

        sub_df = stability_df[(stability_df.type=="normal")&(stability_df.bias_correction==bias_correction)&(stability_df.normalization==normalization)&(stability_df.tissue==tissue)]
        combat_sub_df = stability_df[(stability_df.type=="combat")&(stability_df.bias_correction==bias_correction)&(stability_df.normalization==normalization)&(stability_df.tissue==tissue)]

        features = [feat for feat in sub_df.feature.to_list() if "shape" not in feat]

        sub_df = sub_df[sub_df.feature.isin(features)]
        combat_sub_df = combat_sub_df[combat_sub_df.feature.isin(features)]

        # print(len(sub_df[sub_df.ccc>=0.70])/len(combat_sub_df[combat_sub_df.ccc>=0.70]))
        # print(len(sub_df[sub_df.ccc>=0.70])/len(sub_df))
        # print(len(combat_sub_df[combat_sub_df.ccc>=0.70])/len(combat_sub_df))
        improvement = len(sub_df[sub_df.ccc>=0.85])/len(combat_sub_df[combat_sub_df.ccc>=0.85])
        improvements.append(improvement)

        print("improvement = ", improvement)

#%%
improvements = []
degradations = []
for bias_correction in ["default", "brain", "gwm"]:

    for normalization in ["brain", "gwm", "gm", "wm"]:

        print(bias_correction, normalization)

        sub_df = stability_df[(stability_df.type=="normal")&(stability_df.bias_correction=="none")&(stability_df.normalization=="none")&(stability_df.tissue==tissue)]
        post_sub_df = stability_df[(stability_df.type=="normal")&(stability_df.bias_correction==bias_correction)&(stability_df.normalization==normalization)&(stability_df.tissue==tissue)]

        features = [feat for feat in sub_df.feature.to_list() if "shape" not in feat]

        sub_df = sub_df[sub_df.feature.isin(features)]
        post_sub_df = post_sub_df[post_sub_df.feature.isin(features)]

        # print(len(sub_df[sub_df.ccc>=0.70])/len(combat_sub_df[combat_sub_df.ccc>=0.70]))
        # print(len(sub_df[sub_df.ccc>=0.70])/len(sub_df))
        # print(len(combat_sub_df[combat_sub_df.ccc>=0.70])/len(combat_sub_df))
        improvement = len(sub_df[sub_df.ccc>=0.85])/len(post_sub_df[post_sub_df.ccc>=0.85])
        degradation = len(post_sub_df[post_sub_df.ccc>=0.85])/len(sub_df[sub_df.ccc>=0.85])

        if improvement > 1.0:
            degradations.append(degradation)
        else:
            improvements.append(improvement)
        

        print(f"degradation = {degradation:.3f}" if improvement>1.0 else f"improvement = {improvement:.3f}")
        

    
#%%
if __name__=="__main__":

    stability_df = pd.read_csv(STABILITY_FILE)
    stability_df = stability_df[(stability_df.bias_correction==BIAS)&(stability_df.normalization==NORMALIZATION)&(stability_df.tissue==TISSUE)&(stability_df.type==TYPE)]
    radiomics_features = [feat for feat in stability_df.feature.to_list() if "shape" not in feat]

    radiomics_df = pd.read_csv(RADIOMICS_FILE)
    radiomics_df = radiomics_df.dropna(subset=radiomics_features, inplace=False)
    radiomics_df = radiomics_df[radiomics_df.exclude==0].reset_index(drop=True)

    ## Performance Analysis on the heterogenous data
    piped_estimators = [make_pipeline(StandardScaler(), LogisticRegression(random_state=42)), make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1))]

    k = 4
    
    signatures = {}

    for target_label in ["brn_grade>=g1"]:

        print(f"\nTarget label: {target_label} (prevalance= {len(radiomics_df[radiomics_df[target_label]==1])} / {len(radiomics_df)} = {radiomics_df[target_label].mean():.3f})")
        
        for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():
            print(f"\tCCC threshold: {ccc_stability}")

            for harmonziation_type, data_df in {"combat": radiomics_df.copy().reset_index(drop=True)}.items(): # "normal": radiomics_df.copy().reset_index(drop=True), 
                
                print(f"\t\tHarmonization: {harmonziation_type}")

                stable_features = stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()
                stable_features = [feat for feat in stable_features if "shape" not in feat]

                print(f"\t\t# stable features = {len(stable_features)}")
                filtered_features = filter_near_zero(data_df[stable_features])
                filtered_features = filter_high_corr(data_df[filtered_features])
                
                # stable_features = [feat for feat in filtered_features if feat in stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()]
                
                print(f"\t\t# filtered features = {len(filtered_features)}")

                if harmonziation_type=="combat":
                    data_df[filtered_features] = combat_correction(data_df[filtered_features], covars=data_df[["sequence"]], batch_col="sequence")

                X = data_df[filtered_features].values
                y = data_df[target_label].values

                selected_idxs = select_topk_boruta(X, y, k=k)
                X_selected = X[:, selected_idxs]

                signatures[("fs + non-fs", ccc_stability)] = pd.DataFrame(X_selected, columns=np.array(filtered_features)[selected_idxs])
                global_signatures[("fs + non-fs", "heterogeneous", ccc_stability)] = pd.DataFrame(X_selected, columns=np.array(filtered_features)[selected_idxs])

        
                for estimator in piped_estimators:
                    # cv =  StratifiedKFold(n_splits=10, random_state=42, shuffle=True) 
                    # aucs = cross_val_score(clone(estimator), X_selected, y, cv=cv, scoring="roc_auc")
                    # auc = aucs.mean()
                    # std = aucs.std()
                    # print("\t\t\t", estimator[-1].__class__.__name__, f"AUC={auc:.3f} +/- {std:.3f}")
                    
                    cv = LeaveOneOut() #
                    y_pred = cross_val_predict(clone(estimator), X_selected, y, cv=cv, method="predict_proba")[:, 1]
                    auc = roc_auc_score(y, y_pred)
                    print(f"\t\t\t {estimator[-1].__class__.__name__} AUC={auc:.3f}")

                    # cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True) 
                    # y_preds, y_trues = [], []
                    # for train_idx, test_idx in cv.split(X, y):
                    #     X_train = X[train_idx]
                    #     y_train = y[train_idx]

                    #     X_test = X[test_idx]
                    #     y_test = y[test_idx]

                    #     train_selected_idxs = select_topk_boruta(X_train, y_train, k=k)

                    #     X_train_selected = X_train[:, train_selected_idxs]
                    #     X_test_selected = X_test[:, train_selected_idxs]

                    #     y_pred = clone(estimator).fit(X_train_selected, y_train).predict_proba(X_test_selected)[:, 1]
                        
                    #     y_preds.append(y_pred)
                    #     y_trues.append(y_test)

                    # y_preds = np.concatenate(y_preds)
                    # y_trues = np.concatenate(y_trues)

                    # auc = roc_auc_score(y_trues, y_preds)
                    

    config_names = list(signatures.keys())
    n_configs = len(config_names)
    pearson_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)
    spearman_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)

    for i in range(n_configs):
        for j in range(n_configs):
            name_A = config_names[i]
            name_B = config_names[j]
            
            pearson_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(signatures[name_A], signatures[name_B], method='pearson')
            spearman_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(signatures[name_A], signatures[name_B], method='spearman')


    for method, matrix in {"pearson":pearson_mwmc_matrix, "spearman":spearman_mwmc_matrix}.items():
        visualize_corr_matrix(matrix, method.capitalize() + " Correlation Matrix (Heterogeneous)")

#%%

# if __name__=="__main__":

#     stability_df = pd.read_csv(STABILITY_FILE)
#     stability_df = stability_df[(stability_df.bias_correction==BIAS)&(stability_df.normalization==NORMALIZATION)&(stability_df.tissue==TISSUE)&(stability_df.type==TYPE)]
#     radiomics_features = [feat for feat in stability_df.feature.to_list() if "shape" not in feat]

#     radiomics_df = pd.read_csv(RADIOMICS_FILE).dropna()
#     radiomics_df = radiomics_df.dropna(subset=radiomics_features, inplace=False)
#     radiomics_df = radiomics_df[radiomics_df.exclude==0].reset_index(drop=True)

    

#     ## Performance Analysis on the heterogenous data
#     piped_estimators = [make_pipeline(StandardScaler(), LogisticRegression(random_state=42)), make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1))]

#     k = 2

    
#     signatures = {}

#     for sequence in ["t2w_fs", "t2w_nfs"]:

#         print(f"\n**Homogeneous Subgroups: Sequence = {sequence}**")

#         for target_label in ["brn_grade>=g1"]:

#             print(f"\nTarget label: {target_label} (prevalance= {len(radiomics_df[(radiomics_df['sequence']==sequence) & (radiomics_df[target_label]==1)])} / {len(radiomics_df[(radiomics_df['sequence']==sequence)])} = {radiomics_df[(radiomics_df['sequence']==sequence)][target_label].mean():.3f})")
            
#             for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():
#                 print(f"\tCCC threshold: {ccc_stability}")

#                 for harmonziation_type, data_df in {"normal": radiomics_df[(radiomics_df['sequence']==sequence)].copy().reset_index(drop=True)}.items():
                    
#                     print(f"\t\tHarmonization: {harmonziation_type}")

#                     filtered_features = filter_near_zero(data_df[radiomics_features])
#                     filtered_features = filter_high_corr(data_df[filtered_features])
                
                
#                     stable_features = [feat for feat in filtered_features if feat in stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()]
                    
#                     print(f"# features = {len(stable_features)}")
#                     # if harmonziation_type=="combat":
#                     #     data_df[stable_features] = combat_correction(data_df[stable_features], covars=data_df[["sequence"]], batch_col="sequence")

#                     X = data_df[stable_features].values
#                     y = data_df[target_label].values

#                     selected_idxs = select_topk_boruta(X, y, k=k)
#                     X_selected = X[:, selected_idxs]

#                     signatures[(target_label, sequence, ccc_stability, harmonziation_type)] = pd.DataFrame(X_selected, columns=np.array(stable_features)[selected_idxs])
#                     global_signatures[(sequence, "heterogeneous_patients", target_label, ccc_stability, harmonziation_type)] = pd.DataFrame(X_selected, columns=np.array(stable_features)[selected_idxs])

        
#                     for estimator in piped_estimators:
#                         cv = LeaveOneOut() # StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

#                         # selected_idxs = select_topk_sfs(X, y, k=k, estimator=clone(estimator), direction="backward")
#                         # X_selected = X[:, selected_idxs]

                    
#                         # signatures[(target_label, ccc_stability, harmonziation_type, estimator[-1].__class__.__name__)] = data_df[np.array(stable_features)[selected_idxs]]
                        
#                         y_pred = cross_val_predict(clone(estimator), X_selected, y, cv=cv, method="predict_proba")[:, 1]

#                         auc = roc_auc_score(y, y_pred)
#                         print("\t\t\t", estimator[-1].__class__.__name__, f"AUC={auc:.3f}")

            
#     config_names = list(signatures.keys())
#     n_configs = len(config_names)
#     pearson_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)
#     spearman_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)

#     for i in range(n_configs):
#         for j in range(n_configs):
#             name_A = config_names[i]
#             name_B = config_names[j]
            
#             pearson_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(signatures[name_A], signatures[name_B], method='pearson')
#             spearman_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(signatures[name_A], signatures[name_B], method='spearman')


#     for method, matrix in {"pearson":pearson_mwmc_matrix, "spearman":spearman_mwmc_matrix}.items():
#         visualize_corr_matrix(matrix, method.capitalize() + " Correlation Matrix")


#%%

if __name__=="__main__":

    stability_df = pd.read_csv(STABILITY_FILE)
    stability_df = stability_df[(stability_df.bias_correction==BIAS)&(stability_df.normalization==NORMALIZATION)&(stability_df.tissue==TISSUE)&(stability_df.type==TYPE)]
    radiomics_features = [feat for feat in stability_df.feature.to_list() if "shape" not in feat]

    radiomics_df = pd.read_csv(RADIOMICS_FILE)
    radiomics_df = radiomics_df.dropna(subset=radiomics_features, inplace=False)

    grp_radiomics_df = radiomics_df.groupby(by=["pid"]).agg(list).reset_index()
    pids_with_both_sequences = grp_radiomics_df[grp_radiomics_df.sequence.apply(lambda x: len(x)==2)].pid.to_list()
    radiomics_df = radiomics_df[radiomics_df.pid.isin(pids_with_both_sequences)].reset_index(drop=True)

    ## Performance Analysis on the heterogenous data
    piped_estimators = [make_pipeline(StandardScaler(), LogisticRegression(random_state=42)), make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1))]

    k = 2

    signatures = {}

    for sequence in ["t2w_fs", "t2w_nfs"]:

        print(f"\n**Homogeneous Subgroups: Sequence = {sequence}**")

        for target_label in ["brn_grade>=g1"]:

            print(f"\nTarget label: {target_label} (prevalance= {len(radiomics_df[(radiomics_df['sequence']==sequence) & (radiomics_df[target_label]==1)])} / {len(radiomics_df[(radiomics_df['sequence']==sequence)])} = {radiomics_df[(radiomics_df['sequence']==sequence)][target_label].mean():.3f})")
            for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():
                print(f"\tCCC threshold: {ccc_stability}")

                for harmonziation_type, data_df in {"normal": radiomics_df[(radiomics_df["sequence"]==sequence)].copy().reset_index(drop=True)}.items():
                    
                    print(f"\t\tHarmonization: {harmonziation_type}")

                    stable_features = stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()
                    stable_features = [feat for feat in stable_features if "shape" not in feat]

                    print(f"\t\t# stable features = {len(stable_features)}")

                    filtered_features = filter_near_zero(data_df[stable_features])
                    filtered_features = filter_high_corr(data_df[filtered_features])
                
                    # stable_features = [feat for feat in filtered_features if feat in stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()]
                    
                    print(f"\t\t# filtered features = {len(filtered_features)}")
                    # if harmonziation_type=="combat":
                    #     data_df[stable_features] = combat_correction(data_df[stable_features], covars=data_df[["sequence"]], batch_col="sequence")

                    X = data_df[filtered_features].values
                    y = data_df[target_label].values

                    selected_idxs = select_topk_boruta(X, y, k=k)
                    X_selected = X[:, selected_idxs]

                    signatures[("fs" if sequence=="t2w_fs" else "non-fs", ccc_stability)] = pd.DataFrame(X_selected, columns=np.array(filtered_features)[selected_idxs])
                    global_signatures[("fs" if sequence=="t2w_fs" else "non-fs", "homogeneous", ccc_stability)] = pd.DataFrame(X_selected, columns=np.array(filtered_features)[selected_idxs])

        
                    for estimator in piped_estimators:
                        cv =   LeaveOneOut() #cv = LeaveOneOut() # StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
                         

                        y_pred = cross_val_predict(clone(estimator), X_selected, y, cv=cv, method="predict_proba")[:, 1]
                        auc = roc_auc_score(y, y_pred)
                        
                        # cv =  StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
                        # aucs = cross_val_score(clone(estimator), X_selected, y, cv=cv, scoring="roc_auc")
                        # auc = aucs.mean()
                        # std = aucs.std()
                        print("\t\t\t", estimator[-1].__class__.__name__, f"AUC={auc:.3f}")

    config_names = list(signatures.keys())
    n_configs = len(config_names)
    pearson_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)
    spearman_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)

    for i in range(n_configs):
        for j in range(n_configs):
            name_A = config_names[i]
            name_B = config_names[j]
            
            pearson_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(signatures[name_A], signatures[name_B], method='pearson')
            spearman_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(signatures[name_A], signatures[name_B], method='spearman')


    for method, matrix in {"pearson":pearson_mwmc_matrix, "spearman":spearman_mwmc_matrix}.items():
        visualize_corr_matrix(matrix, method.capitalize() + " Correlation Matrix (Homogeneous)")


#%%

config_names = list(global_signatures.keys())

print(config_names)

config_names = [
    ("fs + non-fs", "heterogeneous", "baseline"),
    ("fs + non-fs", "heterogeneous", ">=good"),
    ("fs + non-fs", "heterogeneous", "excellent"),
    ("fs", "homogeneous", "baseline"),
    ("fs", "homogeneous", ">=good"),
    ("fs", "homogeneous", "excellent"),
    ("non-fs", "homogeneous", "baseline"),
    ("non-fs", "homogeneous", ">=good"),
    ("non-fs", "homogeneous", "excellent")
]

n_configs = len(config_names)
pearson_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_onfigs)), index=config_names, columns=config_names)
spearman_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)

for i in range(n_configs):
    for j in range(n_configs):
        name_A = config_names[i]
        name_B = config_names[j]
        
        pearson_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(global_signatures[name_A], global_signatures[name_B], method='pearson')
        spearman_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(global_signatures[name_A], global_signatures[name_B], method='spearman')


for method, matrix in {"pearson":pearson_mwmc_matrix, "spearman":spearman_mwmc_matrix}.items():
    visualize_corr_matrix(matrix, method.capitalize() + " Correlation Matrix")




#%%

stability_df = pd.read_csv(STABILITY_FILE)
stability_df = stability_df[(stability_df.bias_correction==BIAS)&(stability_df.normalization==NORMALIZATION)&(stability_df.tissue==TISSUE)&(stability_df.type==TYPE)]

radiomics_df = pd.read_csv(RADIOMICS_FILE)

radiomics_df = radiomics_df[radiomics_df.exclude==0].reset_index(drop=True)

radiomics_features = [feat for feat in stability_df.feature.to_list() if "shape" not in feat]
## Performance Analysis on the heterogenous data
piped_estimators = [make_pipeline(StandardScaler(), LogisticRegression(random_state=42)), make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1))]

k = 3

signatures = {}
pca_results = {}

for sequence in ["t2w_fs", "t2w_nfs"]:

    print(f"\n**Homogeneous Subgroups: Sequence = {sequence}**")

    for target_label in ["brn_grade>=g1"]:
        print(f"\nTarget label: {target_label} (prevalance = {len(radiomics_df[(radiomics_df.sequence==sequence)&(radiomics_df[target_label]==1)])} / {len(radiomics_df[radiomics_df.sequence==sequence])} = {radiomics_df[radiomics_df.sequence==sequence][target_label].mean():.3f})")
        
        for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():

            print(f"\tCCC threshold: {ccc_stability}")

            for harmonziation_type, data_df in {"normal": radiomics_df[radiomics_df.sequence==sequence].copy(), "combat": radiomics_df[radiomics_df.sequence==sequence].copy()}.items():

                
                if harmonziation_type=="combat":
                    continue
                
                print(f"\t\tHarmonization: {harmonziation_type}")
            
                
                filtered_features = filter_near_zero(data_df[radiomics_features])
                filtered_features = filter_high_corr(data_df[filtered_features])
                stable_features = [feat for feat in filtered_features if feat in stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()]
                
                if harmonziation_type=="combat":
                    data_df[stable_features] = combat_correction(data_df[stable_features], covars=data_df[["sequence"]], batch_col="sequence")
                
                X = data_df[stable_features].values
                y = data_df[target_label].values

                X_scaled_for_pca = StandardScaler().fit_transform(X)

                n_comps = 3
                pca = PCA(n_components=n_comps, random_state=42)
                X_pca = pca.fit_transform(X_scaled_for_pca)

                # Store the transformed components and explained variance
                pca_results[(sequence, target_label, ccc_stability, harmonziation_type)] = pd.DataFrame(data=X_pca, columns=["pc1", "pc2", "pc3"])


                selected_idxs = select_topk_utest(X, y, k=k)

                X_selected = X[:, selected_idxs]

                signatures[(sequence,target_label, ccc_stability, harmonziation_type)] = data_df[np.array(stable_features)[selected_idxs]].reset_index(drop=True)
        
        
                for estimator in piped_estimators:
                    cv = LeaveOneOut()
                    y_pred = cross_val_predict(estimator, X_selected, y, cv=cv, method="predict_proba")[:, 1]

                    auc = roc_auc_score(y, y_pred)
                    print("\t\t\t", estimator[-1].__class__.__name__, f"AUC={auc:.3f}")


config_names = list(signatures.keys())
n_configs = len(config_names)
pearson_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)
spearman_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)

for i in range(n_configs):
    for j in range(n_configs):
        name_A = config_names[i]
        name_B = config_names[j]
        
        pearson_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(signatures[name_A], signatures[name_B], method='pearson')
        spearman_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(signatures[name_A], signatures[name_B], method='spearman')

# Plot Pearson MWMC Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    pearson_mwmc_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=0, 
    vmax=1, 
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title("Pearson Signature Similarity", fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
pearson_output_path = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/pearson_correlation_heatmap.png"
plt.savefig(pearson_output_path, dpi=300)
plt.close()
print(f"Pearson MWMC heatmap saved to: {pearson_output_path}")

# Plot Spearman MWMC Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    spearman_mwmc_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=0, 
    vmax=1, 
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title("Spearman Signature Similarity", fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
spearman_output_path = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/spearman_correlation_heatmap.png"
plt.savefig(spearman_output_path, dpi=300)
plt.close()
print(f"Spearman MWMC heatmap saved to: {spearman_output_path}")


config_names = list(pca_results.keys())
n_configs = len(config_names)
pearson_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)
spearman_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)

for i in range(n_configs):
    for j in range(n_configs):
        name_A = config_names[i]
        name_B = config_names[j]
        
        pearson_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(pca_results[name_A], pca_results[name_B], method='pearson')
        spearman_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(pca_results[name_A], pca_results[name_B], method='spearman')

# Plot Pearson MWMC Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    pearson_mwmc_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=0, 
    vmax=1, 
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title("Pearson PCA Component Similarity", fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
pearson_output_path = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/pearson_pca_component_similarity.png"
plt.savefig(pearson_output_path, dpi=300)
plt.close()
print(f"Pearson MWMC heatmap saved to: {pearson_output_path}")

# Plot Spearman MWMC Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    spearman_mwmc_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=0, 
    vmax=1, 
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title("Spearman PCA Component Similarity", fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
spearman_output_path = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/spearman_pca_component_similarity.png"
plt.savefig(spearman_output_path, dpi=300)
plt.close()
print(f"Spearman MWMC heatmap saved to: {spearman_output_path}")

#%%

stability_df = pd.read_csv(STABILITY_FILE)
stability_df = stability_df[(stability_df.bias_correction==BIAS)&(stability_df.normalization==NORMALIZATION)&(stability_df.tissue==TISSUE)&(stability_df.type==TYPE)]

radiomics_df = pd.read_csv(RADIOMICS_FILE)
grp_radiomics_df = radiomics_df.groupby(by=["pid"]).agg(list).reset_index()
pids_with_both_sequences = grp_radiomics_df[grp_radiomics_df.sequence.apply(lambda x: len(x)==2)].pid.to_list()
radiomics_df = radiomics_df[radiomics_df.pid.isin(pids_with_both_sequences)].reset_index(drop=True)

radiomics_features = [feat for feat in stability_df.feature.to_list() if "shape" not in feat]
## Performance Analysis on the heterogenous data
piped_estimators = [make_pipeline(StandardScaler(), LogisticRegression(random_state=42)), make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1))]

k = 3

signatures = {}
pca_results = {}

for sequence in ["t2w_fs", "t2w_nfs"]:

    print(f"\n**Homogeneous Subgroups: Sequence = {sequence}**")

    for target_label in ["brn_grade>=g1"]:
        print(f"\nTarget label: {target_label} (prevalance = {len(radiomics_df[(radiomics_df.sequence==sequence)&(radiomics_df[target_label]==1)])} / {len(radiomics_df[radiomics_df.sequence==sequence])} = {radiomics_df[radiomics_df.sequence==sequence][target_label].mean():.3f})")
        
        for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():

            print(f"\tCCC threshold: {ccc_stability}")

            for harmonziation_type, data_df in {"normal": radiomics_df[radiomics_df.sequence==sequence].copy(), "combat": radiomics_df[radiomics_df.sequence==sequence].copy()}.items():

                if harmonziation_type=="combat":
                    continue
                
                print(f"\t\tHarmonization: {harmonziation_type}")
            
                
                filtered_features = filter_near_zero(data_df[radiomics_features])
                filtered_features = filter_high_corr(data_df[filtered_features])
                stable_features = [feat for feat in filtered_features if feat in stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()]
                
                if harmonziation_type=="combat":
                    data_df[stable_features] = combat_correction(data_df[stable_features], covars=data_df[["sequence"]], batch_col="sequence")
                
                X = data_df[stable_features].values
                y = data_df[target_label].values

            
                selected_idxs = select_topk_utest(X, y, k=k)

                X_selected = X[:, selected_idxs]

                signatures[(sequence,target_label, ccc_stability, harmonziation_type)] = data_df[np.array(stable_features)[selected_idxs]].reset_index(drop=True)
        
        
                for estimator in piped_estimators:
                    cv = LeaveOneOut()
                    y_pred = cross_val_predict(estimator, X_selected, y, cv=cv, method="predict_proba")[:, 1]

                    auc = roc_auc_score(y, y_pred)
                    print("\t\t\t", estimator[-1].__class__.__name__, f"AUC={auc:.3f}")


config_names = list(signatures.keys())
n_configs = len(config_names)
pearson_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)
spearman_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)

for i in range(n_configs):
    for j in range(n_configs):
        name_A = config_names[i]
        name_B = config_names[j]
        
        pearson_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(signatures[name_A], signatures[name_B], method='pearson')
        spearman_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(signatures[name_A], signatures[name_B], method='spearman')

# Plot Pearson MWMC Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    pearson_mwmc_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=0, 
    vmax=1, 
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title("Pearson Signature Similarity", fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
pearson_output_path = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/pearson_correlation_heatmap.png"
plt.savefig(pearson_output_path, dpi=300)
plt.close()
print(f"Pearson MWMC heatmap saved to: {pearson_output_path}")

# Plot Spearman MWMC Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    spearman_mwmc_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=0, 
    vmax=1, 
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title("Spearman Signature Similarity", fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
spearman_output_path = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/spearman_correlation_heatmap.png"
plt.savefig(spearman_output_path, dpi=300)
plt.close()
print(f"Spearman MWMC heatmap saved to: {spearman_output_path}")


config_names = list(pca_results.keys())
n_configs = len(config_names)
pearson_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)
spearman_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)

for i in range(n_configs):
    for j in range(n_configs):
        name_A = config_names[i]
        name_B = config_names[j]
        
        pearson_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(pca_results[name_A], pca_results[name_B], method='pearson')
        spearman_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(pca_results[name_A], pca_results[name_B], method='spearman')

# Plot Pearson MWMC Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    pearson_mwmc_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=0, 
    vmax=1, 
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title("Pearson PCA Component Similarity", fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
pearson_output_path = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/pearson_pca_component_similarity.png"
plt.savefig(pearson_output_path, dpi=300)
plt.close()
print(f"Pearson MWMC heatmap saved to: {pearson_output_path}")

# Plot Spearman MWMC Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    spearman_mwmc_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=0, 
    vmax=1, 
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title("Spearman PCA Component Similarity", fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
spearman_output_path = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/spearman_pca_component_similarity.png"
plt.savefig(spearman_output_path, dpi=300)
plt.close()
print(f"Spearman MWMC heatmap saved to: {spearman_output_path}")

# print("\n**Unbiased Estimates**")
# # Unbiased estimates
# piped_estimators = [make_pipeline(StandardScaler(), LogisticRegression(random_state=42)), make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1))]

# k = 3

# for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():

#     print(f"CCC threshold: {ccc_stability}")

#     for harmonziation_type, data in {"normal": radiomics_df, "combat": combat_radiomics_df}.items():
        
#         if ccc_threshold>0:
#             if harmonziation_type=="normal":
#                 continue;
        
#         print(f"\tHarmonization: {harmonziation_type}")
#         stable_features = [feat for feat in features if feat in stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()]
        
#         filtered_features = filter_near_zero(data[stable_features])
#         filtered_features = filter_high_corr(data[filtered_features])

#         X, y = radiomics_df[filtered_features].to_numpy(), radiomics_df[TARGET_LABEL].to_numpy()

#         cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
#         for estimator in piped_estimators:

#             aucs = []
#             probs = []
#             targets = []
        
#             for train_index, test_index in cv.split(X, y):
                
#                 X_train, X_test = X[train_index], X[test_index]
#                 y_train, y_test = y[train_index], y[test_index]

#                 selected_idxs = select_topk_utest(X_train, y_train, k=k)

#                 estimator.fit(X_train[:, selected_idxs], y_train)
#                 y_pred = estimator.predict_proba(X_test[:, selected_idxs])[:, 1]
#                 aucs.append(roc_auc_score(y_test, y_pred))
            
#                 probs.extend(y_pred)
#                 targets.extend(y_test)

#             print("\t\t",estimator[-1].__class__.__name__,np.mean(aucs), roc_auc_score(targets, probs))
    


        
#%%
# Performance in homogenous groups

for sequence in ["t2w_fs", "t2w_nfs"]:

    print(f"**only {sequence}: Biased Estimates**")
    piped_estimators = [make_pipeline(StandardScaler(), LogisticRegression(random_state=42)), make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1))]

    k = 3

    for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():

        print(f"CCC threshold: {ccc_stability}")

        print(f"\tHarmonization: normal")
        stable_features = [feat for feat in features if feat in stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()]
        
        filtered_features = filter_near_zero(data[stable_features])
        filtered_features = filter_high_corr(data[filtered_features])
        selected_features = select_topk_utest(X = data[filtered_features], y = data[TARGET_LABEL], k=k)
        
        for estimator in piped_estimators:
            cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
            scores = cross_val_score(estimator, data[selected_features], data[TARGET_LABEL], cv=cv, scoring="roc_auc")
            print("\t\t",estimator[-1].__class__.__name__,np.mean(scores))


    # print(f"\n**{sequence}: Unbiased Estimates**")
    # # Unbiased estimates
    # piped_estimators = [make_pipeline(StandardScaler(), LogisticRegression(random_state=42)), make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1))]

    # k = 3

    # for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():

    #     print(f"CCC threshold: {ccc_stability}")

    #     for harmonziation_type, data in {"normal": radiomics_df, "combat": combat_radiomics_df}.items():
            
    #         if ccc_threshold>0:
    #             if harmonziation_type=="normal":
    #                 continue;
            
    #         print(f"\tHarmonization: {harmonziation_type}")
    #         stable_features = [feat for feat in features if feat in stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()]
            
    #         filtered_features = filter_near_zero(data[stable_features])
    #         filtered_features = filter_high_corr(data[filtered_features])

    #         X, y = radiomics_df[filtered_features].to_numpy(), radiomics_df[TARGET_LABEL].to_numpy()

    #         cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    #         for estimator in piped_estimators:

    #             aucs = []
    #             probs = []
    #             targets = []
            
    #             for train_index, test_index in cv.split(X, y):
                    
    #                 X_train, X_test = X[train_index], X[test_index]
    #                 y_train, y_test = y[train_index], y[test_index]

    #                 selected_idxs = select_topk_utest(X_train, y_train, k=k)

    #                 estimator.fit(X_train[:, selected_idxs], y_train)
    #                 y_pred = estimator.predict_proba(X_test[:, selected_idxs])[:, 1]
    #                 aucs.append(roc_auc_score(y_test, y_pred))
                
    #                 probs.extend(y_pred)
    #                 targets.extend(y_test)

    #             print("\t\t",estimator[-1].__class__.__name__,np.mean(aucs), roc_auc_score(targets, probs))
        


#%%
# Clustering and Dimensionality Reduction Analysis
print("\n**K-Means Clustering Quality & t-SNE Visualization**")

# We will create a grid of plots for the 4 signatures
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()
plot_idx = 0

clustering_results = []

for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():
    for harmonization_type, data in {"normal": radiomics_df, "combat": combat_radiomics_df}.items():
        if ccc_threshold > 0 and harmonization_type == "normal":
            continue
        
        stable_features = [feat for feat in features if feat in stability_df[stability_df.ccc >= ccc_threshold].feature.to_list()]
        
        filtered_features = filter_near_zero(data[stable_features])
        filtered_features = filter_high_corr(data[filtered_features])
        selected_features = select_topk_utest(X=data[filtered_features], y=data[TARGET_LABEL], k=k)
        
        # Prepare the features
        X_sig = data[selected_features].values
        y_sig = data[TARGET_LABEL].values
        
        # Standardize features prior to clustering/t-SNE
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sig)
        
        # 1. K-Means Clustering (n_clusters=2 matching binary target)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate quality metrics
        sil = silhouette_score(X_scaled, cluster_labels)
        ari = adjusted_rand_score(y_sig, cluster_labels)
        nmi = normalized_mutual_info_score(y_sig, cluster_labels)
        
        clustering_results.append({
            "CCC Threshold": ccc_stability,
            "Harmonization": harmonization_type,
            "Signature Features": selected_features,
            "Silhouette Score": sil,
            "Adjusted Rand Index (ARI)": ari,
            "Normalized Mutual Info (NMI)": nmi
        })
        
        print(f"Signature: CCC {ccc_stability} ({harmonization_type}) -> Features: {selected_features}")
        print(f"\tSilhouette Score: {sil:.4f}")
        print(f"\tAdjusted Rand Index (ARI): {ari:.4f}")
        print(f"\tNormalized Mutual Info (NMI): {nmi:.4f}\n")
        
        # 2. t-SNE Projection (n_components=2)
        perplexity = min(30, len(X_scaled) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Plotting
        ax = axes[plot_idx]
        
        tsne_df = pd.DataFrame({
            "t-SNE Dimension 1": X_tsne[:, 0],
            "t-SNE Dimension 2": X_tsne[:, 1],
            "Target Label": y_sig,
            "K-Means Cluster": cluster_labels,
            "Sequence": data["sequence"].values
        })
        
        # Scatter plot colored by Target Label with different marker shapes for K-Means Clusters
        sns.scatterplot(
            x="t-SNE Dimension 1", 
            y="t-SNE Dimension 2", 
            hue="Target Label", 
            style="K-Means Cluster",
            data=tsne_df, 
            palette="Set1", 
            alpha=0.8,
            s=80,
            ax=ax
        )
        
        ax.set_title(f"CCC {ccc_stability} ({harmonization_type})\nSilhouette: {sil:.3f} | ARI: {ari:.3f}", fontsize=11)
        ax.legend(title="Target / Cluster", loc="best", framealpha=0.5)
        
        plot_idx += 1

plt.tight_layout()
output_plot_path = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/tsne_kmeans_clustering.png"
os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
plt.savefig(output_plot_path, dpi=300)
plt.close()

print(f"t-SNE plot saved to: {output_plot_path}")

# Display summary dataframe
df_clustering = pd.DataFrame(clustering_results)
try:
    display(df_clustering)
except NameError:
    print(df_clustering.to_string())


#%%
# Correlation Analysis of Selected Signatures using Mean Weighted Matching Correlation
print("\n**Correlation Analysis of Selected Signatures (Mean Weighted Matching Correlation)**")

from scipy.optimize import linear_sum_assignment
from typing import Literal

def mean_weighted_matching_correlation(df_A, df_B, method: Literal['pearson', 'spearman'] = 'pearson'):
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

# Collect signature data for each configuration
signatures = {}
cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():
    for harmonization_type, data in {"normal": radiomics_df, "combat": combat_radiomics_df}.items():
        # Only for >=0 threshold consider both normal and combat; for the rest, only consider normal
        if ccc_stability != ">=0" and harmonization_type == "combat":
            continue
            
        stable_features = [feat for feat in features if feat in stability_df[stability_df.ccc >= ccc_threshold].feature.to_list()]
        
        filtered_features = filter_near_zero(data[stable_features])
        filtered_features = filter_high_corr(data[filtered_features])
        selected_features = select_topk_utest(X=data[filtered_features], y=data[TARGET_LABEL], k=k)
        
        for estimator in piped_estimators:
            est_name = estimator[-1].__class__.__name__
            est_abbr = "LR" if est_name == "LogisticRegression" else "RF"
            
            config_name = f"{est_abbr} ({ccc_stability}, {harmonization_type})"
            sig_df = pd.DataFrame()
            
            # 1. Add the selected features (from the corresponding data frame)
            for feat in selected_features:
                sig_df[feat] = data[feat].values
                
            # 2. Add the estimator predictions (out-of-fold predicted probabilities)
            y_pred = cross_val_predict(
                estimator, 
                data[selected_features], 
                data[TARGET_LABEL], 
                cv=cv, 
                method='predict_proba'
            )[:, 1]
            sig_df["pred_prob"] = y_pred
            
            signatures[config_name] = sig_df

# Compute pairwise Mean Weighted Matching Correlation (MWMC) matrices
config_names = list(signatures.keys())
n_configs = len(config_names)

pearson_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)
spearman_mwmc_matrix = pd.DataFrame(np.zeros((n_configs, n_configs)), index=config_names, columns=config_names)

for i in range(n_configs):
    for j in range(n_configs):
        name_A = config_names[i]
        name_B = config_names[j]
        
        pearson_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(signatures[name_A], signatures[name_B], method='pearson')
        spearman_mwmc_matrix.iloc[i, j] = mean_weighted_matching_correlation(signatures[name_A], signatures[name_B], method='spearman')

# Plot Pearson MWMC Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    pearson_mwmc_matrix, 
    annot=True, 
    fmt=".3f", 
    cmap="coolwarm", 
    vmin=0, 
    vmax=1, 
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title("Pearson Mean Weighted Matching Correlation\nBetween Signature Configurations", fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
pearson_output_path = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/pearson_correlation_heatmap.png"
plt.savefig(pearson_output_path, dpi=300)
plt.close()
print(f"Pearson MWMC heatmap saved to: {pearson_output_path}")

# Plot Spearman MWMC Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    spearman_mwmc_matrix, 
    annot=True, 
    fmt=".3f", 
    cmap="coolwarm", 
    vmin=0, 
    vmax=1, 
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title("Spearman Mean Weighted Matching Correlation\nBetween Signature Configurations", fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
spearman_output_path = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/spearman_correlation_heatmap.png"
plt.savefig(spearman_output_path, dpi=300)
plt.close()
print(f"Spearman MWMC heatmap saved to: {spearman_output_path}")


#%%
# Within-Signature Pearson Correlation Heatmaps
print("\n**Generating Within-Signature Pearson Correlation Heatmaps**")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()
plot_idx = 0

for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():
    for harmonization_type, data in {"normal": radiomics_df, "combat": combat_radiomics_df}.items():
        # Only for >=0 threshold consider both normal and combat; for the rest, only consider normal
        if ccc_stability != ">=0" and harmonization_type == "combat":
            continue
            
        stable_features = [feat for feat in features if feat in stability_df[stability_df.ccc >= ccc_threshold].feature.to_list()]
        
        filtered_features = filter_near_zero(data[stable_features])
        filtered_features = filter_high_corr(data[filtered_features])
        selected_features = select_topk_utest(X=data[filtered_features], y=data[TARGET_LABEL], k=k)
        
        # DataFrame representing the signature elements
        sig_df = pd.DataFrame()
        for feat in selected_features:
            # Clean feature names for visualization
            short_feat = feat.replace("original_gldm_", "").replace("original_glcm_", "").replace("original_firstorder_", "")
            sig_df[short_feat] = data[feat].values
            
        # Get out-of-fold predictions from both estimators
        for estimator in piped_estimators:
            est_name = estimator[-1].__class__.__name__
            est_abbr = "LR_pred" if est_name == "LogisticRegression" else "RF_pred"
            
            y_pred = cross_val_predict(
                estimator, 
                data[selected_features], 
                data[TARGET_LABEL], 
                cv=cv, 
                method='predict_proba'
            )[:, 1]
            sig_df[est_abbr] = y_pred
            
        # Compute Pearson correlation matrix
        corr_matrix = sig_df.corr(method='pearson')
        
        # Plot
        ax = axes[plot_idx]
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            vmin=-1, 
            vmax=1, 
            square=True,
            ax=ax,
            cbar_kws={"shrink": .8}
        )
        ax.set_title(f"CCC {ccc_stability} ({harmonization_type})", fontsize=12, fontweight='bold', pad=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
        
        plot_idx += 1

plt.suptitle("Pearson Correlation Heatmaps Within Each Signature Configuration", fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
sig_corr_plot_path = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/features_correlation_heatmaps.png"
os.makedirs(os.path.dirname(sig_corr_plot_path), exist_ok=True)
plt.savefig(sig_corr_plot_path, dpi=300)
plt.close()
print(f"Within-signature correlation heatmaps saved to: {sig_corr_plot_path}")



