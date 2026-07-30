'''
[] mann-whitney u with all features -> predictive modelling
    [] using logisticregression
    [] using RF
[] mann-whitney u with all features and combat correction-> predictive modelling
    [] using logisticregression
    [] using RF
[] mann-whitney u with stable features -> predictive modelling
    [] using logisticregression
    [] using RF
[] pca all features -> 3 components -> predictive modelling
    [] using logisticregression
    [] using RF
[] pca stable features -> 3 components -> predictive modelling
    [] using logisticregression
    [] using RF

[] Try >=G2 as the target label
[] Try >=G1 as the target label
[] Both with Combat
'''

#%%
from sklearn.utils import metaestimators
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from neuroCombat import neuroCombat

import pandas as pd
import numpy as np

STABILITY_FILE = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/stability_df.csv"
RADIOMICS_FILE = r"/home/thulasiseetha/research/sithin_projects/t2w_stability/outputs/radiomicsFeatures3D.csv"
BIAS = "none"
NORMALIZATION = "none"
TISSUE = "gwm"
TYPE = "combat"
CCC_THRESHOLDS = {">=0":-np.inf, ">=good":0.70, ">=excellent":0.85}
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

def select_topk_utest(X, y, k=5, features = None):

    if isinstance(X, pd.DataFrame):
        features = X.columns.to_list()
        X, y = X.to_numpy(), y.to_numpy()
    
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    p_values = []
    for i in range(X.shape[1]):  
        _, p_val = mannwhitneyu(class_0[:, i], class_1[:, i])
        p_values.append(p_val)

    p_values = np.array(p_values)
    # Select top k features with lowest p-values
    selected_idxs = np.argsort(p_values)[:k]
    selected = np.array(features)[selected_idxs].tolist() if features else selected_idxs

    return selected

#%%
# Biased Estimates
stability_df = pd.read_csv(STABILITY_FILE)
stability_df = stability_df[(stability_df.bias_correction==BIAS)&(stability_df.normalization==NORMALIZATION)&(stability_df.tissue==TISSUE)&(stability_df.type==TYPE)]
radiomics_df = pd.read_csv(RADIOMICS_FILE)
radiomics_df = radiomics_df[radiomics_df.exclude==0].reset_index(drop=True)

features = [feat for feat in stability_df.feature.to_list() if "shape" not in feat]
combat_radiomics_df = radiomics_df.copy()
combat_radiomics_df[features] = combat_correction(radiomics_df[features], covars=radiomics_df[["pid", "sequence"]], batch_col="sequence")

display(combat_radiomics_df.head())

#%%
## Performance Analysis on the heterogenous data

print("**Biased Estimates**")
piped_estimators = [make_pipeline(StandardScaler(), LogisticRegression(random_state=42)), make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1))]

k = 3

for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():

    print(f"CCC threshold: {ccc_stability}")

    for harmonziation_type, data in {"normal": radiomics_df, "combat": combat_radiomics_df}.items():

        if ccc_threshold>0:
            if harmonziation_type=="normal":
                continue;
        print(f"\tHarmonization: {harmonziation_type}")
        stable_features = [feat for feat in features if feat in stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()]
        
        filtered_features = filter_near_zero(data[stable_features])
        filtered_features = filter_high_corr(data[filtered_features])
        selected_features = select_topk_utest(X = data[filtered_features], y = data[TARGET_LABEL], k=k)
        
        for estimator in piped_estimators:
            cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
            scores = cross_val_score(estimator, data[selected_features], data[TARGET_LABEL], cv=cv, scoring="roc_auc")
            print("\t\t",estimator[-1].__class__.__name__,np.mean(scores))


print("\n**Unbiased Estimates**")
# Unbiased estimates
piped_estimators = [make_pipeline(StandardScaler(), LogisticRegression(random_state=42)), make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1))]

k = 3

for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():

    print(f"CCC threshold: {ccc_stability}")

    for harmonziation_type, data in {"normal": radiomics_df, "combat": combat_radiomics_df}.items():
        
        if ccc_threshold>0:
            if harmonziation_type=="normal":
                continue;
        
        print(f"\tHarmonization: {harmonziation_type}")
        stable_features = [feat for feat in features if feat in stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()]
        
        filtered_features = filter_near_zero(data[stable_features])
        filtered_features = filter_high_corr(data[filtered_features])

        X, y = radiomics_df[filtered_features].to_numpy(), radiomics_df[TARGET_LABEL].to_numpy()

        cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
        for estimator in piped_estimators:

            aucs = []
            probs = []
            targets = []
        
            for train_index, test_index in cv.split(X, y):
                
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                selected_idxs = select_topk_utest(X_train, y_train, k=k)

                estimator.fit(X_train[:, selected_idxs], y_train)
                y_pred = estimator.predict_proba(X_test[:, selected_idxs])[:, 1]
                aucs.append(roc_auc_score(y_test, y_pred))
            
                probs.extend(y_pred)
                targets.extend(y_test)

            print("\t\t",estimator[-1].__class__.__name__,np.mean(aucs), roc_auc_score(targets, probs))
    


        
#%%
# Performance in homogenous groups

for sequence in ["t2w_fs", "t2w_nfs"]:

    print(f"**only {sequence}: Biased Estimates**")
    piped_estimators = [make_pipeline(StandardScaler(), LogisticRegression(random_state=42)), make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1))]

    k = 3

    for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():

        print(f"CCC threshold: {ccc_stability}")

        for harmonziation_type, data in {"normal": radiomics_df[radiomics_df.sequence==sequence], "combat": combat_radiomics_df[combat_radiomics_df.sequence==sequence]}.items():

            if ccc_threshold>0:
                if harmonziation_type=="normal":
                    continue;
            print(f"\tHarmonization: {harmonziation_type}")
            stable_features = [feat for feat in features if feat in stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()]
            
            filtered_features = filter_near_zero(data[stable_features])
            filtered_features = filter_high_corr(data[filtered_features])
            selected_features = select_topk_utest(X = data[filtered_features], y = data[TARGET_LABEL], k=k)
            
            for estimator in piped_estimators:
                cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
                scores = cross_val_score(estimator, data[selected_features], data[TARGET_LABEL], cv=cv, scoring="roc_auc")
                print("\t\t",estimator[-1].__class__.__name__,np.mean(scores))


    print(f"\n**{sequence}: Unbiased Estimates**")
    # Unbiased estimates
    piped_estimators = [make_pipeline(StandardScaler(), LogisticRegression(random_state=42)), make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_jobs=-1))]

    k = 3

    for ccc_stability, ccc_threshold in CCC_THRESHOLDS.items():

        print(f"CCC threshold: {ccc_stability}")

        for harmonziation_type, data in {"normal": radiomics_df, "combat": combat_radiomics_df}.items():
            
            if ccc_threshold>0:
                if harmonziation_type=="normal":
                    continue;
            
            print(f"\tHarmonization: {harmonziation_type}")
            stable_features = [feat for feat in features if feat in stability_df[stability_df.ccc>=ccc_threshold].feature.to_list()]
            
            filtered_features = filter_near_zero(data[stable_features])
            filtered_features = filter_high_corr(data[filtered_features])

            X, y = radiomics_df[filtered_features].to_numpy(), radiomics_df[TARGET_LABEL].to_numpy()

            cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
            for estimator in piped_estimators:

                aucs = []
                probs = []
                targets = []
            
                for train_index, test_index in cv.split(X, y):
                    
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    selected_idxs = select_topk_utest(X_train, y_train, k=k)

                    estimator.fit(X_train[:, selected_idxs], y_train)
                    y_pred = estimator.predict_proba(X_test[:, selected_idxs])[:, 1]
                    aucs.append(roc_auc_score(y_test, y_pred))
                
                    probs.extend(y_pred)
                    targets.extend(y_test)

                print("\t\t",estimator[-1].__class__.__name__,np.mean(aucs), roc_auc_score(targets, probs))
        

