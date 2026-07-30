import numpy as np
from neuroCombat import neuroCombat
import pandas as pd
from scipy import stats
from scipy import special



def ccc(y1, y2, axis=-1):
    
    e = 1e-12
    
    mean_1 = np.mean(y1, axis=axis).reshape(-1, 1)
    mean_2 = np.mean(y2, axis=axis).reshape(-1, 1)
    
    var_1 = (np.std(y1, axis=axis)**2).reshape(-1,1)
    var_2 = (np.std(y2, axis=axis)**2).reshape(-1,1)

    covar_12 = np.mean((y1-mean_1)*(y2-mean_2), axis=axis).reshape(-1,1)
    
    numerator = 2 * covar_12
    denominator = var_1 + var_2 + (mean_1 - mean_2)**2
    

    return numerator/(denominator+e)


def compute_ccc(df1, df2):
    
    features = df1.columns.to_list()
    ccc_df = {"feature":[], "ccc":[]}
    for feat in features:
        ccc_df["feature"].append(feat)
        ccc_df["ccc"].append(ccc(df1[feat].values, df2[feat].values).item())
        
    return pd.DataFrame(ccc_df)
    
def combat_correction(feats_df, covars_df, batch_col):

    assert batch_col in covars_df.columns, "batch_col should be in covars_df"
    
    features = feats_df.columns.to_list()
    data = feats_df.copy()
    covars = pd.DataFrame({col:covars_df[col].astype("category").cat.codes for col in covars_df.columns})

    combat_feats = neuroCombat(dat=data.T, covars = covars, batch_col=batch_col)["data"].T
    combat_feats_df = pd.DataFrame(combat_feats, columns=features, index=feats_df.index)

    return combat_feats_df


## Confidence interval estimation using boostrapping
def bca_bootstrap_ci(estimator, fs_df, nfs_df, sample_estimate, bootstrap_estimates, alpha=0.05): #bias corrected and accelerated
    """
    Calculate Bias corrected and accelerated confidence intervals for a bootstrap distribution.
    
    https://www.tau.ac.il/~saharon/Boot/10.1.1.133.8405.pdf
    
    equations: https://www.erikdrysdale.com/bca_python/
    
    Parameters:
    
    Returns:
        Lower and upper bounds of the confidence interval.
    """
    # Sort the bootstrap statistics
    
    true_statistic = sample_estimate
    bootstrap_statistics = np.sort(bootstrap_estimates)
    B = len(bootstrap_statistics)

    # Bias correction factor (z0)
    # https://github.com/scipy/scipy/blob/v1.16.2/scipy/stats/_resampling.py#L73
    ## Option 1
    prop = ((bootstrap_statistics < true_statistic).sum() + (bootstrap_statistics <= true_statistic).sum()) / (2*B) 
    prop = np.clip(prop, 0.1, 0.9) #this prevent weird answers for extremely skewed bootstrap distributions
    z0 = special.ndtri(prop)
    
    ##Option 2
    # prop = np.mean(bootstrap_statistics < true_statistic)
    # z0 = special.ndtri(prop)
    
    # Acceleration constant (a)
    # jackknife estimates (leave one out)
    jackknife_estimates = np.array([estimator(fs_df.drop(index=i, inplace=False), nfs_df.drop(index=i, inplace=False)) for i in fs_df.index])
    mean_jackknife = np.mean(jackknife_estimates)
    
    num = np.sum((mean_jackknife - jackknife_estimates) ** 3)
    den = (6 * np.sum((mean_jackknife - jackknife_estimates) ** 2) ** 1.5)
    a =  num/den if den!=0 else 0.0 

    ## Option 1: Adjusted percentiles
    # z_alpha_low = stats.norm.ppf(alpha / 2)  # Lower z-score
    # z_alpha_high = stats.norm.ppf(1 - alpha / 2)  # Upper z-score
    
    # lower_percentile = stats.norm.cdf(z0 + ((z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low))))
    # upper_percentile = stats.norm.cdf(z0 + ((z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high))))
    
    # ci = np.quantile(bootstrap_statistics, [lower_percentile, upper_percentile])
    
    ## Option 2:
    z_alpha = special.ndtri(alpha)
    z_1alpha = -z_alpha
    
    num1 = z0 + z_alpha
    alpha_1 = special.ndtr(z0 + num1/(1 - a*num1))
    num2 = z0 + z_1alpha
    alpha_2 = special.ndtr(z0 + num2/(1 - a*num2))
    
    corr_ci = np.quantile(bootstrap_statistics, [alpha_1, alpha_2])

    return corr_ci


## Delong's test
def compare_aucs(probs1, probs2, targets):

    z_score, p_value = Delong_test(targets, probs1, probs2)
    
    return p_value

