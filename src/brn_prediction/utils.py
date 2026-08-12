import numpy as np
import pandas as pd
from scipy import stats
from scipy import special


## Confidence interval estimation using boostrapping

## Confidence interval estimation using boostrapping

def get_bootstrap_sample(X_df, y_df, n_splits=2):
    """
    Returns a stratified bootstrap sample of X and y,
    preserving class proportions.
    """
    
    df = pd.concat([X_df, y_df], axis=1)
    
    while True:
        sampled_df = df.sample(n=len(df), replace=True).reset_index(drop=True)
        if sampled_df[y_df.name].sum()>=n_splits:
            break;

    return sampled_df[X_df.columns], sampled_df[y_df.name]


def percentile_bootstrap_ci(X_df, y_df, statistic, n_splits, n_iterations, alpha=0.05):
    
    true_statistic = statistic(X_df, y_df)
    
    assert np.isscalar(true_statistic), "the statistic method should return a scalar value"
    
    bootstrap_statistics = []
    
    for i in range(n_iterations):
        
        X_sample, y_sample = get_bootstrap_sample(X_df, y_df, n_splits) 
        estimate = statistic(X_sample, y_sample)
        bootstrap_statistics.append(estimate)

    ci_boot = np.quantile(bootstrap_statistics, [alpha/2, 1-(alpha/2)])
    
    return true_statistic, ci_boot, bootstrap_statistics

def bca_bootstrap_ci(X_df, y_df, statistic, bootstrap_statistics, alpha=0.05): #bias corrected and accelerated
    """
    Calculate Bias corrected and accelerated confidence intervals for a bootstrap distribution.
    
    https://www.tau.ac.il/~saharon/Boot/10.1.1.133.8405.pdf
    
    equations: https://www.erikdrysdale.com/bca_python/
    
    Parameters:
        X_df the dataframe with features
        y_df the pandas series with values
        statistic: the function that will return a scalar estimate of the statistic of interest
        bootstrapped_statistics: Array of bootstrap statistics.
        alpha: Significance level (default 0.05 for 95% CI).
    
    Returns:
        Lower and upper bounds of the confidence interval.
    """
    # Sort the bootstrap statistics
    
    true_statistic = statistic(X_df, y_df)
    assert np.isscalar(true_statistic), "the statistic function should return a scalar value"
    
    # bootstrap_statistics = np.append(bootstrap_statistics, true_statistic) 
    bootstrap_statistics = np.sort(bootstrap_statistics)
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
    jackknife_estimates = np.array([statistic(X_df.drop(index=i, inplace=False), y_df.drop(index=i, inplace=False)) 
                                    for i in X_df.index])
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
    
    ci = np.quantile(bootstrap_statistics, [alpha_1, alpha_2])

   
    return true_statistic, ci, bootstrap_statistics



## Delong's test
def compare_aucs(probs1, probs2, targets):

    z_score, p_value = Delong_test(targets, probs1, probs2)
    
    return p_value

