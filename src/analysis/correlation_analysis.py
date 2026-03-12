"""
Correlation and Feature Analysis Module

This module provides functions for correlation analysis including:
- Pearson correlation
- Cross-correlation functions (CCF)
- Lag relationship identification
- Feature importance analysis for storm prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.signal import correlate
from sklearn.feature_selection import mutual_info_regression
import warnings


def compute_correlations(df: pd.DataFrame,
                         target: str,
                         predictors: List[str],
                         method: str = 'pearson') -> pd.DataFrame:
    """
    Compute correlations between target and predictor variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target : str
        Target variable name
    predictors : List[str]
        List of predictor variable names
    method : str, optional
        Correlation method: 'pearson', 'spearman', 'kendall' (default: 'pearson')
    
    Returns
    -------
    pd.DataFrame
        DataFrame with correlation coefficients and p-values
    
    Example
    -------
    >>> corr_df = compute_correlations(df, 'Kp', ['Bz_GSM', 'plasma_speed'])
    """
    results = []
    
    available_predictors = [p for p in predictors if p in df.columns]
    
    for pred in available_predictors:
        # Get aligned data without NaN
        data = df[[target, pred]].dropna()
        
        if len(data) < 3:
            continue
        
        if method == 'pearson':
            corr, pvalue = stats.pearsonr(data[target], data[pred])
        elif method == 'spearman':
            corr, pvalue = stats.spearmanr(data[target], data[pred])
        elif method == 'kendall':
            corr, pvalue = stats.kendalltau(data[target], data[pred])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results.append({
            'predictor': pred,
            'target': target,
            'correlation': corr,
            'p_value': pvalue,
            'significant': pvalue < 0.05,
            'n_observations': len(data),
            'method': method
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('correlation', key=abs, ascending=False)
    
    return results_df


def compute_pearson_correlation(series1: pd.Series,
                                 series2: pd.Series,
                                 name1: str = 'X',
                                 name2: str = 'Y') -> Dict:
    """
    Compute Pearson correlation between two series with detailed statistics.
    
    Parameters
    ----------
    series1 : pd.Series
        First time series
    series2 : pd.Series
        Second time series
    name1 : str, optional
        Name of first series
    name2 : str, optional
        Name of second series
    
    Returns
    -------
    Dict
        Dictionary with correlation statistics
    
    Example
    -------
    >>> result = compute_pearson_correlation(df['Bz_GSM'], df['DST'], 'Bz', 'DST')
    """
    # Align series
    aligned = pd.DataFrame({name1: series1, name2: series2}).dropna()
    
    if len(aligned) < 3:
        return {'error': 'Insufficient data points'}
    
    x = aligned[name1].values
    y = aligned[name2].values
    
    # Correlation
    corr, pvalue = stats.pearsonr(x, y)
    
    # R-squared
    r_squared = corr ** 2
    
    # Confidence interval (Fisher's z-transformation)
    n = len(x)
    z = np.arctanh(corr)
    se = 1 / np.sqrt(n - 3)
    z_lower = z - 1.96 * se
    z_upper = z + 1.96 * se
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)
    
    result = {
        'variable_1': name1,
        'variable_2': name2,
        'correlation': corr,
        'r_squared': r_squared,
        'p_value': pvalue,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_observations': n,
        'is_significant': pvalue < 0.05
    }
    
    # Interpretation
    abs_corr = abs(corr)
    if abs_corr >= 0.8:
        strength = 'very strong'
    elif abs_corr >= 0.6:
        strength = 'strong'
    elif abs_corr >= 0.4:
        strength = 'moderate'
    elif abs_corr >= 0.2:
        strength = 'weak'
    else:
        strength = 'very weak'
    
    direction = 'positive' if corr > 0 else 'negative'
    result['interpretation'] = f'{strength} {direction}'
    
    return result


def compute_cross_correlation(series1: pd.Series,
                               series2: pd.Series,
                               max_lag: int = 48,
                               normalize: bool = True) -> pd.DataFrame:
    """
    Compute cross-correlation function between two time series.
    
    Positive lag means series1 leads series2.
    
    Parameters
    ----------
    series1 : pd.Series
        First time series (typically the predictor)
    series2 : pd.Series
        Second time series (typically the target)
    max_lag : int, optional
        Maximum lag to compute (default: 48)
    normalize : bool, optional
        Whether to normalize the correlation (default: True)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with lag and cross-correlation values
    
    Example
    -------
    >>> ccf = compute_cross_correlation(df['Bz_GSM'], df['DST'], max_lag=72)
    """
    # Align and clean
    aligned = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
    s1 = aligned['s1'].values
    s2 = aligned['s2'].values
    n = len(s1)
    
    if normalize:
        s1 = (s1 - s1.mean()) / s1.std()
        s2 = (s2 - s2.mean()) / s2.std()
    
    # Compute cross-correlation for different lags
    lags = list(range(-max_lag, max_lag + 1))
    ccf_values = []
    
    for lag in lags:
        if lag < 0:
            # Series1 lags behind series2
            ccf = np.corrcoef(s1[:lag], s2[-lag:])[0, 1]
        elif lag > 0:
            # Series1 leads series2
            ccf = np.corrcoef(s1[lag:], s2[:-lag])[0, 1]
        else:
            ccf = np.corrcoef(s1, s2)[0, 1]
        ccf_values.append(ccf)
    
    # Confidence bounds
    conf_bound = 1.96 / np.sqrt(n)
    
    result_df = pd.DataFrame({
        'lag': lags,
        'ccf': ccf_values,
        'is_significant': [abs(v) > conf_bound for v in ccf_values]
    })
    
    # Find optimal lag
    max_idx = np.argmax(np.abs(ccf_values))
    result_df.attrs['optimal_lag'] = lags[max_idx]
    result_df.attrs['max_ccf'] = ccf_values[max_idx]
    result_df.attrs['conf_bound'] = conf_bound
    
    return result_df


def identify_lag_relationships(df: pd.DataFrame,
                                targets: List[str],
                                predictors: List[str],
                                max_lag: int = 48,
                                significance_threshold: float = 0.05) -> pd.DataFrame:
    """
    Identify optimal lag relationships between predictors and targets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    targets : List[str]
        Target variable names
    predictors : List[str]
        Predictor variable names
    max_lag : int, optional
        Maximum lag to search (default: 48)
    significance_threshold : float, optional
        P-value threshold for significance (default: 0.05)
    
    Returns
    -------
    pd.DataFrame
        DataFrame summarizing lag relationships
    
    Example
    -------
    >>> lag_df = identify_lag_relationships(df, ['Kp', 'DST'], ['Bz_GSM', 'plasma_speed'])
    """
    results = []
    
    available_targets = [t for t in targets if t in df.columns]
    available_predictors = [p for p in predictors if p in df.columns]
    
    for target in available_targets:
        for predictor in available_predictors:
            # Compute cross-correlation
            ccf_result = compute_cross_correlation(
                df[predictor], df[target], max_lag=max_lag
            )
            
            optimal_lag = ccf_result.attrs['optimal_lag']
            max_ccf = ccf_result.attrs['max_ccf']
            
            # Test significance at optimal lag
            aligned = pd.DataFrame({'pred': df[predictor], 'target': df[target]}).dropna()
            n = len(aligned)
            
            # Fisher transform for p-value
            z = np.arctanh(max_ccf)
            se = 1 / np.sqrt(n - 3)
            z_stat = abs(z) / se
            p_value = 2 * (1 - stats.norm.cdf(z_stat))
            
            results.append({
                'predictor': predictor,
                'target': target,
                'optimal_lag': optimal_lag,
                'max_correlation': max_ccf,
                'abs_correlation': abs(max_ccf),
                'p_value': p_value,
                'is_significant': p_value < significance_threshold,
                'interpretation': f'{predictor} leads {target} by {optimal_lag} hours' if optimal_lag > 0 
                                  else f'{target} leads {predictor} by {-optimal_lag} hours' if optimal_lag < 0
                                  else f'{predictor} and {target} are contemporaneous'
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('abs_correlation', ascending=False)
    
    return results_df


def compute_mutual_information(df: pd.DataFrame,
                                target: str,
                                predictors: List[str],
                                n_neighbors: int = 3) -> pd.DataFrame:
    """
    Compute mutual information between target and predictors.
    
    Mutual information captures non-linear relationships.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target : str
        Target variable name
    predictors : List[str]
        Predictor variable names
    n_neighbors : int, optional
        Number of neighbors for MI estimation (default: 3)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with mutual information scores
    
    Example
    -------
    >>> mi_df = compute_mutual_information(df, 'Kp', ['Bz_GSM', 'plasma_speed'])
    """
    available_predictors = [p for p in predictors if p in df.columns]
    
    # Prepare data
    data = df[[target] + available_predictors].dropna()
    X = data[available_predictors]
    y = data[target]
    
    # Compute mutual information
    mi_scores = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=42)
    
    results_df = pd.DataFrame({
        'predictor': available_predictors,
        'mutual_information': mi_scores
    })
    results_df = results_df.sort_values('mutual_information', ascending=False)
    
    # Normalize to [0, 1] for interpretation
    results_df['mi_normalized'] = results_df['mutual_information'] / results_df['mutual_information'].max()
    
    return results_df


def create_lagged_features(df: pd.DataFrame,
                           columns: List[str],
                           lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features for specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : List[str]
        Columns to lag
    lags : List[int]
        List of lag values (positive integers)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with original and lagged features
    
    Example
    -------
    >>> df_lagged = create_lagged_features(df, ['Bz_GSM', 'plasma_speed'], [1, 2, 3, 6, 12])
    """
    df_result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df_result[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    return df_result


def compute_rolling_correlation(series1: pd.Series,
                                 series2: pd.Series,
                                 window: int = 168,
                                 min_periods: int = 24) -> pd.Series:
    """
    Compute rolling correlation between two time series.
    
    Parameters
    ----------
    series1 : pd.Series
        First time series
    series2 : pd.Series
        Second time series
    window : int, optional
        Rolling window size (default: 168 = 1 week of hourly data)
    min_periods : int, optional
        Minimum observations for correlation (default: 24)
    
    Returns
    -------
    pd.Series
        Rolling correlation values
    """
    aligned = pd.DataFrame({'s1': series1, 's2': series2})
    rolling_corr = aligned['s1'].rolling(window=window, min_periods=min_periods).corr(aligned['s2'])
    
    return rolling_corr


def create_correlation_summary(df: pd.DataFrame,
                                targets: List[str],
                                predictors: List[str],
                                include_lags: bool = True,
                                max_lag: int = 24) -> pd.DataFrame:
    """
    Create comprehensive correlation summary table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    targets : List[str]
        Target variable names
    predictors : List[str]
        Predictor variable names
    include_lags : bool, optional
        Whether to include lag analysis (default: True)
    max_lag : int, optional
        Maximum lag for analysis (default: 24)
    
    Returns
    -------
    pd.DataFrame
        Summary table with correlation analysis
    
    Example
    -------
    >>> summary = create_correlation_summary(df, ['Kp', 'DST'], ['Bz_GSM', 'plasma_speed'])
    """
    results = []
    
    available_targets = [t for t in targets if t in df.columns]
    available_predictors = [p for p in predictors if p in df.columns]
    
    for target in available_targets:
        for predictor in available_predictors:
            row = {
                'target': target,
                'predictor': predictor
            }
            
            # Pearson correlation at lag 0
            pearson = compute_pearson_correlation(
                df[predictor], df[target], predictor, target
            )
            if 'error' not in pearson:
                row['pearson_r'] = pearson['correlation']
                row['pearson_p'] = pearson['p_value']
                row['pearson_r2'] = pearson['r_squared']
            
            # Spearman correlation
            corr_df = compute_correlations(df, target, [predictor], method='spearman')
            if len(corr_df) > 0:
                row['spearman_r'] = corr_df.iloc[0]['correlation']
                row['spearman_p'] = corr_df.iloc[0]['p_value']
            
            # Lag analysis
            if include_lags:
                ccf_result = compute_cross_correlation(
                    df[predictor], df[target], max_lag=max_lag
                )
                row['optimal_lag'] = ccf_result.attrs['optimal_lag']
                row['max_ccf'] = ccf_result.attrs['max_ccf']
            
            results.append(row)
    
    summary_df = pd.DataFrame(results)
    
    # Add predictive power ranking
    if 'max_ccf' in summary_df.columns:
        summary_df['predictive_rank'] = summary_df.groupby('target')['max_ccf'].apply(
            lambda x: x.abs().rank(ascending=False)
        ).values
    
    return summary_df


def analyze_storm_predictors(df: pd.DataFrame,
                              storm_column: str = 'is_storm',
                              predictors: Optional[List[str]] = None,
                              pre_storm_hours: int = 24) -> pd.DataFrame:
    """
    Analyze which predictors show significant changes before storms.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with storm indicators
    storm_column : str, optional
        Column name for storm indicator (default: 'is_storm')
    predictors : List[str], optional
        Predictor variables to analyze
    pre_storm_hours : int, optional
        Hours before storm to analyze (default: 24)
    
    Returns
    -------
    pd.DataFrame
        Analysis of predictor behavior before storms
    """
    if predictors is None:
        predictors = ['Bz_GSM', 'Bz_GSE', 'plasma_speed', 'proton_density', 
                      'flow_pressure', 'B_mag_avg', 'electric_field']
    
    available_predictors = [p for p in predictors if p in df.columns]
    
    if storm_column not in df.columns:
        warnings.warn(f"Storm column '{storm_column}' not found. Run detect_storms() first.")
        return pd.DataFrame()
    
    results = []
    
    # Find storm onset times
    storm_starts = df[df[storm_column] & ~df[storm_column].shift(1).fillna(False)].index
    
    for predictor in available_predictors:
        pre_storm_values = []
        quiet_values = []
        
        for storm_start in storm_starts:
            try:
                # Get values 1-24 hours before storm
                pre_storm_data = df.loc[:storm_start].iloc[-(pre_storm_hours+1):-1][predictor].dropna()
                pre_storm_values.extend(pre_storm_data.values)
            except Exception:
                continue
        
        # Get quiet time values (non-storm periods)
        quiet_data = df[~df[storm_column]][predictor].dropna()
        quiet_values = quiet_data.values
        
        if len(pre_storm_values) > 10 and len(quiet_values) > 10:
            # Statistical tests
            pre_storm_arr = np.array(pre_storm_values)
            quiet_arr = np.array(quiet_values)
            
            # T-test (difference in means)
            t_stat, t_pvalue = stats.ttest_ind(pre_storm_arr, quiet_arr)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pvalue = stats.mannwhitneyu(pre_storm_arr, quiet_arr, alternative='two-sided')
            
            results.append({
                'predictor': predictor,
                'pre_storm_mean': np.mean(pre_storm_arr),
                'quiet_mean': np.mean(quiet_arr),
                'mean_difference': np.mean(pre_storm_arr) - np.mean(quiet_arr),
                'pre_storm_std': np.std(pre_storm_arr),
                'quiet_std': np.std(quiet_arr),
                't_statistic': t_stat,
                't_pvalue': t_pvalue,
                'mannwhitney_u': u_stat,
                'mannwhitney_p': u_pvalue,
                'significant_difference': (t_pvalue < 0.05) or (u_pvalue < 0.05),
                'n_pre_storm': len(pre_storm_arr),
                'n_quiet': len(quiet_arr)
            })
    
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('t_pvalue')
    
    return results_df


def compute_partial_correlation(df: pd.DataFrame,
                                 var1: str,
                                 var2: str,
                                 control_vars: List[str]) -> Dict:
    """
    Compute partial correlation between two variables, controlling for others.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    var1 : str
        First variable
    var2 : str
        Second variable
    control_vars : List[str]
        Variables to control for
    
    Returns
    -------
    Dict
        Partial correlation results
    """
    available_controls = [c for c in control_vars if c in df.columns and c not in [var1, var2]]
    
    all_vars = [var1, var2] + available_controls
    data = df[all_vars].dropna()
    
    if len(data) < len(all_vars) + 10:
        return {'error': 'Insufficient data'}
    
    # Compute correlation matrix
    corr_matrix = data.corr()
    
    # Extract relevant correlations
    r_12 = corr_matrix.loc[var1, var2]
    
    if len(available_controls) == 0:
        # No control variables, return simple correlation
        return {
            'partial_correlation': r_12,
            'controlled_for': [],
            'note': 'No control variables'
        }
    
    # For multiple control variables, use matrix inversion method
    try:
        inv_corr = np.linalg.inv(corr_matrix.values)
        # Partial correlation from precision matrix
        i = list(corr_matrix.columns).index(var1)
        j = list(corr_matrix.columns).index(var2)
        partial_r = -inv_corr[i, j] / np.sqrt(inv_corr[i, i] * inv_corr[j, j])
        
        # P-value (approximate)
        n = len(data)
        k = len(available_controls)
        df_stat = n - k - 2
        t_stat = partial_r * np.sqrt(df_stat / (1 - partial_r**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_stat))
        
        return {
            'partial_correlation': partial_r,
            'simple_correlation': r_12,
            'controlled_for': available_controls,
            't_statistic': t_stat,
            'p_value': p_value,
            'df': df_stat,
            'is_significant': p_value < 0.05
        }
    except np.linalg.LinAlgError:
        return {'error': 'Singular matrix - cannot compute partial correlation'}
