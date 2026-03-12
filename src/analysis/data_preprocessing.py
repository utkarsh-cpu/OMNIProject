"""
Data Preprocessing Module for Solar Wind and Geomagnetic Time Series Analysis

This module provides functions to load, clean, and prepare OMNI2 solar wind data
for time series analysis including handling missing values, resampling, and normalization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

# Fill values used in OMNI2 dataset to represent missing data
FILL_VALUES = {
    'B_mag_avg': 999.9,
    'B_vec_mag': 999.9,
    'B_lat': 999.9,
    'B_long': 999.9,
    'Bx_GSE': 999.9,
    'By_GSE': 999.9,
    'Bz_GSE': 999.9,
    'By_GSM': 999.9,
    'Bz_GSM': 999.9,
    'sigma_B_mag': 999.9,
    'sigma_B': 999.9,
    'sigma_Bx': 999.9,
    'sigma_By': 999.9,
    'sigma_Bz': 999.9,
    'proton_temp': 9999999.0,
    'proton_density': 999.9,
    'plasma_speed': 9999.0,
    'plasma_long_angle': 999.9,
    'plasma_lat_angle': 999.9,
    'alpha_proton_ratio': 9.999,
    'flow_pressure': 99.99,
    'sigma_T': 9999999.0,
    'sigma_N': 999.9,
    'sigma_V': 9999.0,
    'sigma_phi_V': 999.9,
    'sigma_theta_V': 999.9,
    'sigma_alpha_ratio': 9.999,
    'electric_field': 999.99,
    'plasma_beta': 999.99,
    'alfven_mach': 999.9,
    'Kp': 99,
    'sunspot_number': 999,
    'DST': 99999,
    'AE': 9999,
    'proton_flux_gt1': 999999.99,
    'proton_flux_gt2': 99999.99,
    'proton_flux_gt4': 99999.99,
    'proton_flux_gt10': 99999.99,
    'proton_flux_gt30': 99999.99,
    'proton_flux_gt60': 99999.99,
    'ap_index': 999,
    'f107_index': 999.9,
    'PCN_index': 999.9,
    'AL_index': 99999,
    'AU_index': 99999,
    'magnetosonic_mach': 99.9,
    'bartels_rotation': 9999,
    'imf_spacecraft_id': 99,
    'plasma_spacecraft_id': 99,
    'num_imf_points': 999,
    'num_plasma_points': 999,
}

# Key variables for storm analysis
STORM_INDICES = ['Kp', 'DST', 'AE', 'ap_index', 'AL_index', 'AU_index', 'PCN_index']
SOLAR_WIND_PARAMS = ['Bz_GSE', 'Bz_GSM', 'B_mag_avg', 'plasma_speed', 'proton_density', 
                     'proton_temp', 'flow_pressure', 'electric_field', 'plasma_beta', 
                     'alfven_mach', 'By_GSM', 'Bx_GSE']


def load_data(filepath: str, 
              parse_dates: bool = True,
              date_column: str = 'datetime') -> pd.DataFrame:
    """
    Load OMNI2 time series data from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing OMNI2 data
    parse_dates : bool, optional
        Whether to parse the datetime column (default: True)
    date_column : str, optional
        Name of the datetime column (default: 'datetime')
    
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with datetime index if parse_dates is True
    
    Example
    -------
    >>> df = load_data('omni2_full_dataset.csv')
    >>> print(df.shape)
    (561024, 55)
    """
    print(f"Loading data from {filepath}...")
    
    df = pd.read_csv(filepath)
    
    if parse_dates and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        df.index.name = 'datetime'
    
    print(f"Loaded {len(df)} records spanning {df.index.min()} to {df.index.max()}")
    print(f"Columns: {len(df.columns)}")
    
    return df


def clean_data(df: pd.DataFrame,
               columns: Optional[List[str]] = None,
               fill_method: str = 'interpolate',
               max_gap: int = 24) -> pd.DataFrame:
    """
    Clean the OMNI2 dataset by handling missing values.
    
    Missing values in OMNI2 are represented by specific fill values (e.g., 999.9).
    This function replaces those with NaN and then applies interpolation or other methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with OMNI2 data
    columns : List[str], optional
        Specific columns to clean. If None, cleans all columns with known fill values
    fill_method : str, optional
        Method to fill missing values: 'interpolate', 'forward', 'backward', or 'drop'
        (default: 'interpolate')
    max_gap : int, optional
        Maximum number of consecutive missing values to fill (default: 24 hours)
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with missing values handled
    
    Example
    -------
    >>> df_clean = clean_data(df, columns=['Kp', 'DST', 'Bz_GSM'])
    """
    df_clean = df.copy()
    
    # Determine columns to process
    if columns is None:
        columns = [col for col in df.columns if col in FILL_VALUES]
    else:
        columns = [col for col in columns if col in df.columns]
    
    print(f"Cleaning {len(columns)} columns...")
    
    # Replace fill values with NaN
    for col in columns:
        if col in FILL_VALUES:
            fill_val = FILL_VALUES[col]
            # Handle both exact matches and values close to fill value
            mask = np.isclose(df_clean[col].values, fill_val, rtol=0.001, atol=0.001)
            df_clean.loc[mask, col] = np.nan
    
    # Report missing data
    missing_report = df_clean[columns].isnull().sum()
    missing_pct = (missing_report / len(df_clean) * 100).round(2)
    print("\nMissing data after replacing fill values:")
    for col, pct in missing_pct.items():
        if pct > 0:
            print(f"  {col}: {pct}%")
    
    # Fill missing values based on method
    if fill_method == 'interpolate':
        for col in columns:
            df_clean[col] = df_clean[col].interpolate(method='time', limit=max_gap)
    elif fill_method == 'forward':
        df_clean[columns] = df_clean[columns].fillna(method='ffill', limit=max_gap)
    elif fill_method == 'backward':
        df_clean[columns] = df_clean[columns].fillna(method='bfill', limit=max_gap)
    elif fill_method == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    
    print(f"\nCleaned data shape: {df_clean.shape}")
    
    return df_clean


def detect_storms(df: pd.DataFrame,
                  dst_threshold: float = -50,
                  kp_threshold: float = 50,
                  ae_threshold: float = 500) -> pd.DataFrame:
    """
    Detect geomagnetic storms based on index thresholds.
    
    Storm classification:
    - Weak storm: -50 <= DST < -30 or Kp >= 5
    - Moderate storm: -100 <= DST < -50 or Kp >= 6
    - Intense storm: -200 <= DST < -100 or Kp >= 7
    - Severe storm: DST < -200 or Kp >= 8
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with geomagnetic indices
    dst_threshold : float, optional
        DST threshold for storm detection (default: -50 nT)
    kp_threshold : float, optional
        Kp threshold for storm detection (default: 50, i.e., Kp=5)
    ae_threshold : float, optional
        AE threshold for substorm detection (default: 500 nT)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with storm flags and classifications added
    
    Example
    -------
    >>> df_storms = detect_storms(df)
    >>> print(df_storms['storm_category'].value_counts())
    """
    df_storms = df.copy()
    
    # Initialize storm flag columns
    df_storms['is_storm'] = False
    df_storms['storm_category'] = 'quiet'
    df_storms['is_substorm'] = False
    
    # Detect storms based on DST
    if 'DST' in df_storms.columns:
        # Remove fill values
        dst = df_storms['DST'].replace(99999, np.nan)
        
        df_storms.loc[dst < -30, 'is_storm'] = True
        df_storms.loc[dst < -30, 'storm_category'] = 'weak'
        df_storms.loc[dst < -50, 'storm_category'] = 'moderate'
        df_storms.loc[dst < -100, 'storm_category'] = 'intense'
        df_storms.loc[dst < -200, 'storm_category'] = 'severe'
    
    # Detect based on Kp (Kp values are *10, so Kp=5 is stored as 50)
    if 'Kp' in df_storms.columns:
        kp = df_storms['Kp'].replace(99, np.nan)
        
        df_storms.loc[(~df_storms['is_storm']) & (kp >= 50), 'is_storm'] = True
        df_storms.loc[(kp >= 50) & (df_storms['storm_category'] == 'quiet'), 'storm_category'] = 'weak'
        df_storms.loc[kp >= 60, 'storm_category'] = np.where(
            df_storms.loc[kp >= 60, 'storm_category'].isin(['quiet', 'weak']),
            'moderate', df_storms.loc[kp >= 60, 'storm_category'])
        df_storms.loc[kp >= 70, 'storm_category'] = np.where(
            df_storms.loc[kp >= 70, 'storm_category'].isin(['quiet', 'weak', 'moderate']),
            'intense', df_storms.loc[kp >= 70, 'storm_category'])
        df_storms.loc[kp >= 80, 'storm_category'] = 'severe'
    
    # Detect substorms based on AE
    if 'AE' in df_storms.columns:
        ae = df_storms['AE'].replace(9999, np.nan)
        df_storms.loc[ae >= ae_threshold, 'is_substorm'] = True
    
    # Print summary
    storm_counts = df_storms['storm_category'].value_counts()
    print("\nStorm Detection Summary:")
    print(f"  Total hours: {len(df_storms)}")
    print(f"  Storm hours: {df_storms['is_storm'].sum()} ({df_storms['is_storm'].mean()*100:.1f}%)")
    print(f"  Substorm hours: {df_storms['is_substorm'].sum()}")
    print("\nStorm categories:")
    for cat, count in storm_counts.items():
        print(f"  {cat}: {count} ({count/len(df_storms)*100:.2f}%)")
    
    return df_storms


def get_statistical_properties(df: pd.DataFrame,
                                columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute statistical properties of the time series data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : List[str], optional
        Columns to analyze. If None, uses STORM_INDICES + SOLAR_WIND_PARAMS
    
    Returns
    -------
    pd.DataFrame
        DataFrame with statistical properties (mean, std, min, max, skewness, kurtosis)
    
    Example
    -------
    >>> stats = get_statistical_properties(df)
    >>> print(stats)
    """
    if columns is None:
        columns = [col for col in STORM_INDICES + SOLAR_WIND_PARAMS if col in df.columns]
    else:
        columns = [col for col in columns if col in df.columns]
    
    stats_dict = {
        'count': [],
        'mean': [],
        'std': [],
        'min': [],
        '25%': [],
        '50%': [],
        '75%': [],
        'max': [],
        'skewness': [],
        'kurtosis': [],
        'missing_pct': []
    }
    
    for col in columns:
        series = df[col].dropna()
        stats_dict['count'].append(len(series))
        stats_dict['mean'].append(series.mean())
        stats_dict['std'].append(series.std())
        stats_dict['min'].append(series.min())
        stats_dict['25%'].append(series.quantile(0.25))
        stats_dict['50%'].append(series.quantile(0.50))
        stats_dict['75%'].append(series.quantile(0.75))
        stats_dict['max'].append(series.max())
        stats_dict['skewness'].append(stats.skew(series))
        stats_dict['kurtosis'].append(stats.kurtosis(series))
        stats_dict['missing_pct'].append((df[col].isna().sum() / len(df)) * 100)
    
    stats_df = pd.DataFrame(stats_dict, index=columns)
    
    return stats_df


def perform_stationarity_tests(series: pd.Series,
                                name: str = 'series') -> Dict[str, any]:
    """
    Perform stationarity tests on a time series.
    
    Performs:
    - Augmented Dickey-Fuller (ADF) test
    - KPSS test
    
    Parameters
    ----------
    series : pd.Series
        Time series to test
    name : str, optional
        Name of the series for reporting
    
    Returns
    -------
    Dict[str, any]
        Dictionary containing test results:
        - adf_statistic: ADF test statistic
        - adf_pvalue: ADF p-value
        - adf_critical_values: Critical values at 1%, 5%, 10%
        - kpss_statistic: KPSS test statistic
        - kpss_pvalue: KPSS p-value
        - is_stationary: Boolean indicating likely stationarity
    
    Example
    -------
    >>> results = perform_stationarity_tests(df['Kp'], 'Kp')
    >>> print(f"Kp is stationary: {results['is_stationary']}")
    """
    # Remove NaN values
    series_clean = series.dropna()
    
    if len(series_clean) < 50:
        warnings.warn(f"Series {name} has fewer than 50 observations. Results may be unreliable.")
    
    results = {'name': name}
    
    # ADF Test (null hypothesis: series has a unit root, i.e., non-stationary)
    try:
        adf_result = adfuller(series_clean, autolag='AIC')
        results['adf_statistic'] = adf_result[0]
        results['adf_pvalue'] = adf_result[1]
        results['adf_critical_values'] = adf_result[4]
        results['adf_is_stationary'] = adf_result[1] < 0.05
    except Exception as e:
        results['adf_error'] = str(e)
        results['adf_is_stationary'] = None
    
    # KPSS Test (null hypothesis: series is stationary)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(series_clean, regression='c', nlags='auto')
        results['kpss_statistic'] = kpss_result[0]
        results['kpss_pvalue'] = kpss_result[1]
        results['kpss_critical_values'] = kpss_result[3]
        results['kpss_is_stationary'] = kpss_result[1] > 0.05
    except Exception as e:
        results['kpss_error'] = str(e)
        results['kpss_is_stationary'] = None
    
    # Combined interpretation
    if results.get('adf_is_stationary') is not None and results.get('kpss_is_stationary') is not None:
        # Stationary if ADF rejects null AND KPSS fails to reject null
        results['is_stationary'] = results['adf_is_stationary'] and results['kpss_is_stationary']
    else:
        results['is_stationary'] = None
    
    # Print summary
    print(f"\nStationarity Tests for {name}:")
    print(f"  ADF Test:")
    if 'adf_statistic' in results:
        print(f"    Statistic: {results['adf_statistic']:.4f}")
        print(f"    p-value: {results['adf_pvalue']:.4f}")
        print(f"    Stationary (p<0.05): {results['adf_is_stationary']}")
    print(f"  KPSS Test:")
    if 'kpss_statistic' in results:
        print(f"    Statistic: {results['kpss_statistic']:.4f}")
        print(f"    p-value: {results['kpss_pvalue']:.4f}")
        print(f"    Stationary (p>0.05): {results['kpss_is_stationary']}")
    print(f"  Overall Stationary: {results['is_stationary']}")
    
    return results


def resample_data(df: pd.DataFrame,
                  freq: str = 'D',
                  agg_method: str = 'mean',
                  columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Resample time series data to a different frequency.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index
    freq : str, optional
        Target frequency: 'H' (hourly), 'D' (daily), 'W' (weekly), 'M' (monthly)
        (default: 'D')
    agg_method : str, optional
        Aggregation method: 'mean', 'max', 'min', 'sum' (default: 'mean')
    columns : List[str], optional
        Columns to resample. If None, resamples all numeric columns
    
    Returns
    -------
    pd.DataFrame
        Resampled DataFrame
    
    Example
    -------
    >>> df_daily = resample_data(df, freq='D', agg_method='mean')
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        columns = [col for col in columns if col in df.columns]
    
    df_subset = df[columns]
    
    if agg_method == 'mean':
        df_resampled = df_subset.resample(freq).mean()
    elif agg_method == 'max':
        df_resampled = df_subset.resample(freq).max()
    elif agg_method == 'min':
        df_resampled = df_subset.resample(freq).min()
    elif agg_method == 'sum':
        df_resampled = df_subset.resample(freq).sum()
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")
    
    print(f"Resampled from {len(df)} to {len(df_resampled)} records ({freq} frequency)")
    
    return df_resampled


def normalize_data(df: pd.DataFrame,
                   columns: Optional[List[str]] = None,
                   method: str = 'zscore') -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Normalize time series data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : List[str], optional
        Columns to normalize. If None, normalizes all numeric columns
    method : str, optional
        Normalization method: 'zscore', 'minmax', 'robust' (default: 'zscore')
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        - Normalized DataFrame
        - Dictionary with normalization parameters for inverse transformation
    
    Example
    -------
    >>> df_norm, params = normalize_data(df, columns=['Kp', 'DST'], method='zscore')
    """
    df_norm = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        columns = [col for col in columns if col in df.columns]
    
    params = {}
    
    for col in columns:
        series = df[col]
        
        if method == 'zscore':
            mean_val = series.mean()
            std_val = series.std()
            df_norm[col] = (series - mean_val) / std_val
            params[col] = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
            
        elif method == 'minmax':
            min_val = series.min()
            max_val = series.max()
            df_norm[col] = (series - min_val) / (max_val - min_val)
            params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
            
        elif method == 'robust':
            median_val = series.median()
            iqr_val = series.quantile(0.75) - series.quantile(0.25)
            df_norm[col] = (series - median_val) / iqr_val
            params[col] = {'method': 'robust', 'median': median_val, 'iqr': iqr_val}
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return df_norm, params


def denormalize_data(series: pd.Series,
                     params: Dict) -> pd.Series:
    """
    Reverse normalization using stored parameters.
    
    Parameters
    ----------
    series : pd.Series
        Normalized series
    params : Dict
        Normalization parameters from normalize_data()
    
    Returns
    -------
    pd.Series
        Denormalized series
    """
    method = params['method']
    
    if method == 'zscore':
        return series * params['std'] + params['mean']
    elif method == 'minmax':
        return series * (params['max'] - params['min']) + params['min']
    elif method == 'robust':
        return series * params['iqr'] + params['median']
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def prepare_analysis_subset(df: pd.DataFrame,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            target_variables: Optional[List[str]] = None,
                            predictor_variables: Optional[List[str]] = None,
                            resample_freq: Optional[str] = None) -> pd.DataFrame:
    """
    Prepare a subset of data for analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    start_date : str, optional
        Start date for subset (e.g., '2010-01-01')
    end_date : str, optional
        End date for subset (e.g., '2020-12-31')
    target_variables : List[str], optional
        Target variables (storm indices)
    predictor_variables : List[str], optional
        Predictor variables (solar wind parameters)
    resample_freq : str, optional
        Resample frequency if needed
    
    Returns
    -------
    pd.DataFrame
        Prepared subset ready for analysis
    """
    # Default variables
    if target_variables is None:
        target_variables = ['Kp', 'DST', 'AE']
    if predictor_variables is None:
        predictor_variables = ['Bz_GSM', 'plasma_speed', 'proton_density', 'flow_pressure']
    
    # Select columns
    all_vars = target_variables + predictor_variables
    available_vars = [v for v in all_vars if v in df.columns]
    df_subset = df[available_vars].copy()
    
    # Filter by date
    if start_date is not None:
        df_subset = df_subset[df_subset.index >= start_date]
    if end_date is not None:
        df_subset = df_subset[df_subset.index <= end_date]
    
    # Resample if requested
    if resample_freq is not None:
        df_subset = resample_data(df_subset, freq=resample_freq)
    
    # Clean data
    df_subset = clean_data(df_subset, columns=available_vars)
    
    print(f"\nPrepared analysis subset:")
    print(f"  Shape: {df_subset.shape}")
    print(f"  Date range: {df_subset.index.min()} to {df_subset.index.max()}")
    print(f"  Variables: {available_vars}")
    
    return df_subset
