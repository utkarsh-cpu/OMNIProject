"""
Model Evaluation Module

This module provides functions for evaluating time series forecasting models:
- Forecast accuracy metrics (MAE, RMSE, MAPE)
- Residual analysis
- Ljung-Box test for autocorrelation
- Model comparison and summary generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import json
from datetime import datetime


def compute_forecast_metrics(actual: pd.Series,
                              forecast: pd.Series,
                              train_actual: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Compute forecast accuracy metrics.
    
    Parameters
    ----------
    actual : pd.Series
        Actual observed values
    forecast : pd.Series
        Forecasted values
    train_actual : pd.Series, optional
        Training data (for scaled metrics like MASE)
    
    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy metrics
    
    Metrics
    -------
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error (skips zeros)
    - sMAPE: Symmetric MAPE
    - MASE: Mean Absolute Scaled Error (if train provided)
    - MSE: Mean Squared Error
    - ME: Mean Error (bias)
    - MPE: Mean Percentage Error
    
    Example
    -------
    >>> metrics = compute_forecast_metrics(test_kp, forecast_kp, train_kp)
    >>> print(f"RMSE: {metrics['RMSE']:.4f}")
    """
    # Align and clean
    aligned = pd.DataFrame({
        'actual': actual,
        'forecast': forecast
    }).dropna()
    
    y = aligned['actual'].values
    yhat = aligned['forecast'].values
    
    n = len(y)
    
    if n == 0:
        return {'error': 'No aligned data points'}
    
    # Basic errors
    errors = y - yhat
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    metrics = {}
    
    # MAE - Mean Absolute Error
    metrics['MAE'] = np.mean(abs_errors)
    
    # MSE - Mean Squared Error
    metrics['MSE'] = np.mean(squared_errors)
    
    # RMSE - Root Mean Squared Error
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    
    # ME - Mean Error (measures bias)
    metrics['ME'] = np.mean(errors)
    
    # MAPE - Mean Absolute Percentage Error (exclude zeros)
    non_zero_mask = y != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs(errors[non_zero_mask] / y[non_zero_mask])) * 100
        metrics['MAPE'] = mape
    else:
        metrics['MAPE'] = np.nan
    
    # sMAPE - Symmetric Mean Absolute Percentage Error
    denominator = (np.abs(y) + np.abs(yhat)) / 2
    non_zero_denom = denominator != 0
    if np.any(non_zero_denom):
        smape = np.mean(np.abs(errors[non_zero_denom]) / denominator[non_zero_denom]) * 100
        metrics['sMAPE'] = smape
    else:
        metrics['sMAPE'] = np.nan
    
    # MPE - Mean Percentage Error
    if np.any(non_zero_mask):
        mpe = np.mean(errors[non_zero_mask] / y[non_zero_mask]) * 100
        metrics['MPE'] = mpe
    else:
        metrics['MPE'] = np.nan
    
    # MASE - Mean Absolute Scaled Error (requires training data)
    if train_actual is not None:
        train_clean = train_actual.dropna().values
        if len(train_clean) > 1:
            # Naive forecast scale (1-step differences)
            naive_errors = np.abs(np.diff(train_clean))
            scale = np.mean(naive_errors)
            if scale > 0:
                mase = np.mean(abs_errors) / scale
                metrics['MASE'] = mase
            else:
                metrics['MASE'] = np.nan
    
    # Additional metrics
    metrics['Max_Error'] = np.max(abs_errors)
    metrics['Min_Error'] = np.min(abs_errors)
    metrics['Std_Error'] = np.std(errors)
    metrics['n_observations'] = n
    
    # R-squared (coefficient of determination)
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot > 0:
        metrics['R2'] = 1 - (ss_res / ss_tot)
    else:
        metrics['R2'] = np.nan
    
    # Theil's U statistic
    actual_changes = np.diff(y)
    forecast_changes = np.diff(yhat)
    if len(actual_changes) > 0:
        theil_num = np.sqrt(np.mean((actual_changes - forecast_changes) ** 2))
        theil_denom = np.sqrt(np.mean(actual_changes ** 2)) + np.sqrt(np.mean(forecast_changes ** 2))
        if theil_denom > 0:
            metrics['Theil_U'] = theil_num / theil_denom
    
    return metrics


def perform_residual_analysis(residuals: pd.Series,
                               model_name: str = "Model") -> Dict:
    """
    Perform comprehensive residual analysis.
    
    Parameters
    ----------
    residuals : pd.Series
        Model residuals
    model_name : str, optional
        Name of the model
    
    Returns
    -------
    Dict
        Dictionary with residual analysis results
    
    Example
    -------
    >>> residual_analysis = perform_residual_analysis(model.resid, 'ARIMA')
    """
    residuals_clean = residuals.dropna()
    n = len(residuals_clean)
    
    if n < 10:
        return {'error': 'Insufficient residuals for analysis'}
    
    results = {
        'model_name': model_name,
        'n_residuals': n
    }
    
    # Basic statistics
    results['mean'] = residuals_clean.mean()
    results['std'] = residuals_clean.std()
    results['median'] = residuals_clean.median()
    results['skewness'] = stats.skew(residuals_clean)
    results['kurtosis'] = stats.kurtosis(residuals_clean)
    
    # Test for zero mean (t-test)
    t_stat, t_pvalue = stats.ttest_1samp(residuals_clean, 0)
    results['mean_test'] = {
        't_statistic': t_stat,
        'p_value': t_pvalue,
        'zero_mean': t_pvalue > 0.05
    }
    
    # Normality tests
    # Shapiro-Wilk (max 5000 samples)
    if n <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuals_clean)
        results['shapiro_wilk'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05
        }
    
    # D'Agostino-Pearson (requires n >= 20)
    if n >= 20:
        dagostino_stat, dagostino_p = stats.normaltest(residuals_clean)
        results['dagostino_pearson'] = {
            'statistic': dagostino_stat,
            'p_value': dagostino_p,
            'is_normal': dagostino_p > 0.05
        }
    
    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(residuals_clean)
    results['jarque_bera'] = {
        'statistic': jb_stat,
        'p_value': jb_p,
        'is_normal': jb_p > 0.05
    }
    
    # Durbin-Watson test for autocorrelation
    dw_stat = durbin_watson(residuals_clean)
    results['durbin_watson'] = {
        'statistic': dw_stat,
        'interpretation': 'no autocorrelation' if 1.5 < dw_stat < 2.5 
                         else 'positive autocorrelation' if dw_stat < 1.5
                         else 'negative autocorrelation'
    }
    
    # Ljung-Box test
    lb_results = ljung_box_test(residuals_clean)
    results['ljung_box'] = lb_results
    
    # Overall assessment
    results['white_noise'] = (
        results['mean_test']['zero_mean'] and
        results['ljung_box'].get('is_white_noise', False)
    )
    
    return results


def ljung_box_test(residuals: pd.Series,
                    lags: List[int] = [10, 20, 30]) -> Dict:
    """
    Perform Ljung-Box test for autocorrelation in residuals.
    
    H0: The residuals are independently distributed (white noise)
    H1: The residuals exhibit serial correlation
    
    Parameters
    ----------
    residuals : pd.Series
        Model residuals
    lags : List[int], optional
        Lag values to test (default: [10, 20, 30])
    
    Returns
    -------
    Dict
        Ljung-Box test results
    
    Example
    -------
    >>> lb_test = ljung_box_test(model.resid)
    >>> print(f"White noise: {lb_test['is_white_noise']}")
    """
    residuals_clean = residuals.dropna()
    n = len(residuals_clean)
    
    # Filter lags that are valid (less than n/2)
    valid_lags = [l for l in lags if l < n // 2]
    
    if not valid_lags:
        return {'error': 'Insufficient data for Ljung-Box test'}
    
    # Perform test
    lb_result = acorr_ljungbox(residuals_clean, lags=valid_lags, return_df=True)
    
    results = {
        'lags_tested': valid_lags,
        'statistics': lb_result['lb_stat'].tolist(),
        'p_values': lb_result['lb_pvalue'].tolist()
    }
    
    # Overall assessment (white noise if all p-values > 0.05)
    results['is_white_noise'] = all(p > 0.05 for p in results['p_values'])
    results['min_p_value'] = min(results['p_values'])
    
    # Create summary table
    results['summary'] = {
        lag: {'statistic': stat, 'p_value': pval, 'significant': pval < 0.05}
        for lag, stat, pval in zip(valid_lags, results['statistics'], results['p_values'])
    }
    
    return results


def evaluate_models(model_results: Dict[str, Any],
                    actual: pd.Series,
                    train: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Evaluate and compare multiple models.
    
    Parameters
    ----------
    model_results : Dict[str, Any]
        Dictionary mapping model names to ForecastResult objects or forecasts
    actual : pd.Series
        Actual test values
    train : pd.Series, optional
        Training data for MASE calculation
    
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each model
    
    Example
    -------
    >>> results = {'ARIMA': arima_result, 'ETS': ets_result}
    >>> comparison = evaluate_models(results, test_kp, train_kp)
    """
    metrics_list = []
    
    for model_name, result in model_results.items():
        # Extract forecast (handle both ForecastResult objects and raw Series)
        if hasattr(result, 'forecast'):
            forecast = result.forecast
        else:
            forecast = result
        
        # Compute metrics
        metrics = compute_forecast_metrics(actual, forecast, train)
        metrics['model'] = model_name
        
        # Add model-specific info if available
        if hasattr(result, 'parameters'):
            if 'aic' in result.parameters:
                metrics['AIC'] = result.parameters['aic']
            if 'bic' in result.parameters:
                metrics['BIC'] = result.parameters['bic']
        
        metrics_list.append(metrics)
    
    comparison_df = pd.DataFrame(metrics_list)
    
    # Reorder columns
    first_cols = ['model', 'MAE', 'RMSE', 'MAPE', 'sMAPE', 'R2']
    other_cols = [c for c in comparison_df.columns if c not in first_cols]
    comparison_df = comparison_df[[c for c in first_cols if c in comparison_df.columns] + other_cols]
    
    # Sort by RMSE
    comparison_df = comparison_df.sort_values('RMSE')
    
    # Add rankings
    for metric in ['MAE', 'RMSE', 'MAPE']:
        if metric in comparison_df.columns:
            comparison_df[f'{metric}_rank'] = comparison_df[metric].rank()
    
    return comparison_df


def generate_summary(model_comparison: pd.DataFrame,
                     correlation_summary: Optional[pd.DataFrame] = None,
                     stationarity_results: Optional[List[Dict]] = None,
                     output_path: Optional[str] = None) -> Dict:
    """
    Generate a comprehensive results summary.
    
    Parameters
    ----------
    model_comparison : pd.DataFrame
        Model comparison table from evaluate_models()
    correlation_summary : pd.DataFrame, optional
        Correlation analysis summary
    stationarity_results : List[Dict], optional
        Stationarity test results
    output_path : str, optional
        Path to save summary as JSON
    
    Returns
    -------
    Dict
        Comprehensive summary dictionary
    
    Example
    -------
    >>> summary = generate_summary(model_comparison, correlation_summary)
    >>> print(summary['best_model'])
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'Solar Wind Time Series Forecasting'
    }
    
    # Model Performance Summary
    if model_comparison is not None and len(model_comparison) > 0:
        best_model_idx = model_comparison['RMSE'].idxmin()
        best_model = model_comparison.loc[best_model_idx]
        
        summary['model_performance'] = {
            'n_models_compared': len(model_comparison),
            'best_model': {
                'name': best_model['model'],
                'MAE': float(best_model['MAE']),
                'RMSE': float(best_model['RMSE']),
                'MAPE': float(best_model.get('MAPE', np.nan)),
                'R2': float(best_model.get('R2', np.nan))
            },
            'all_models': model_comparison.to_dict(orient='records')
        }
    
    # Correlation Summary
    if correlation_summary is not None and len(correlation_summary) > 0:
        # Find top predictors
        if 'abs_correlation' in correlation_summary.columns:
            sort_col = 'abs_correlation'
        elif 'max_ccf' in correlation_summary.columns:
            sort_col = correlation_summary['max_ccf'].abs()
            correlation_summary['sort_val'] = sort_col
            sort_col = 'sort_val'
        else:
            sort_col = 'pearson_r' if 'pearson_r' in correlation_summary.columns else None
        
        if sort_col:
            top_predictors = correlation_summary.nlargest(5, sort_col)
            summary['correlation_analysis'] = {
                'top_predictors': top_predictors.to_dict(orient='records'),
                'total_pairs_analyzed': len(correlation_summary)
            }
    
    # Stationarity Summary
    if stationarity_results is not None:
        summary['stationarity'] = {
            'variables_tested': len(stationarity_results),
            'results': stationarity_results
        }
    
    # Key Findings
    findings = []
    
    if 'model_performance' in summary:
        best = summary['model_performance']['best_model']
        findings.append(
            f"Best performing model: {best['name']} with RMSE={best['RMSE']:.4f}"
        )
    
    if 'correlation_analysis' in summary:
        top = summary['correlation_analysis']['top_predictors'][0] if summary['correlation_analysis']['top_predictors'] else None
        if top:
            predictor = top.get('predictor', 'Unknown')
            corr = top.get('max_ccf', top.get('pearson_r', top.get('correlation', 0)))
            findings.append(
                f"Strongest predictor: {predictor} with correlation={corr:.4f}"
            )
    
    summary['key_findings'] = findings
    
    # Save if path provided
    if output_path:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj
        
        summary_json = convert_types(summary)
        
        with open(output_path, 'w') as f:
            json.dump(summary_json, f, indent=2)
        print(f"Summary saved to {output_path}")
    
    return summary


def print_evaluation_report(metrics: Dict[str, float],
                            model_name: str = "Model",
                            residual_analysis: Optional[Dict] = None):
    """
    Print a formatted evaluation report.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Forecast metrics from compute_forecast_metrics()
    model_name : str, optional
        Name of the model
    residual_analysis : Dict, optional
        Residual analysis from perform_residual_analysis()
    """
    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {model_name}")
    print("=" * 60)
    
    print("\nForecast Accuracy Metrics:")
    print("-" * 40)
    print(f"  MAE  (Mean Absolute Error):        {metrics.get('MAE', 'N/A'):.4f}")
    print(f"  RMSE (Root Mean Squared Error):    {metrics.get('RMSE', 'N/A'):.4f}")
    print(f"  MAPE (Mean Absolute % Error):      {metrics.get('MAPE', 'N/A'):.2f}%")
    print(f"  sMAPE (Symmetric MAPE):            {metrics.get('sMAPE', 'N/A'):.2f}%")
    if 'MASE' in metrics:
        print(f"  MASE (Mean Absolute Scaled Error): {metrics['MASE']:.4f}")
    print(f"  R² (Coefficient of Determination): {metrics.get('R2', 'N/A'):.4f}")
    
    print("\nError Statistics:")
    print("-" * 40)
    print(f"  Mean Error (Bias):                 {metrics.get('ME', 'N/A'):.4f}")
    print(f"  Std Error:                         {metrics.get('Std_Error', 'N/A'):.4f}")
    print(f"  Max Error:                         {metrics.get('Max_Error', 'N/A'):.4f}")
    
    if residual_analysis is not None and 'error' not in residual_analysis:
        print("\nResidual Diagnostics:")
        print("-" * 40)
        print(f"  Mean of residuals:                 {residual_analysis['mean']:.6f}")
        print(f"  Std of residuals:                  {residual_analysis['std']:.4f}")
        
        if 'mean_test' in residual_analysis:
            mt = residual_analysis['mean_test']
            print(f"  Zero mean test p-value:            {mt['p_value']:.4f} ({'✓' if mt['zero_mean'] else '✗'})")
        
        if 'jarque_bera' in residual_analysis:
            jb = residual_analysis['jarque_bera']
            print(f"  Jarque-Bera normality p-value:     {jb['p_value']:.4f} ({'✓' if jb['is_normal'] else '✗'})")
        
        if 'ljung_box' in residual_analysis:
            lb = residual_analysis['ljung_box']
            print(f"  Ljung-Box test (white noise):      {'✓ Yes' if lb.get('is_white_noise', False) else '✗ No'}")
        
        wn = residual_analysis.get('white_noise', False)
        print(f"\n  Overall: Residuals are {'✓ white noise' if wn else '✗ not white noise'}")
    
    print("\n" + "=" * 60)


def create_forecast_summary_table(model_results: Dict[str, Any],
                                   actual: pd.Series) -> pd.DataFrame:
    """
    Create a summary table comparing forecasts from multiple models.
    
    Parameters
    ----------
    model_results : Dict[str, Any]
        Dictionary of model results
    actual : pd.Series
        Actual values
    
    Returns
    -------
    pd.DataFrame
        Table with actual vs forecasted values for each model
    """
    data = {'actual': actual}
    
    for model_name, result in model_results.items():
        if hasattr(result, 'forecast'):
            data[f'{model_name}_forecast'] = result.forecast
            data[f'{model_name}_error'] = actual - result.forecast
        else:
            data[f'{model_name}_forecast'] = result
            data[f'{model_name}_error'] = actual - result
    
    df = pd.DataFrame(data)
    
    return df


def compute_directional_accuracy(actual: pd.Series,
                                  forecast: pd.Series) -> Dict[str, float]:
    """
    Compute directional accuracy metrics.
    
    Parameters
    ----------
    actual : pd.Series
        Actual values
    forecast : pd.Series
        Forecasted values
    
    Returns
    -------
    Dict[str, float]
        Directional accuracy metrics
    """
    # Align
    aligned = pd.DataFrame({'actual': actual, 'forecast': forecast}).dropna()
    
    if len(aligned) < 2:
        return {'error': 'Insufficient data'}
    
    # Actual direction of change
    actual_direction = np.sign(aligned['actual'].diff())
    forecast_direction = np.sign(aligned['forecast'].diff())
    
    # Remove first NaN
    actual_direction = actual_direction.iloc[1:]
    forecast_direction = forecast_direction.iloc[1:]
    
    # Directional accuracy
    n = len(actual_direction)
    correct_direction = (actual_direction == forecast_direction).sum()
    
    metrics = {
        'directional_accuracy': correct_direction / n * 100,
        'correct_directions': int(correct_direction),
        'total_predictions': n
    }
    
    # Hit rate for up/down movements
    up_actual = actual_direction == 1
    up_forecast = forecast_direction == 1
    
    if up_actual.sum() > 0:
        metrics['up_hit_rate'] = ((up_actual & up_forecast).sum() / up_actual.sum()) * 100
    
    down_actual = actual_direction == -1
    down_forecast = forecast_direction == -1
    
    if down_actual.sum() > 0:
        metrics['down_hit_rate'] = ((down_actual & down_forecast).sum() / down_actual.sum()) * 100
    
    return metrics
