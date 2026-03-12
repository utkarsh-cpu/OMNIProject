"""
Multivariate Time Series Modeling Module

This module provides functions for multivariate time series analysis including:
- ARIMAX (ARIMA with Exogenous variables)
- SARIMAX (Seasonal ARIMAX)
- VAR (Vector AutoRegression)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import pmdarima as pm


@dataclass
class MultivariateResult:
    """Container for multivariate model results."""
    model_name: str
    forecast: Union[pd.Series, pd.DataFrame]
    confidence_intervals: Optional[pd.DataFrame]
    fitted_values: Union[pd.Series, pd.DataFrame]
    residuals: Union[pd.Series, pd.DataFrame]
    model_object: any
    parameters: Dict
    exog_variables: Optional[List[str]]


def prepare_multivariate_data(df: pd.DataFrame,
                               target: str,
                               exog: List[str],
                               dropna: bool = True) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Prepare data for multivariate modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target : str
        Target variable name
    exog : List[str]
        List of exogenous variable names
    dropna : bool, optional
        Whether to drop missing values (default: True)
    
    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        Target series and exogenous DataFrame
    """
    # Filter available columns
    available_exog = [col for col in exog if col in df.columns]
    
    if len(available_exog) == 0:
        raise ValueError(f"No exogenous variables found in data. Requested: {exog}")
    
    all_cols = [target] + available_exog
    data = df[all_cols].copy()
    
    if dropna:
        data = data.dropna()
    
    y = data[target]
    X = data[available_exog]
    
    print(f"Prepared multivariate data:")
    print(f"  Target: {target} ({len(y)} observations)")
    print(f"  Exogenous: {available_exog}")
    
    return y, X


def fit_arimax(train_y: pd.Series,
               train_X: pd.DataFrame,
               test_y: pd.Series,
               test_X: pd.DataFrame,
               order: Optional[Tuple[int, int, int]] = None,
               auto_select: bool = True,
               max_p: int = 5,
               max_q: int = 5) -> MultivariateResult:
    """
    Fit ARIMAX model (ARIMA with exogenous variables).
    
    Parameters
    ----------
    train_y : pd.Series
        Target variable training data
    train_X : pd.DataFrame
        Exogenous variables training data
    test_y : pd.Series
        Target variable test data (for forecast horizon)
    test_X : pd.DataFrame
        Exogenous variables test data (required for forecasting)
    order : Tuple[int, int, int], optional
        (p, d, q) order. If None and auto_select=True, determined automatically
    auto_select : bool, optional
        If True, use auto_arima with exogenous (default: True)
    max_p : int, optional
        Maximum AR order (default: 5)
    max_q : int, optional
        Maximum MA order (default: 5)
    
    Returns
    -------
    MultivariateResult
        Object containing forecast, residuals, and model
    
    Example
    -------
    >>> result = fit_arimax(train_y, train_X, test_y, test_X)
    """
    print("Fitting ARIMAX model...")
    
    forecast_horizon = len(test_y)
    exog_vars = list(train_X.columns)
    
    if auto_select:
        print("Running auto_arima with exogenous variables...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto_model = pm.auto_arima(
                train_y,
                exogenous=train_X,
                start_p=0, max_p=max_p,
                start_q=0, max_q=max_q,
                d=None, max_d=2,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                information_criterion='aic'
            )
        order = auto_model.order
        print(f"Auto-selected ARIMAX order: {order}")
    else:
        if order is None:
            order = (1, 1, 1)
    
    # Fit SARIMAX model (ARIMAX is SARIMAX without seasonal component)
    model = SARIMAX(
        train_y,
        exog=train_X,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted_model = model.fit(disp=False)
    
    # Generate forecast
    forecast_obj = fitted_model.get_forecast(steps=forecast_horizon, exog=test_X)
    forecast = forecast_obj.predicted_mean
    forecast.index = test_y.index
    
    # Confidence intervals
    conf_int_raw = forecast_obj.conf_int(alpha=0.05)
    conf_int = pd.DataFrame({
        'lower': conf_int_raw.iloc[:, 0].values,
        'upper': conf_int_raw.iloc[:, 1].values
    }, index=test_y.index)
    
    # Fitted values and residuals
    fitted_values = fitted_model.fittedvalues
    residuals = fitted_model.resid
    
    params = {
        'order': order,
        'aic': fitted_model.aic,
        'bic': fitted_model.bic,
        'exog_coefs': dict(zip(exog_vars, fitted_model.params[exog_vars].values))
    }
    
    print(f"ARIMAX Model fitted: order={order}, AIC={fitted_model.aic:.2f}")
    print(f"Exogenous variable coefficients:")
    for var, coef in params['exog_coefs'].items():
        print(f"  {var}: {coef:.4f}")
    
    return MultivariateResult(
        model_name='ARIMAX',
        forecast=forecast,
        confidence_intervals=conf_int,
        fitted_values=fitted_values,
        residuals=residuals,
        model_object=fitted_model,
        parameters=params,
        exog_variables=exog_vars
    )


def fit_sarimax(train_y: pd.Series,
                train_X: pd.DataFrame,
                test_y: pd.Series,
                test_X: pd.DataFrame,
                order: Optional[Tuple[int, int, int]] = None,
                seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                auto_select: bool = True) -> MultivariateResult:
    """
    Fit SARIMAX model (Seasonal ARIMA with exogenous variables).
    
    Parameters
    ----------
    train_y : pd.Series
        Target variable training data
    train_X : pd.DataFrame
        Exogenous variables training data
    test_y : pd.Series
        Target variable test data
    test_X : pd.DataFrame
        Exogenous variables test data
    order : Tuple[int, int, int], optional
        (p, d, q) order
    seasonal_order : Tuple[int, int, int, int], optional
        (P, D, Q, m) seasonal order
    auto_select : bool, optional
        If True, use auto_arima (default: True)
    
    Returns
    -------
    MultivariateResult
        Object containing forecast and model
    
    Example
    -------
    >>> result = fit_sarimax(train_y, train_X, test_y, test_X, 
    ...                      seasonal_order=(1, 1, 1, 24))
    """
    print("Fitting SARIMAX model...")
    
    forecast_horizon = len(test_y)
    exog_vars = list(train_X.columns)
    
    if auto_select:
        print("Running auto_arima with seasonal + exogenous...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto_model = pm.auto_arima(
                train_y,
                exogenous=train_X,
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                d=None, max_d=2,
                seasonal=True,
                m=24,  # Hourly data with daily seasonality
                start_P=0, max_P=2,
                start_Q=0, max_Q=2,
                D=None, max_D=1,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                information_criterion='aic'
            )
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        print(f"Auto-selected: order={order}, seasonal_order={seasonal_order}")
    else:
        if order is None:
            order = (1, 1, 1)
        if seasonal_order is None:
            seasonal_order = (1, 1, 1, 24)
    
    # Fit model
    model = SARIMAX(
        train_y,
        exog=train_X,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted_model = model.fit(disp=False, maxiter=200)
    
    # Generate forecast
    forecast_obj = fitted_model.get_forecast(steps=forecast_horizon, exog=test_X)
    forecast = forecast_obj.predicted_mean
    forecast.index = test_y.index
    
    # Confidence intervals
    conf_int_raw = forecast_obj.conf_int(alpha=0.05)
    conf_int = pd.DataFrame({
        'lower': conf_int_raw.iloc[:, 0].values,
        'upper': conf_int_raw.iloc[:, 1].values
    }, index=test_y.index)
    
    # Fitted values and residuals
    fitted_values = fitted_model.fittedvalues
    residuals = fitted_model.resid
    
    params = {
        'order': order,
        'seasonal_order': seasonal_order,
        'aic': fitted_model.aic,
        'bic': fitted_model.bic
    }
    
    print(f"SARIMAX Model fitted: order={order}, seasonal={seasonal_order}, AIC={fitted_model.aic:.2f}")
    
    return MultivariateResult(
        model_name='SARIMAX',
        forecast=forecast,
        confidence_intervals=conf_int,
        fitted_values=fitted_values,
        residuals=residuals,
        model_object=fitted_model,
        parameters=params,
        exog_variables=exog_vars
    )


def fit_var(train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            variables: List[str],
            target: str,
            maxlags: Optional[int] = None,
            ic: str = 'aic') -> MultivariateResult:
    """
    Fit Vector AutoRegression (VAR) model.
    
    VAR models all variables jointly, capturing their interdependencies.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with all variables
    test_df : pd.DataFrame
        Test data with all variables
    variables : List[str]
        List of variables to include in VAR
    target : str
        Primary target variable for forecast extraction
    maxlags : int, optional
        Maximum lags to consider. If None, selected by information criterion
    ic : str, optional
        Information criterion: 'aic', 'bic', 'hqic', 'fpe' (default: 'aic')
    
    Returns
    -------
    MultivariateResult
        Object containing forecasts and model
    
    Example
    -------
    >>> result = fit_var(train_df, test_df, ['Kp', 'Bz_GSM', 'plasma_speed'], target='Kp')
    """
    print(f"Fitting VAR model for variables: {variables}")
    
    # Filter and clean data
    available_vars = [v for v in variables if v in train_df.columns]
    
    train_data = train_df[available_vars].dropna()
    test_data = test_df[available_vars].dropna()
    
    forecast_horizon = len(test_data)
    
    # Check stationarity and difference if needed
    differenced = False
    train_diff = train_data.copy()
    
    for var in available_vars:
        adf_result = adfuller(train_data[var].dropna())
        if adf_result[1] > 0.05:  # Not stationary
            train_diff[var] = train_data[var].diff()
            differenced = True
            print(f"  {var}: differenced for stationarity")
    
    if differenced:
        train_diff = train_diff.dropna()
    
    # Fit VAR model
    model = VAR(train_diff)
    
    # Select optimal lag order
    if maxlags is None:
        lag_order = model.select_order(maxlags=15)
        optimal_lag = getattr(lag_order, ic)
        print(f"Selected lag order by {ic.upper()}: {optimal_lag}")
    else:
        optimal_lag = maxlags
    
    # Ensure at least lag 1
    optimal_lag = max(1, optimal_lag)
    
    # Fit model with selected lag
    fitted_model = model.fit(optimal_lag)
    
    print(f"VAR({optimal_lag}) model fitted")
    print(f"AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")
    
    # Generate forecasts
    forecast_input = train_diff.values[-optimal_lag:]
    forecast_values = fitted_model.forecast(forecast_input, steps=forecast_horizon)
    
    forecast_df = pd.DataFrame(
        forecast_values,
        columns=available_vars,
        index=test_data.index[:forecast_horizon]
    )
    
    # If we differenced, need to integrate back
    if differenced:
        for i, var in enumerate(available_vars):
            # Add last level from training data
            last_level = train_data[var].iloc[-1]
            forecast_df[var] = forecast_df[var].cumsum() + last_level
    
    # Confidence intervals for forecast
    # VAR forecast_interval method
    try:
        fc, lower, upper = fitted_model.forecast_interval(
            forecast_input, steps=forecast_horizon, alpha=0.05
        )
        target_idx = available_vars.index(target)
        conf_int = pd.DataFrame({
            'lower': lower[:, target_idx],
            'upper': upper[:, target_idx]
        }, index=test_data.index[:forecast_horizon])
        
        if differenced:
            last_level = train_data[target].iloc[-1]
            conf_int['lower'] = np.cumsum(conf_int['lower']) + last_level
            conf_int['upper'] = np.cumsum(conf_int['upper']) + last_level
    except Exception:
        conf_int = None
    
    # Extract target forecast
    target_forecast = forecast_df[target]
    
    # Fitted values and residuals
    fitted_values = pd.DataFrame(
        fitted_model.fittedvalues,
        columns=available_vars,
        index=train_diff.index[optimal_lag:]
    )
    residuals = pd.DataFrame(
        fitted_model.resid,
        columns=available_vars,
        index=train_diff.index[optimal_lag:]
    )
    
    params = {
        'lag_order': optimal_lag,
        'aic': fitted_model.aic,
        'bic': fitted_model.bic,
        'variables': available_vars,
        'differenced': differenced
    }
    
    # Granger causality test
    print("\nGranger Causality Tests (H0: X does not Granger-cause Y):")
    granger_results = {}
    for caused_var in available_vars:
        for causing_var in available_vars:
            if caused_var != causing_var:
                try:
                    test_data_gc = train_diff[[caused_var, causing_var]].dropna()
                    gc_result = grangercausalitytests(test_data_gc, maxlag=optimal_lag, verbose=False)
                    min_pvalue = min([gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, optimal_lag + 1)])
                    granger_results[f'{causing_var} -> {caused_var}'] = min_pvalue
                    if min_pvalue < 0.05:
                        print(f"  {causing_var} -> {caused_var}: p={min_pvalue:.4f} (significant)")
                except Exception:
                    pass
    
    params['granger_causality'] = granger_results
    
    return MultivariateResult(
        model_name='VAR',
        forecast=target_forecast,
        confidence_intervals=conf_int,
        fitted_values=fitted_values[target] if target in fitted_values.columns else fitted_values,
        residuals=residuals[target] if target in residuals.columns else residuals,
        model_object=fitted_model,
        parameters=params,
        exog_variables=available_vars
    )


def compute_impulse_response(var_result: MultivariateResult,
                              periods: int = 24,
                              shock_variable: Optional[str] = None) -> pd.DataFrame:
    """
    Compute impulse response functions from a VAR model.
    
    Parameters
    ----------
    var_result : MultivariateResult
        Fitted VAR model result
    periods : int, optional
        Number of periods for impulse response (default: 24)
    shock_variable : str, optional
        Variable to shock. If None, returns all responses
    
    Returns
    -------
    pd.DataFrame
        Impulse response coefficients
    """
    if var_result.model_name != 'VAR':
        raise ValueError("Impulse response requires a VAR model")
    
    fitted_model = var_result.model_object
    irf = fitted_model.irf(periods)
    
    variables = var_result.parameters['variables']
    
    # Create DataFrame of IRF values
    irf_data = {}
    for i, shocked_var in enumerate(variables):
        for j, response_var in enumerate(variables):
            key = f'{shocked_var} -> {response_var}'
            irf_data[key] = irf.irfs[:, j, i]
    
    irf_df = pd.DataFrame(irf_data, index=range(periods + 1))
    
    if shock_variable:
        cols = [c for c in irf_df.columns if c.startswith(f'{shock_variable} ->')]
        irf_df = irf_df[cols]
    
    return irf_df


def compute_forecast_error_variance_decomposition(var_result: MultivariateResult,
                                                   periods: int = 24) -> Dict[str, pd.DataFrame]:
    """
    Compute Forecast Error Variance Decomposition (FEVD) from a VAR model.
    
    Parameters
    ----------
    var_result : MultivariateResult
        Fitted VAR model result
    periods : int, optional
        Number of periods (default: 24)
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping variable names to their FEVD DataFrames
    """
    if var_result.model_name != 'VAR':
        raise ValueError("FEVD requires a VAR model")
    
    fitted_model = var_result.model_object
    fevd = fitted_model.fevd(periods)
    
    variables = var_result.parameters['variables']
    
    fevd_dict = {}
    for i, var in enumerate(variables):
        fevd_dict[var] = pd.DataFrame(
            fevd.decomp[:, i, :],
            columns=variables,
            index=range(1, periods + 1)
        )
    
    return fevd_dict


def compare_with_univariate(multivariate_result: MultivariateResult,
                             actual: pd.Series,
                             univariate_forecast: pd.Series) -> pd.DataFrame:
    """
    Compare multivariate model forecast with univariate model.
    
    Parameters
    ----------
    multivariate_result : MultivariateResult
        Multivariate model result
    actual : pd.Series
        Actual values
    univariate_forecast : pd.Series
        Univariate model forecast
    
    Returns
    -------
    pd.DataFrame
        Comparison DataFrame with forecasts and errors
    """
    comparison = pd.DataFrame({
        'actual': actual,
        'multivariate_forecast': multivariate_result.forecast,
        'univariate_forecast': univariate_forecast
    })
    
    comparison['multi_error'] = comparison['actual'] - comparison['multivariate_forecast']
    comparison['uni_error'] = comparison['actual'] - comparison['univariate_forecast']
    comparison['multi_abs_error'] = comparison['multi_error'].abs()
    comparison['uni_abs_error'] = comparison['uni_error'].abs()
    
    print("\nForecast Comparison Summary:")
    print(f"Multivariate ({multivariate_result.model_name}):")
    print(f"  MAE: {comparison['multi_abs_error'].mean():.4f}")
    print(f"  RMSE: {np.sqrt((comparison['multi_error']**2).mean()):.4f}")
    print(f"Univariate:")
    print(f"  MAE: {comparison['uni_abs_error'].mean():.4f}")
    print(f"  RMSE: {np.sqrt((comparison['uni_error']**2).mean()):.4f}")
    
    improvement = (comparison['uni_abs_error'].mean() - comparison['multi_abs_error'].mean()) / comparison['uni_abs_error'].mean() * 100
    print(f"\nMultivariate improvement: {improvement:.2f}%")
    
    return comparison
