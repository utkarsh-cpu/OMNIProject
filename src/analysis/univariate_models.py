"""
Univariate Time Series Forecasting Module

This module provides functions for univariate time series forecasting using:
- Exponential Smoothing (ETS)
- Croston method (for intermittent series)
- ARIMA / SARIMA
- LSTM (Deep Learning)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler

# Statsmodels imports
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

# For LSTM - import deferred to avoid TensorFlow registration issues
KERAS_AVAILABLE = False

def _init_keras():
    """Lazy initialization of Keras/TensorFlow."""
    global KERAS_AVAILABLE
    try:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        KERAS_AVAILABLE = True
        return Sequential, LSTM, Dense, Dropout, EarlyStopping
    except Exception:
        KERAS_AVAILABLE = False
        return None, None, None, None, None


@dataclass
class ForecastResult:
    """Container for forecast results."""
    model_name: str
    forecast: pd.Series
    confidence_intervals: Optional[pd.DataFrame]
    fitted_values: Optional[pd.Series]
    residuals: Optional[pd.Series]
    model_object: any
    parameters: Dict
    

def train_test_split_ts(series: pd.Series,
                         test_size: Union[int, float] = 0.2,
                         return_indices: bool = False) -> Tuple:
    """
    Split a time series into training and test sets.
    
    Parameters
    ----------
    series : pd.Series
        Time series to split
    test_size : Union[int, float], optional
        If float, proportion of data for test set (default: 0.2)
        If int, number of observations for test set
    return_indices : bool, optional
        If True, also return the split index
    
    Returns
    -------
    Tuple
        (train, test) or (train, test, split_idx) if return_indices=True
    
    Example
    -------
    >>> train, test = train_test_split_ts(df['Kp'], test_size=0.2)
    """
    n = len(series)
    
    if isinstance(test_size, float):
        test_n = int(n * test_size)
    else:
        test_n = test_size
    
    split_idx = n - test_n
    
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    
    print(f"Train set: {len(train)} observations ({train.index.min()} to {train.index.max()})")
    print(f"Test set: {len(test)} observations ({test.index.min()} to {test.index.max()})")
    
    if return_indices:
        return train, test, split_idx
    return train, test


def fit_ets(train: pd.Series,
            test: pd.Series,
            seasonal_periods: Optional[int] = None,
            trend: Optional[str] = 'add',
            seasonal: Optional[str] = 'add',
            damped_trend: bool = False,
            auto_select: bool = True) -> ForecastResult:
    """
    Fit Exponential Smoothing (ETS) model.
    
    Parameters
    ----------
    train : pd.Series
        Training time series
    test : pd.Series
        Test time series (for forecast horizon)
    seasonal_periods : int, optional
        Number of periods in a season (e.g., 24 for hourly data with daily seasonality)
    trend : str, optional
        Trend component: 'add', 'mul', or None (default: 'add')
    seasonal : str, optional
        Seasonal component: 'add', 'mul', or None (default: 'add')
    damped_trend : bool, optional
        Whether to use damped trend (default: False)
    auto_select : bool, optional
        If True, automatically select best parameters (default: True)
    
    Returns
    -------
    ForecastResult
        Object containing forecast, confidence intervals, residuals, and model
    
    Example
    -------
    >>> result = fit_ets(train, test, seasonal_periods=24)
    >>> print(result.forecast)
    """
    print("Fitting Exponential Smoothing (ETS) model...")
    
    train_clean = train.dropna()
    forecast_horizon = len(test)
    
    if auto_select:
        # Try different configurations
        best_aic = np.inf
        best_model = None
        best_params = {}
        
        trend_options = ['add', 'mul', None]
        seasonal_options = ['add', 'mul', None] if seasonal_periods else [None]
        damped_options = [True, False]
        
        for t in trend_options:
            for s in seasonal_options:
                for d in damped_options:
                    if d and t is None:  # Can't have damped without trend
                        continue
                    try:
                        model = ExponentialSmoothing(
                            train_clean,
                            trend=t,
                            seasonal=s,
                            seasonal_periods=seasonal_periods if s else None,
                            damped_trend=d
                        )
                        fitted = model.fit(optimized=True)
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_params = {
                                'trend': t, 'seasonal': s, 
                                'damped_trend': d, 'seasonal_periods': seasonal_periods
                            }
                    except Exception:
                        continue
        
        if best_model is None:
            # Fallback to simple model
            model = ExponentialSmoothing(train_clean)
            best_model = model.fit()
            best_params = {'trend': None, 'seasonal': None, 'damped_trend': False}
        
        fitted_model = best_model
        params = best_params
    else:
        model = ExponentialSmoothing(
            train_clean,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal else None,
            damped_trend=damped_trend
        )
        fitted_model = model.fit()
        params = {
            'trend': trend, 'seasonal': seasonal,
            'damped_trend': damped_trend, 'seasonal_periods': seasonal_periods
        }
    
    # Generate forecast
    forecast = fitted_model.forecast(forecast_horizon)
    forecast.index = test.index
    
    # Confidence intervals (approximation using prediction variance)
    # ETS doesn't provide built-in prediction intervals, so we use simulation
    try:
        simulations = fitted_model.simulate(forecast_horizon, repetitions=1000)
        lower = simulations.quantile(0.025, axis=1)
        upper = simulations.quantile(0.975, axis=1)
        lower.index = test.index
        upper.index = test.index
        conf_int = pd.DataFrame({'lower': lower, 'upper': upper})
    except Exception:
        conf_int = None
    
    # Fitted values and residuals
    fitted_values = pd.Series(fitted_model.fittedvalues, index=train_clean.index)
    residuals = pd.Series(fitted_model.resid, index=train_clean.index)
    
    print(f"ETS Model fitted: AIC={fitted_model.aic:.2f}")
    print(f"Parameters: {params}")
    
    return ForecastResult(
        model_name='ETS',
        forecast=forecast,
        confidence_intervals=conf_int,
        fitted_values=fitted_values,
        residuals=residuals,
        model_object=fitted_model,
        parameters=params
    )


def fit_croston(train: pd.Series,
                test: pd.Series,
                alpha: float = 0.1) -> ForecastResult:
    """
    Fit Croston's method for intermittent demand forecasting.
    
    Croston's method is suitable for series with many zeros or intermittent patterns.
    
    Parameters
    ----------
    train : pd.Series
        Training time series
    test : pd.Series
        Test time series (for forecast horizon)
    alpha : float, optional
        Smoothing parameter (default: 0.1)
    
    Returns
    -------
    ForecastResult
        Object containing forecast and model parameters
    
    Example
    -------
    >>> result = fit_croston(train, test, alpha=0.1)
    """
    print("Fitting Croston's method...")
    
    train_clean = train.dropna().values
    forecast_horizon = len(test)
    
    # Initialize
    n = len(train_clean)
    
    # Find first non-zero value
    first_nonzero = np.where(train_clean != 0)[0]
    if len(first_nonzero) == 0:
        # All zeros - return zero forecast
        forecast = pd.Series(np.zeros(forecast_horizon), index=test.index)
        return ForecastResult(
            model_name='Croston',
            forecast=forecast,
            confidence_intervals=None,
            fitted_values=None,
            residuals=None,
            model_object=None,
            parameters={'alpha': alpha, 'note': 'all zeros in training'}
        )
    
    # Initialize demand level (z) and inter-arrival time (p)
    z = train_clean[first_nonzero[0]]
    p = first_nonzero[0] + 1 if first_nonzero[0] > 0 else 1
    
    # Track fitted values
    fitted = np.zeros(n)
    q = 0  # periods since last demand
    
    for i in range(n):
        q += 1
        if train_clean[i] != 0:
            z = alpha * train_clean[i] + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p
            q = 0
        fitted[i] = z / p
    
    # Forecast (flat forecast)
    forecast_value = z / p
    forecast = pd.Series(np.full(forecast_horizon, forecast_value), index=test.index)
    
    fitted_values = pd.Series(fitted, index=train.index[:n])
    residuals = train.iloc[:n] - fitted_values
    
    params = {
        'alpha': alpha,
        'final_demand_level': z,
        'final_inter_arrival': p,
        'forecast_value': forecast_value
    }
    
    print(f"Croston's method fitted: forecast={forecast_value:.4f}")
    
    return ForecastResult(
        model_name='Croston',
        forecast=forecast,
        confidence_intervals=None,
        fitted_values=fitted_values,
        residuals=residuals,
        model_object=None,
        parameters=params
    )


def fit_arima(train: pd.Series,
              test: pd.Series,
              order: Optional[Tuple[int, int, int]] = None,
              auto_select: bool = True,
              seasonal: bool = False,
              seasonal_order: Optional[Tuple[int, int, int, int]] = None,
              max_p: int = 5,
              max_q: int = 5,
              max_d: int = 2) -> ForecastResult:
    """
    Fit ARIMA or SARIMA model.
    
    Parameters
    ----------
    train : pd.Series
        Training time series
    test : pd.Series
        Test time series (for forecast horizon)
    order : Tuple[int, int, int], optional
        (p, d, q) order. If None and auto_select=True, will be determined automatically
    auto_select : bool, optional
        If True, use auto_arima to find best parameters (default: True)
    seasonal : bool, optional
        Whether to fit a seasonal model (default: False)
    seasonal_order : Tuple[int, int, int, int], optional
        (P, D, Q, m) seasonal order
    max_p : int, optional
        Maximum AR order to try (default: 5)
    max_q : int, optional
        Maximum MA order to try (default: 5)
    max_d : int, optional
        Maximum differencing order (default: 2)
    
    Returns
    -------
    ForecastResult
        Object containing forecast, confidence intervals, residuals, and model
    
    Example
    -------
    >>> result = fit_arima(train, test, auto_select=True)
    >>> result = fit_arima(train, test, order=(2, 1, 2))
    """
    model_name = 'SARIMA' if seasonal else 'ARIMA'
    print(f"Fitting {model_name} model...")
    
    train_clean = train.dropna()
    forecast_horizon = len(test)
    
    if auto_select:
        print("Running auto_arima to find best parameters...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto_model = pm.auto_arima(
                train_clean,
                start_p=0, max_p=max_p,
                start_q=0, max_q=max_q,
                d=None, max_d=max_d,
                seasonal=seasonal,
                m=24 if seasonal else 1,  # Assume hourly with daily seasonality
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
        seasonal_order = auto_model.seasonal_order if seasonal else None
        print(f"Auto-selected order: {order}, seasonal_order: {seasonal_order}")
    else:
        if order is None:
            order = (1, 1, 1)
    
    # Fit the model
    if seasonal and seasonal_order:
        model = SARIMAX(
            train_clean,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
    else:
        model = ARIMA(train_clean, order=order)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted_model = model.fit()
    
    # Generate forecast with confidence intervals
    forecast_obj = fitted_model.get_forecast(steps=forecast_horizon)
    forecast = forecast_obj.predicted_mean
    forecast.index = test.index
    
    conf_int_raw = forecast_obj.conf_int(alpha=0.05)
    conf_int = pd.DataFrame({
        'lower': conf_int_raw.iloc[:, 0].values,
        'upper': conf_int_raw.iloc[:, 1].values
    }, index=test.index)
    
    # Fitted values and residuals
    fitted_values = fitted_model.fittedvalues
    residuals = fitted_model.resid
    
    params = {
        'order': order,
        'seasonal_order': seasonal_order,
        'aic': fitted_model.aic,
        'bic': fitted_model.bic
    }
    
    print(f"{model_name} Model fitted: order={order}, AIC={fitted_model.aic:.2f}")
    
    return ForecastResult(
        model_name=model_name,
        forecast=forecast,
        confidence_intervals=conf_int,
        fitted_values=fitted_values,
        residuals=residuals,
        model_object=fitted_model,
        parameters=params
    )


def fit_sarima(train: pd.Series,
               test: pd.Series,
               order: Optional[Tuple[int, int, int]] = None,
               seasonal_order: Optional[Tuple[int, int, int, int]] = None,
               auto_select: bool = True) -> ForecastResult:
    """
    Fit SARIMA model (wrapper for fit_arima with seasonal=True).
    
    Parameters
    ----------
    train : pd.Series
        Training time series
    test : pd.Series
        Test time series
    order : Tuple[int, int, int], optional
        (p, d, q) order
    seasonal_order : Tuple[int, int, int, int], optional
        (P, D, Q, m) seasonal order
    auto_select : bool, optional
        If True, use auto_arima (default: True)
    
    Returns
    -------
    ForecastResult
        Object containing forecast and model
    
    Example
    -------
    >>> result = fit_sarima(train, test, auto_select=True)
    """
    return fit_arima(
        train, test, 
        order=order,
        seasonal=True,
        seasonal_order=seasonal_order,
        auto_select=auto_select
    )


def create_lstm_sequences(data: np.ndarray,
                          lookback: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Parameters
    ----------
    data : np.ndarray
        Input data (scaled)
    lookback : int, optional
        Number of time steps to look back (default: 24)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X (input sequences) and y (targets)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def fit_lstm(train: pd.Series,
             test: pd.Series,
             lookback: int = 24,
             lstm_units: int = 50,
             epochs: int = 50,
             batch_size: int = 32,
             dropout: float = 0.2,
             verbose: int = 0) -> ForecastResult:
    """
    Fit an LSTM deep learning model for time series forecasting.
    
    Parameters
    ----------
    train : pd.Series
        Training time series
    test : pd.Series
        Test time series
    lookback : int, optional
        Number of time steps to look back (default: 24)
    lstm_units : int, optional
        Number of LSTM units (default: 50)
    epochs : int, optional
        Number of training epochs (default: 50)
    batch_size : int, optional
        Batch size for training (default: 32)
    dropout : float, optional
        Dropout rate (default: 0.2)
    verbose : int, optional
        Verbosity level (default: 0)
    
    Returns
    -------
    ForecastResult
        Object containing forecast and model
    
    Example
    -------
    >>> result = fit_lstm(train, test, lookback=24, epochs=100)
    """
    # Lazy load TensorFlow/Keras
    Sequential, LSTM_Layer, Dense, Dropout, EarlyStopping = _init_keras()
    
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required for LSTM. Install with: pip install tensorflow")
    
    print("Fitting LSTM model...")
    
    # Prepare data
    train_clean = train.dropna()
    test_clean = test.dropna()
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_clean.values.reshape(-1, 1))
    
    # Create sequences
    X_train, y_train = create_lstm_sequences(train_scaled, lookback)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    # Build LSTM model
    model = Sequential([
        LSTM_Layer(lstm_units, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout),
        LSTM_Layer(lstm_units, return_sequences=False),
        Dropout(dropout),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Early stopping
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )
    
    # Generate forecasts
    # Use last 'lookback' values from training to start forecasting
    forecast_horizon = len(test)
    
    # Combine train and test for scaling context
    full_data = pd.concat([train_clean, test_clean])
    full_scaled = scaler.transform(full_data.values.reshape(-1, 1))
    
    # Generate predictions for test period
    predictions = []
    current_input = train_scaled[-lookback:].reshape(1, lookback, 1)
    
    for _ in range(forecast_horizon):
        pred = model.predict(current_input, verbose=0)[0, 0]
        predictions.append(pred)
        # Update input window
        new_input = np.append(current_input[0, 1:, 0], pred).reshape(1, lookback, 1)
        current_input = new_input
    
    # Inverse transform predictions
    predictions_inverse = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    forecast = pd.Series(predictions_inverse.flatten(), index=test.index)
    
    # Calculate fitted values for training period
    fitted_preds_scaled = model.predict(X_train, verbose=0)
    fitted_preds = scaler.inverse_transform(fitted_preds_scaled)
    fitted_values = pd.Series(fitted_preds.flatten(), 
                               index=train_clean.index[lookback:])
    
    # Residuals
    residuals = train_clean.iloc[lookback:] - fitted_values
    
    params = {
        'lookback': lookback,
        'lstm_units': lstm_units,
        'epochs': len(history.history['loss']),
        'final_loss': history.history['loss'][-1],
        'dropout': dropout,
        'batch_size': batch_size
    }
    
    print(f"LSTM Model fitted: epochs={params['epochs']}, final_loss={params['final_loss']:.6f}")
    
    return ForecastResult(
        model_name='LSTM',
        forecast=forecast,
        confidence_intervals=None,  # LSTM doesn't provide native CIs
        fitted_values=fitted_values,
        residuals=residuals,
        model_object={'model': model, 'scaler': scaler, 'history': history},
        parameters=params
    )


def forecast_multiple_steps(model_result: ForecastResult,
                            steps: int,
                            return_conf_int: bool = True) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """
    Generate multi-step forecasts using a fitted model.
    
    Parameters
    ----------
    model_result : ForecastResult
        Fitted model result
    steps : int
        Number of steps to forecast
    return_conf_int : bool, optional
        Whether to return confidence intervals
    
    Returns
    -------
    Tuple[pd.Series, Optional[pd.DataFrame]]
        Forecast and optional confidence intervals
    """
    model = model_result.model_object
    
    if model_result.model_name in ['ARIMA', 'SARIMA']:
        forecast_obj = model.get_forecast(steps=steps)
        forecast = forecast_obj.predicted_mean
        
        if return_conf_int:
            conf_int = forecast_obj.conf_int(alpha=0.05)
            conf_int_df = pd.DataFrame({
                'lower': conf_int.iloc[:, 0],
                'upper': conf_int.iloc[:, 1]
            })
            return forecast, conf_int_df
        return forecast, None
    
    elif model_result.model_name == 'ETS':
        forecast = model.forecast(steps)
        return forecast, None
    
    else:
        raise ValueError(f"Multi-step forecast not implemented for {model_result.model_name}")


def check_series_intermittency(series: pd.Series,
                                zero_threshold: float = 0.3) -> Dict:
    """
    Check if a series is intermittent (has many zeros).
    
    Parameters
    ----------
    series : pd.Series
        Time series to check
    zero_threshold : float, optional
        Proportion of zeros to classify as intermittent (default: 0.3)
    
    Returns
    -------
    Dict
        Dictionary with intermittency statistics
    """
    series_clean = series.dropna()
    n = len(series_clean)
    n_zeros = (series_clean == 0).sum()
    zero_proportion = n_zeros / n
    
    # Average demand interval (ADI) - average periods between non-zero values
    non_zero_idx = np.where(series_clean.values != 0)[0]
    if len(non_zero_idx) > 1:
        intervals = np.diff(non_zero_idx)
        adi = intervals.mean()
    else:
        adi = n
    
    # Coefficient of variation of demand
    non_zero_values = series_clean[series_clean != 0]
    cv2 = (non_zero_values.std() / non_zero_values.mean()) ** 2 if len(non_zero_values) > 0 else 0
    
    # Classification (Syntetos & Boylan)
    # ADI < 1.32 and CV2 < 0.49: Smooth
    # ADI >= 1.32 and CV2 < 0.49: Intermittent
    # ADI < 1.32 and CV2 >= 0.49: Erratic
    # ADI >= 1.32 and CV2 >= 0.49: Lumpy
    
    if adi < 1.32:
        if cv2 < 0.49:
            classification = 'smooth'
        else:
            classification = 'erratic'
    else:
        if cv2 < 0.49:
            classification = 'intermittent'
        else:
            classification = 'lumpy'
    
    is_intermittent = zero_proportion >= zero_threshold or classification in ['intermittent', 'lumpy']
    
    return {
        'n_observations': n,
        'n_zeros': n_zeros,
        'zero_proportion': zero_proportion,
        'adi': adi,
        'cv2': cv2,
        'classification': classification,
        'is_intermittent': is_intermittent,
        'recommendation': 'Croston' if is_intermittent else 'Standard methods (ARIMA/ETS)'
    }
