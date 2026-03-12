"""
Solar Wind and Geomagnetic Time Series Analysis Package

This package provides modular functions for:
- Data preprocessing and exploration
- Univariate time series forecasting (ETS, Croston, ARIMA, SARIMA, LSTM)
- Multivariate time series modeling (ARIMAX, SARIMAX, VAR)
- Correlation and feature analysis
- Model evaluation and diagnostics
- Visualization
"""

# Suppress TensorFlow messages before any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .data_preprocessing import (
    load_data,
    clean_data,
    detect_storms,
    get_statistical_properties,
    perform_stationarity_tests,
    resample_data,
    normalize_data
)

from .visualization import (
    plot_time_series,
    plot_acf_pacf,
    plot_forecast_vs_actual,
    plot_residual_diagnostics,
    plot_correlation_heatmap,
    plot_cross_correlation,
    save_all_plots
)

from .univariate_models import (
    fit_ets,
    fit_croston,
    fit_arima,
    fit_sarima,
    fit_lstm,
    train_test_split_ts
)

from .multivariate_models import (
    fit_arimax,
    fit_sarimax,
    fit_var
)

from .correlation_analysis import (
    compute_correlations,
    compute_pearson_correlation,
    compute_cross_correlation,
    identify_lag_relationships,
    create_correlation_summary
)

from .model_evaluation import (
    evaluate_models,
    compute_forecast_metrics,
    perform_residual_analysis,
    ljung_box_test,
    generate_summary
)

__all__ = [
    # Data preprocessing
    'load_data', 'clean_data', 'detect_storms', 'get_statistical_properties',
    'perform_stationarity_tests', 'resample_data', 'normalize_data',
    # Visualization
    'plot_time_series', 'plot_acf_pacf', 'plot_forecast_vs_actual',
    'plot_residual_diagnostics', 'plot_correlation_heatmap', 
    'plot_cross_correlation', 'save_all_plots',
    # Univariate models
    'fit_ets', 'fit_croston', 'fit_arima', 'fit_sarima', 'fit_lstm',
    'train_test_split_ts',
    # Multivariate models
    'fit_arimax', 'fit_sarimax', 'fit_var',
    # Correlation analysis
    'compute_correlations', 'compute_pearson_correlation',
    'compute_cross_correlation', 'identify_lag_relationships',
    'create_correlation_summary',
    # Model evaluation
    'evaluate_models', 'compute_forecast_metrics', 'perform_residual_analysis',
    'ljung_box_test', 'generate_summary'
]
