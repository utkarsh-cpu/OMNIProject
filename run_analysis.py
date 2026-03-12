"""
Solar Wind and Geomagnetic Time Series Analysis Pipeline

This script runs a comprehensive analysis of OMNI2 solar wind data including:
1. Data Preparation and Exploration
2. AutoTS-based Automated Forecasting (replaces manual ETS/ARIMA/SARIMA/LSTM)
3. Multivariate Time Series Modeling
4. Correlation and Feature Analysis
5. Visualization and Results Summary

Usage:
    python run_analysis.py [--config config/analysis_config.yaml]
    
Author: Time Series Analysis Module
"""

import os
import sys
import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analysis.data_preprocessing import (
    load_data, clean_data, detect_storms, get_statistical_properties,
    perform_stationarity_tests, resample_data, normalize_data,
    STORM_INDICES, SOLAR_WIND_PARAMS
)
from src.analysis.visualization import (
    plot_time_series, plot_acf_pacf, plot_forecast_vs_actual,
    plot_residual_diagnostics, plot_correlation_heatmap,
    plot_cross_correlation, plot_model_comparison,
    create_summary_dashboard, save_all_plots
)
from src.analysis.autots_forecasting import (
    run_autots_forecasting,
    run_autots_for_multiple_targets,
    save_autots_results,
    plot_autots_forecasts
)
from src.analysis.multivariate_models import (
    prepare_multivariate_data, fit_arimax, fit_sarimax, fit_var
)
from src.analysis.correlation_analysis import (
    compute_correlations, compute_pearson_correlation,
    compute_cross_correlation, identify_lag_relationships,
    create_correlation_summary, analyze_storm_predictors
)
from src.analysis.model_evaluation import (
    evaluate_models, compute_forecast_metrics, perform_residual_analysis,
    ljung_box_test, generate_summary, print_evaluation_report
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------
def setup_logging(output_dir: str) -> None:
    """Configure logging to both console and file."""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'analysis_pipeline.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w'),
        ],
    )

logger = logging.getLogger(__name__)

# Configuration
class AnalysisConfig:
    """Configuration for the analysis pipeline."""
    
    # Data paths
    DATA_PATH = 'omni2_full_dataset.csv'
    OUTPUT_DIR = 'results'
    PLOTS_DIR = 'results/plots'
    RESULTS_DIR = 'results/model_results'
    FORECASTS_DIR = 'results/forecasts'
    
    # Analysis settings
    TARGET_VARIABLES = ['Kp', 'DST', 'AE']
    PREDICTOR_VARIABLES = ['Bz_GSM', 'Bz_GSE', 'plasma_speed', 'proton_density',
                           'flow_pressure', 'B_mag_avg', 'electric_field',
                           'plasma_beta', 'alfven_mach']
    
    # Date range for analysis (use recent data with better coverage)
    START_DATE = '2010-01-01'
    END_DATE = '2020-12-31'
    
    # Resampling (None for hourly, 'D' for daily)
    RESAMPLE_FREQ = 'D'  # Daily averages for faster analysis
    
    # Train/test split
    TEST_SIZE = 0.2
    
    # Forecasting settings
    FORECAST_HORIZON = 30  # Days to forecast
    
    # AutoTS settings
    AUTOTS_MAX_GENERATIONS = 5
    AUTOTS_NUM_VALIDATIONS = 2
    AUTOTS_MODEL_LIST = 'fast'
    AUTOTS_ENSEMBLE = 'simple'
    
    # Output settings
    SAVE_PLOTS = True
    PLOT_FORMAT = 'png'


def setup_directories(config: AnalysisConfig):
    """Create output directories."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FORECASTS_DIR, exist_ok=True)
    logger.info("Output directories created under: %s", config.OUTPUT_DIR)


def run_data_preparation(config: AnalysisConfig) -> pd.DataFrame:
    """
    Step 1: Data Preparation and Exploration
    
    - Load data
    - Handle missing values
    - Resample if needed
    - Compute statistics
    - Perform stationarity tests
    """
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION AND EXPLORATION")
    print("="*70)
    
    # Load data
    df = load_data(config.DATA_PATH)
    
    # Filter date range
    df = df[config.START_DATE:config.END_DATE]
    print(f"\nFiltered to date range: {config.START_DATE} to {config.END_DATE}")
    print(f"Records: {len(df)}")
    
    # Clean data
    all_vars = config.TARGET_VARIABLES + config.PREDICTOR_VARIABLES
    available_vars = [v for v in all_vars if v in df.columns]
    df = clean_data(df, columns=available_vars)
    
    # Detect storms
    df = detect_storms(df)
    
    # Resample if configured
    if config.RESAMPLE_FREQ:
        df = resample_data(df, freq=config.RESAMPLE_FREQ, agg_method='mean')
    
    # Statistical properties
    print("\nStatistical Properties:")
    stats_df = get_statistical_properties(df, columns=available_vars)
    print(stats_df.round(2).to_string())
    stats_df.to_csv(os.path.join(config.RESULTS_DIR, 'statistical_properties.csv'))
    
    # Stationarity tests
    print("\nStationarity Tests:")
    stationarity_results = []
    for var in config.TARGET_VARIABLES:
        if var in df.columns:
            result = perform_stationarity_tests(df[var].dropna(), name=var)
            stationarity_results.append(result)
    
    return df


def run_autots_forecasting_step(df: pd.DataFrame,
                                config: AnalysisConfig) -> dict:
    """
    Step 2: Automated Time Series Forecasting using AutoTS

    AutoTS automatically searches for the best forecasting model and
    parameters for each target variable, replacing the manual fitting of
    individual models (ETS, Croston, ARIMA, SARIMA, LSTM).

    - Evaluates multiple models automatically
    - Selects the best model based on validation metrics
    - Produces forecasts with confidence intervals for each target
    - Saves forecasts, model info, and performance summary
    - Generates forecast plots
    """
    print("\n" + "="*70)
    print("STEP 2: AUTOMATED FORECASTING (AutoTS)")
    print("="*70)
    logger.info("Starting AutoTS forecasting step")

    # Run AutoTS for all target variables
    available_targets = [t for t in config.TARGET_VARIABLES if t in df.columns]
    logger.info("Target variables: %s", available_targets)

    autots_results = run_autots_for_multiple_targets(
        df,
        target_columns=available_targets,
        forecast_length=config.FORECAST_HORIZON,
        frequency='infer',
        ensemble=config.AUTOTS_ENSEMBLE,
        model_list=config.AUTOTS_MODEL_LIST,
        max_generations=config.AUTOTS_MAX_GENERATIONS,
        num_validations=config.AUTOTS_NUM_VALIDATIONS,
        validation_method='backwards',
    )

    # Save results (forecasts + model info + performance summary)
    if autots_results:
        performance_df = save_autots_results(autots_results, config.OUTPUT_DIR)

        # Print performance summary
        print("\n--- AutoTS Model Performance Summary ---")
        print(performance_df.to_string(index=False))

        # Log best models
        for target, (_, _model, metrics) in autots_results.items():
            best = metrics.get('best_model', 'Unknown')
            mae = metrics.get('MAE', float('nan'))
            rmse = metrics.get('RMSE', float('nan'))
            smape = metrics.get('sMAPE', float('nan'))
            t_time = metrics.get('training_time_seconds', 0)
            logger.info(
                "%s — Best model: %s | MAE=%.4f RMSE=%.4f sMAPE=%.4f | "
                "Training time: %.1fs",
                target, best, mae, rmse, smape, t_time,
            )

        # Generate forecast plots
        if config.SAVE_PLOTS:
            plot_autots_forecasts(
                autots_results, df, config.PLOTS_DIR,
                plot_format=config.PLOT_FORMAT,
            )

    return {'autots_results': autots_results}


def run_multivariate_modeling(df: pd.DataFrame,
                               config: AnalysisConfig,
                               target: str = 'Kp') -> dict:
    """
    Step 3: Multivariate Time Series Modeling
    
    - Prepare multivariate data
    - Fit ARIMAX, SARIMAX, VAR models
    - Compare with univariate results
    """
    print("\n" + "="*70)
    print(f"STEP 3: MULTIVARIATE MODELING - {target}")
    print("="*70)
    
    # Prepare data
    predictors = [p for p in config.PREDICTOR_VARIABLES if p in df.columns][:4]  # Limit predictors
    y, X = prepare_multivariate_data(df, target, predictors)
    
    # Split
    n = len(y)
    split_idx = int(n * (1 - config.TEST_SIZE))
    
    train_y, test_y = y.iloc[:split_idx], y.iloc[split_idx:]
    train_X, test_X = X.iloc[:split_idx], X.iloc[split_idx:]
    
    print(f"\nTrain: {len(train_y)} observations")
    print(f"Test: {len(test_y)} observations")
    print(f"Predictors: {predictors}")
    
    results = {}
    
    # 1. ARIMAX
    print("\n--- Fitting ARIMAX ---")
    try:
        arimax_result = fit_arimax(train_y, train_X, test_y, test_X, auto_select=True)
        results['ARIMAX'] = arimax_result
        metrics = compute_forecast_metrics(test_y, arimax_result.forecast, train_y)
        print_evaluation_report(metrics, 'ARIMAX')
    except Exception as e:
        print(f"ARIMAX failed: {e}")
    
    # 2. SARIMAX
    print("\n--- Fitting SARIMAX ---")
    try:
        sarimax_result = fit_sarimax(train_y, train_X, test_y, test_X, auto_select=True)
        results['SARIMAX'] = sarimax_result
        metrics = compute_forecast_metrics(test_y, sarimax_result.forecast, train_y)
        print_evaluation_report(metrics, 'SARIMAX')
    except Exception as e:
        print(f"SARIMAX failed: {e}")
    
    # 3. VAR
    print("\n--- Fitting VAR ---")
    try:
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        var_vars = [target] + predictors
        
        var_result = fit_var(train_df, test_df, var_vars, target=target)
        results['VAR'] = var_result
        
        # Align forecast with test_y for evaluation
        forecast_aligned = var_result.forecast.reindex(test_y.index)
        metrics = compute_forecast_metrics(test_y, forecast_aligned, train_y)
        print_evaluation_report(metrics, 'VAR')
    except Exception as e:
        print(f"VAR failed: {e}")
    
    # Compare multivariate models
    if results:
        print("\n--- Multivariate Model Comparison ---")
        # Create aligned forecasts dict
        aligned_results = {}
        for name, res in results.items():
            if hasattr(res, 'forecast'):
                aligned_fc = res.forecast.reindex(test_y.index)
                aligned_results[name] = aligned_fc
        
        comparison_df = evaluate_models(aligned_results, test_y, train_y)
        print(comparison_df[['model', 'MAE', 'RMSE', 'MAPE', 'R2']].to_string(index=False))
        comparison_df.to_csv(
            os.path.join(config.RESULTS_DIR, f'multivariate_comparison_{target}.csv'),
            index=False
        )
    
    return results


def run_correlation_analysis(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """
    Step 4: Correlation and Feature Analysis
    
    - Pearson correlation
    - Cross-correlation with lag analysis
    - Storm predictor analysis
    """
    print("\n" + "="*70)
    print("STEP 4: CORRELATION AND FEATURE ANALYSIS")
    print("="*70)
    
    targets = [t for t in config.TARGET_VARIABLES if t in df.columns]
    predictors = [p for p in config.PREDICTOR_VARIABLES if p in df.columns]
    
    # Pearson correlation for each target
    all_correlations = []
    for target in targets:
        print(f"\n--- Correlations with {target} ---")
        corr_df = compute_correlations(df, target, predictors, method='pearson')
        corr_df['target'] = target
        all_correlations.append(corr_df)
        print(corr_df[['predictor', 'correlation', 'p_value', 'significant']].to_string(index=False))
    
    correlation_df = pd.concat(all_correlations, ignore_index=True)
    correlation_df.to_csv(os.path.join(config.RESULTS_DIR, 'pearson_correlations.csv'), index=False)
    
    # Lag relationship analysis
    print("\n--- Lag Relationships ---")
    lag_df = identify_lag_relationships(df, targets, predictors, max_lag=24)
    print(lag_df[['predictor', 'target', 'optimal_lag', 'max_correlation', 'is_significant']].head(10).to_string(index=False))
    lag_df.to_csv(os.path.join(config.RESULTS_DIR, 'lag_relationships.csv'), index=False)
    
    # Comprehensive correlation summary
    print("\n--- Correlation Summary ---")
    summary_df = create_correlation_summary(df, targets, predictors, include_lags=True)
    print(summary_df.head(15).to_string(index=False))
    summary_df.to_csv(os.path.join(config.RESULTS_DIR, 'correlation_summary.csv'), index=False)
    
    # Storm predictor analysis (if storm detection was run)
    if 'is_storm' in df.columns:
        print("\n--- Storm Predictor Analysis ---")
        storm_analysis = analyze_storm_predictors(df, 'is_storm', predictors)
        if len(storm_analysis) > 0:
            print(storm_analysis[['predictor', 'mean_difference', 't_pvalue', 'significant_difference']].to_string(index=False))
            storm_analysis.to_csv(os.path.join(config.RESULTS_DIR, 'storm_predictor_analysis.csv'), index=False)
    
    # Generate plots
    if config.SAVE_PLOTS:
        # Correlation heatmap
        all_vars = targets + predictors
        available_vars = [v for v in all_vars if v in df.columns]
        fig = plot_correlation_heatmap(df, columns=available_vars, 
                                        title='Correlation Heatmap: Storm Indices & Solar Wind Parameters')
        fig.savefig(os.path.join(config.PLOTS_DIR, f'correlation_heatmap.{config.PLOT_FORMAT}'))
        plt.close(fig)
        
        # Cross-correlation plots for top predictors
        for target in targets[:2]:  # Limit to first 2 targets
            top_predictors = lag_df[lag_df['target'] == target].nsmallest(3, 'optimal_lag')['predictor'].values
            for pred in top_predictors[:2]:  # Top 2 predictors per target
                if pred in df.columns:
                    fig = plot_cross_correlation(df[pred], df[target], pred, target)
                    fig.savefig(os.path.join(config.PLOTS_DIR, f'ccf_{pred}_{target}.{config.PLOT_FORMAT}'))
                    plt.close(fig)
    
    return summary_df


def run_visualization(df: pd.DataFrame, config: AnalysisConfig):
    """
    Step 5: Generate all visualizations
    
    - Time series plots
    - ACF/PACF plots
    - Summary dashboard
    """
    print("\n" + "="*70)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*70)
    
    if not config.SAVE_PLOTS:
        print("Plot saving disabled. Skipping visualizations.")
        return
    
    # Time series plots
    for target in config.TARGET_VARIABLES:
        if target in df.columns:
            # Full time series
            fig = plot_time_series(df, [target], title=f'{target} Time Series',
                                   ylabel=target)
            fig.savefig(os.path.join(config.PLOTS_DIR, f'timeseries_{target}.{config.PLOT_FORMAT}'))
            plt.close(fig)
            
            # ACF/PACF
            fig = plot_acf_pacf(df[target], name=target, lags=50)
            fig.savefig(os.path.join(config.PLOTS_DIR, f'acf_pacf_{target}.{config.PLOT_FORMAT}'))
            plt.close(fig)
    
    # Combined storm indices plot
    available_targets = [t for t in config.TARGET_VARIABLES if t in df.columns]
    fig = plot_time_series(df, available_targets, title='Geomagnetic Storm Indices',
                           subplot_mode=True)
    fig.savefig(os.path.join(config.PLOTS_DIR, f'storm_indices.{config.PLOT_FORMAT}'))
    plt.close(fig)
    
    # Solar wind parameters plot
    available_predictors = [p for p in config.PREDICTOR_VARIABLES[:5] if p in df.columns]
    fig = plot_time_series(df, available_predictors, title='Solar Wind Parameters',
                           subplot_mode=True)
    fig.savefig(os.path.join(config.PLOTS_DIR, f'solar_wind_params.{config.PLOT_FORMAT}'))
    plt.close(fig)
    
    # Summary dashboard
    for target in config.TARGET_VARIABLES[:2]:
        if target in df.columns:
            fig = create_summary_dashboard(df, target_var=target,
                                           predictor_vars=config.PREDICTOR_VARIABLES[:4])
            fig.savefig(os.path.join(config.PLOTS_DIR, f'dashboard_{target}.{config.PLOT_FORMAT}'))
            plt.close(fig)
    
    print(f"Visualizations saved to {config.PLOTS_DIR}")


def generate_final_summary(config: AnalysisConfig,
                            autots_step_results: dict,
                            correlation_summary: pd.DataFrame):
    """
    Step 6: Generate final results summary
    """
    print("\n" + "="*70)
    print("STEP 6: GENERATING FINAL SUMMARY")
    print("="*70)
    logger.info("Generating final summary")
    
    # Build model comparison from AutoTS results
    all_comparisons = []
    autots_results = autots_step_results.get('autots_results', {})
    for target, (_forecast, _model, metrics) in autots_results.items():
        row = {
            'model': metrics.get('best_model', 'AutoTS'),
            'MAE': metrics.get('MAE', np.nan),
            'RMSE': metrics.get('RMSE', np.nan),
            'sMAPE': metrics.get('sMAPE', np.nan),
            'R2': np.nan,
            'target_variable': target,
        }
        all_comparisons.append(row)

    # Also check for saved CSVs from multivariate step (backward compat)
    for target in config.TARGET_VARIABLES:
        comparison_file = os.path.join(config.RESULTS_DIR, f'multivariate_comparison_{target}.csv')
        if os.path.exists(comparison_file):
            comp_df = pd.read_csv(comparison_file)
            comp_df['target_variable'] = target
            all_comparisons.extend(comp_df.to_dict(orient='records'))
    
    if all_comparisons:
        combined_comparison = pd.DataFrame(all_comparisons)
    else:
        combined_comparison = pd.DataFrame()
    
    # Generate summary
    summary = generate_summary(
        model_comparison=combined_comparison,
        correlation_summary=correlation_summary,
        output_path=os.path.join(config.RESULTS_DIR, 'analysis_summary.json')
    )
    
    # Print final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print("\nKey Findings:")
    for finding in summary.get('key_findings', []):
        print(f"  - {finding}")
    
    if 'model_performance' in summary:
        best = summary['model_performance']['best_model']
        print(f"\nBest Model: {best['name']}")
        print(f"  RMSE: {best['RMSE']:.4f}")
        print(f"  MAE: {best['MAE']:.4f}")
        print(f"  R²: {best.get('R2', 'N/A')}")
    
    print(f"\nOutputs saved to:")
    print(f"  Plots: {config.PLOTS_DIR}")
    print(f"  Results: {config.RESULTS_DIR}")
    
    # List generated files
    print("\nGenerated Files:")
    for folder in [config.PLOTS_DIR, config.RESULTS_DIR]:
        if os.path.exists(folder):
            files = os.listdir(folder)
            for f in sorted(files)[:10]:
                print(f"  {os.path.join(folder, f)}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
    
    return summary


def main():
    """Main entry point for the analysis pipeline."""
    parser = argparse.ArgumentParser(description='Solar Wind Time Series Analysis')
    parser.add_argument('--data', type=str, default='omni2_full_dataset.csv',
                        help='Path to dataset')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--start-date', type=str, default='2010-01-01',
                        help='Start date for analysis')
    parser.add_argument('--end-date', type=str, default='2020-12-31',
                        help='End date for analysis')
    parser.add_argument('--resample', type=str, default='D',
                        help='Resampling frequency (H, D, W, M)')
    parser.add_argument('--target', type=str, default='Kp',
                        help='Primary target variable')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Configure
    config = AnalysisConfig()
    config.DATA_PATH = args.data
    config.OUTPUT_DIR = args.output
    config.PLOTS_DIR = os.path.join(args.output, 'plots')
    config.RESULTS_DIR = os.path.join(args.output, 'model_results')
    config.FORECASTS_DIR = os.path.join(args.output, 'forecasts')
    config.START_DATE = args.start_date
    config.END_DATE = args.end_date
    config.RESAMPLE_FREQ = args.resample if args.resample != 'None' else None
    config.SAVE_PLOTS = not args.no_plots
    
    # Setup logging
    setup_logging(config.OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("SOLAR WIND AND GEOMAGNETIC TIME SERIES ANALYSIS")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {config.DATA_PATH}")
    print(f"Date Range: {config.START_DATE} to {config.END_DATE}")
    print(f"Resampling: {config.RESAMPLE_FREQ or 'None (hourly)'}")
    
    logger.info("Pipeline started")
    logger.info("Data: %s | Range: %s to %s | Resampling: %s",
                config.DATA_PATH, config.START_DATE, config.END_DATE,
                config.RESAMPLE_FREQ or 'hourly')
    
    # Setup directories
    setup_directories(config)
    
    # Run pipeline
    try:
        # Step 1: Data Preparation
        df = run_data_preparation(config)
        
        # Step 2: AutoTS Forecasting (replaces individual ETS/ARIMA/SARIMA/LSTM)
        autots_results = run_autots_forecasting_step(df, config)
        
        # Step 3: Multivariate Modeling
        multivariate_results = run_multivariate_modeling(df, config, target=args.target)
        
        # Step 4: Correlation Analysis
        correlation_summary = run_correlation_analysis(df, config)
        
        # Step 5: Visualization
        run_visualization(df, config)
        
        # Step 6: Final Summary
        summary = generate_final_summary(config, autots_results, correlation_summary)
        
        logger.info("Pipeline completed successfully")
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.exception("Error during analysis: %s", e)
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
