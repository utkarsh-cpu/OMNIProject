"""
AutoTS Forecasting Module

This module replaces individual model implementations (ETS, Croston, ARIMA, SARIMA, LSTM)
with a unified AutoTS-based forecasting function that automatically searches for the best
forecasting model and parameters.

AutoTS Workflow:
    1. Accept a dataframe with datetime index and target column(s)
    2. Automatically evaluate multiple forecasting models
    3. Select the best model based on validation metrics
    4. Produce forecasts with confidence intervals
    5. Return predictions, metrics, and the fitted model

Reference: https://github.com/winedarksea/AutoTS
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from autots import AutoTS

logger = logging.getLogger(__name__)


def run_autots_forecasting(
    df: pd.DataFrame,
    target_column: str,
    forecast_length: int = 24,
    frequency: str = "infer",
    ensemble: str = "simple",
    model_list: str = "fast",
    transformer_list: str = "fast",
    max_generations: int = 5,
    num_validations: int = 2,
    validation_method: str = "backwards",
    date_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, AutoTS, Dict[str, float]]:
    """
    Run AutoTS to automatically search for the best forecasting model.

    This function replaces individual model functions (fit_ets, fit_croston,
    fit_arima, fit_sarima, fit_lstm) with a single unified AutoTS call that
    evaluates multiple models and selects the best one.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the time series data.
        Should have a datetime index or a date column specified by ``date_col``.
    target_column : str
        Name of the column to forecast.
    forecast_length : int, optional
        Number of periods to forecast ahead (default: 24).
    frequency : str, optional
        Frequency of the time series. Use ``'infer'`` for auto-detection
        (default: ``'infer'``).
    ensemble : str, optional
        Ensemble method – ``'simple'``, ``'distance'``, or ``None``
        (default: ``'simple'``).
    model_list : str, optional
        Preset model list – ``'fast'``, ``'superfast'``, ``'default'``, or
        ``'all'`` (default: ``'fast'``).
    transformer_list : str, optional
        Preset transformer list – ``'fast'``, ``'superfast'``, or ``'all'``
        (default: ``'fast'``).
    max_generations : int, optional
        Maximum number of generations for the genetic algorithm search
        (default: 5).
    num_validations : int, optional
        Number of back-test validation rounds (default: 2).
    validation_method : str, optional
        Validation strategy – ``'backwards'``, ``'even'``, or
        ``'seasonal <n>'`` (default: ``'backwards'``).
    date_col : str, optional
        Name of the date column. If ``None``, the dataframe index is used
        as the datetime index.
    id_col : str, optional
        Name of the series identifier column for multi-series data.
        ``None`` for single-series forecasting.

    Returns
    -------
    Tuple[pd.DataFrame, AutoTS, Dict[str, float]]
        - **forecast** – DataFrame with forecasted values.
        - **model** – The fitted AutoTS model object.
        - **metrics** – Dictionary with MAE, RMSE, and sMAPE of the best model.

    Example
    -------
    >>> forecast, model, metrics = run_autots_forecasting(
    ...     df, target_column='Kp', forecast_length=30
    ... )
    >>> print(forecast.head())
    >>> print(metrics)
    """
    logger.info("Starting AutoTS forecasting for target: %s", target_column)
    start_time = time.time()

    # --- Prepare the dataframe ------------------------------------------------
    # AutoTS expects a dataframe; if the index is a DatetimeIndex and no
    # date_col is given, reset it into a column named 'date'.
    forecast_df = df.copy()

    if date_col is None:
        if isinstance(forecast_df.index, pd.DatetimeIndex):
            forecast_df = forecast_df.reset_index()
            # The reset index column name may vary; normalise to 'date'.
            idx_name = forecast_df.columns[0]
            forecast_df = forecast_df.rename(columns={idx_name: "date"})
            date_col = "date"
        else:
            raise ValueError(
                "The dataframe must have a DatetimeIndex or a 'date_col' "
                "parameter must be provided."
            )

    # Keep only the date column and the target column for univariate forecast
    forecast_df = forecast_df[[date_col, target_column]].dropna()

    logger.info(
        "Data prepared: %d rows, forecasting %d periods ahead",
        len(forecast_df),
        forecast_length,
    )

    # --- Initialise and fit the AutoTS model ----------------------------------
    model = AutoTS(
        forecast_length=forecast_length,
        frequency=frequency,
        ensemble=ensemble,
        model_list=model_list,
        transformer_list=transformer_list,
        max_generations=max_generations,
        num_validations=num_validations,
        validation_method=validation_method,
    )

    logger.info(
        "Fitting AutoTS (model_list=%s, max_generations=%d, "
        "num_validations=%d, validation=%s) ...",
        model_list,
        max_generations,
        num_validations,
        validation_method,
    )

    model = model.fit(
        forecast_df,
        date_col=date_col,
        value_col=target_column,
        id_col=id_col,
    )

    training_time = time.time() - start_time
    logger.info("AutoTS fitting completed in %.2f seconds", training_time)

    # --- Extract the best model information -----------------------------------
    best_model_name = model.best_model_name
    best_model_params = model.best_model_params
    logger.info("Best model selected: %s", best_model_name)
    logger.info("Best model parameters: %s", best_model_params)

    # --- Generate forecast ----------------------------------------------------
    prediction = model.predict()
    forecast = prediction.forecast
    upper_forecast = prediction.upper_forecast
    lower_forecast = prediction.lower_forecast

    logger.info("Forecast generated for %d periods", len(forecast))

    # --- Extract validation / back-test metrics -------------------------------
    metrics = _extract_metrics(model, training_time, best_model_name)

    logger.info(
        "Best model metrics — MAE: %.4f, RMSE: %.4f, sMAPE: %.4f",
        metrics.get("MAE", float("nan")),
        metrics.get("RMSE", float("nan")),
        metrics.get("sMAPE", float("nan")),
    )

    # Attach confidence intervals to the forecast dataframe for convenience
    if upper_forecast is not None and lower_forecast is not None:
        for col in forecast.columns:
            forecast[f"{col}_upper"] = upper_forecast[col]
            forecast[f"{col}_lower"] = lower_forecast[col]

    return forecast, model, metrics


def run_autots_for_multiple_targets(
    df: pd.DataFrame,
    target_columns: List[str],
    forecast_length: int = 24,
    **kwargs,
) -> Dict[str, Tuple[pd.DataFrame, AutoTS, Dict[str, float]]]:
    """
    Run AutoTS forecasting for multiple target variables.

    Iterates over each target column and runs ``run_autots_forecasting``.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index.
    target_columns : List[str]
        List of column names to forecast (e.g. ``['Kp', 'DST', 'AE']``).
    forecast_length : int, optional
        Forecast horizon (default: 24).
    **kwargs
        Additional keyword arguments forwarded to ``run_autots_forecasting``.

    Returns
    -------
    Dict[str, Tuple[pd.DataFrame, AutoTS, Dict[str, float]]]
        Mapping from target name to ``(forecast, model, metrics)`` tuples.
    """
    all_results: Dict[str, Tuple[pd.DataFrame, AutoTS, Dict[str, float]]] = {}

    for target in target_columns:
        if target not in df.columns:
            logger.warning("Target column '%s' not found in dataframe — skipping.", target)
            continue

        logger.info("=" * 60)
        logger.info("Forecasting target: %s", target)
        logger.info("=" * 60)

        try:
            forecast, model, metrics = run_autots_forecasting(
                df,
                target_column=target,
                forecast_length=forecast_length,
                **kwargs,
            )
            all_results[target] = (forecast, model, metrics)
        except Exception:
            logger.exception("AutoTS forecasting failed for target '%s'", target)

    return all_results


def save_autots_results(
    results: Dict[str, Tuple[pd.DataFrame, AutoTS, Dict[str, float]]],
    output_dir: str,
) -> pd.DataFrame:
    """
    Save AutoTS forecasting results to disk.

    Persisted artefacts:

    * ``results/forecasts/<target>_forecast.csv`` – forecast values per target
    * ``results/model_results/<target>_model_info.csv`` – best model information
    * ``results/model_results/model_performance_summary.csv`` – combined metrics

    Parameters
    ----------
    results : Dict[str, Tuple[pd.DataFrame, AutoTS, Dict[str, float]]]
        Results from ``run_autots_for_multiple_targets``.
    output_dir : str
        Base output directory (e.g. ``'results'``).

    Returns
    -------
    pd.DataFrame
        Combined model performance summary across all targets.
    """
    import os

    forecasts_dir = os.path.join(output_dir, "forecasts")
    model_results_dir = os.path.join(output_dir, "model_results")
    os.makedirs(forecasts_dir, exist_ok=True)
    os.makedirs(model_results_dir, exist_ok=True)

    all_metrics: List[Dict] = []

    for target, (forecast, model, metrics) in results.items():
        # Save forecast CSV
        forecast_path = os.path.join(forecasts_dir, f"{target}_forecast.csv")
        forecast.to_csv(forecast_path)
        logger.info("Saved forecast to %s", forecast_path)

        # Save best-model information
        model_info = {
            "target": target,
            "best_model": metrics.get("best_model", ""),
            "MAE": metrics.get("MAE", None),
            "RMSE": metrics.get("RMSE", None),
            "sMAPE": metrics.get("sMAPE", None),
            "training_time_seconds": metrics.get("training_time_seconds", None),
        }
        info_df = pd.DataFrame([model_info])
        info_path = os.path.join(model_results_dir, f"{target}_model_info.csv")
        info_df.to_csv(info_path, index=False)
        logger.info("Saved model info to %s", info_path)

        all_metrics.append(model_info)

    # Combined performance summary
    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(model_results_dir, "model_performance_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved model performance summary to %s", summary_path)

    return summary_df


def plot_autots_forecasts(
    results: Dict[str, Tuple[pd.DataFrame, AutoTS, Dict[str, float]]],
    df: pd.DataFrame,
    plots_dir: str,
    plot_format: str = "png",
) -> None:
    """
    Generate and save forecast plots for each target variable.

    For every target in *results* two plots are produced:

    1. **Actual vs Forecast** – the historical tail and the forecasted values.
    2. **Forecast with confidence intervals** – upper / lower bounds.

    Parameters
    ----------
    results : Dict[str, Tuple[pd.DataFrame, AutoTS, Dict[str, float]]]
        Results from ``run_autots_for_multiple_targets``.
    df : pd.DataFrame
        Original dataframe with historical data (datetime index).
    plots_dir : str
        Directory to save plots into.
    plot_format : str, optional
        Image format (default: ``'png'``).
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)

    for target, (forecast, _model, metrics) in results.items():
        # Determine the actual target column in the forecast dataframe
        target_cols = [c for c in forecast.columns if not c.endswith(("_upper", "_lower"))]
        if not target_cols:
            continue
        fc_col = target_cols[0]

        fc_values = forecast[fc_col]
        upper_col = f"{fc_col}_upper"
        lower_col = f"{fc_col}_lower"
        has_ci = upper_col in forecast.columns and lower_col in forecast.columns

        # ----- Plot 1: Actual vs Forecast ------------------------------------
        fig, ax = plt.subplots(figsize=(12, 6))

        # Show last portion of actual data for context
        if target in df.columns:
            actual = df[target].dropna()
            n_context = min(len(actual), len(fc_values) * 3)
            context = actual.iloc[-n_context:]
            ax.plot(context.index, context.values, label="Actual", color="#1f77b4")

        ax.plot(fc_values.index, fc_values.values, label="Forecast", color="#d62728", linestyle="--")

        best_name = metrics.get("best_model", "AutoTS")
        ax.set_title(f"{target} — Actual vs Forecast ({best_name})")
        ax.set_xlabel("Date")
        ax.set_ylabel(target)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = os.path.join(plots_dir, f"forecast_{target}_autots.{plot_format}")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved plot: %s", path)

        # ----- Plot 2: Forecast with confidence intervals --------------------
        if has_ci:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(fc_values.index, fc_values.values, label="Forecast", color="#d62728")
            ax.fill_between(
                forecast.index,
                forecast[lower_col].values,
                forecast[upper_col].values,
                alpha=0.2,
                color="#d62728",
                label="Confidence Interval",
            )
            ax.set_title(f"{target} — Forecast Confidence Intervals ({best_name})")
            ax.set_xlabel("Date")
            ax.set_ylabel(target)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            path = os.path.join(plots_dir, f"forecast_ci_{target}_autots.{plot_format}")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            logger.info("Saved plot: %s", path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_metrics(
    model: AutoTS,
    training_time: float,
    best_model_name: str,
) -> Dict[str, float]:
    """
    Extract MAE, RMSE, and sMAPE from a fitted AutoTS model's validation results.

    Parameters
    ----------
    model : AutoTS
        Fitted AutoTS model.
    training_time : float
        Wall-clock training time in seconds.
    best_model_name : str
        Name of the best model selected by AutoTS.

    Returns
    -------
    Dict[str, float]
        Dictionary containing extracted metrics.
    """
    metrics: Dict[str, float] = {
        "best_model": best_model_name,
        "training_time_seconds": round(training_time, 2),
    }

    # AutoTS stores per-model validation results in .results()
    try:
        results_df = model.results()
        if results_df is not None and not results_df.empty:
            # AutoTS results may use different casing across versions;
            # check both lowercase and uppercase variants and normalise.
            best_row = results_df.iloc[0]
            seen = set()
            for metric_name in best_row.index:
                canonical = metric_name.upper()
                if canonical == "SMAPE":
                    canonical = "sMAPE"
                if canonical in {"MAE", "RMSE", "sMAPE", "SPL"} and canonical not in seen:
                    metrics[canonical] = float(best_row[metric_name])
                    seen.add(canonical)
    except Exception:
        logger.debug("Could not extract detailed metrics from AutoTS results.")

    return metrics
