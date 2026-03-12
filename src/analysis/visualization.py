"""
Visualization Module for Solar Wind and Geomagnetic Time Series Analysis

This module provides functions for creating various plots including:
- Time series plots
- ACF/PACF plots
- Forecast vs actual comparisons
- Residual diagnostics
- Correlation heatmaps
- Cross-correlation plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import os


# Set style defaults
plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_DPI = 150
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'forecast': '#d62728',
    'actual': '#1f77b4',
    'residual': '#7f7f7f'
}


def setup_plot_style():
    """Configure matplotlib style settings."""
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'figure.dpi': FIGURE_DPI,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
        'grid.alpha': 0.3
    })


def plot_time_series(df: pd.DataFrame,
                     columns: List[str],
                     title: str = "Time Series Plot",
                     ylabel: str = "Value",
                     figsize: Tuple[int, int] = (14, 8),
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     save_path: Optional[str] = None,
                     subplot_mode: bool = False) -> plt.Figure:
    """
    Create time series plots for specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index
    columns : List[str]
        Columns to plot
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    figsize : Tuple[int, int], optional
        Figure size (default: (14, 8))
    start_date : str, optional
        Start date for plot range
    end_date : str, optional
        End date for plot range
    save_path : str, optional
        Path to save the figure
    subplot_mode : bool, optional
        If True, create separate subplots for each column
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Example
    -------
    >>> fig = plot_time_series(df, ['Kp', 'DST'], title='Geomagnetic Indices')
    """
    setup_plot_style()
    
    # Filter columns that exist
    columns = [col for col in columns if col in df.columns]
    if not columns:
        raise ValueError("No valid columns to plot")
    
    # Filter date range
    df_plot = df.copy()
    if start_date:
        df_plot = df_plot[df_plot.index >= start_date]
    if end_date:
        df_plot = df_plot[df_plot.index <= end_date]
    
    if subplot_mode:
        fig, axes = plt.subplots(len(columns), 1, figsize=(figsize[0], 3 * len(columns)), 
                                  sharex=True)
        if len(columns) == 1:
            axes = [axes]
        
        for ax, col in zip(axes, columns):
            ax.plot(df_plot.index, df_plot[col], label=col, linewidth=1)
            ax.set_ylabel(col)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date')
        fig.suptitle(title, fontsize=14, y=1.02)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, col in enumerate(columns):
            ax.plot(df_plot.index, df_plot[col], label=col, linewidth=1)
        
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_acf_pacf(series: pd.Series,
                  name: str = "Series",
                  lags: int = 40,
                  figsize: Tuple[int, int] = (14, 5),
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ACF and PACF for a time series.
    
    Parameters
    ----------
    series : pd.Series
        Time series to analyze
    name : str, optional
        Series name for title
    lags : int, optional
        Number of lags to display (default: 40)
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Example
    -------
    >>> fig = plot_acf_pacf(df['Kp'], name='Kp', lags=50)
    """
    setup_plot_style()
    
    # Remove NaN values
    series_clean = series.dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ACF plot
    plot_acf(series_clean, ax=axes[0], lags=lags, alpha=0.05)
    axes[0].set_title(f'Autocorrelation Function (ACF) - {name}')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Autocorrelation')
    
    # PACF plot
    plot_pacf(series_clean, ax=axes[1], lags=lags, alpha=0.05, method='ywm')
    axes[1].set_title(f'Partial Autocorrelation Function (PACF) - {name}')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Partial Autocorrelation')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved ACF/PACF plot to {save_path}")
    
    return fig


def plot_forecast_vs_actual(actual: pd.Series,
                             forecast: pd.Series,
                             confidence_intervals: Optional[pd.DataFrame] = None,
                             train: Optional[pd.Series] = None,
                             model_name: str = "Model",
                             variable_name: str = "Value",
                             figsize: Tuple[int, int] = (14, 6),
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot forecast values against actual values.
    
    Parameters
    ----------
    actual : pd.Series
        Actual observed values (test set)
    forecast : pd.Series
        Forecasted values
    confidence_intervals : pd.DataFrame, optional
        DataFrame with 'lower' and 'upper' columns for confidence intervals
    train : pd.Series, optional
        Training data to show context
    model_name : str, optional
        Name of the model for title
    variable_name : str, optional
        Name of the variable being forecast
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Example
    -------
    >>> fig = plot_forecast_vs_actual(test_kp, forecast_kp, model_name='ARIMA', variable_name='Kp')
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training data if provided (last portion)
    if train is not None:
        # Show last 20% of training data for context
        n_train_show = min(len(train), int(0.2 * len(train)))
        train_show = train.iloc[-n_train_show:]
        ax.plot(train_show.index, train_show.values, 
                color=COLORS['tertiary'], label='Training Data', linewidth=1, alpha=0.7)
    
    # Plot actual values
    ax.plot(actual.index, actual.values, 
            color=COLORS['actual'], label='Actual', linewidth=2)
    
    # Plot forecast
    ax.plot(forecast.index, forecast.values, 
            color=COLORS['forecast'], label='Forecast', linewidth=2, linestyle='--')
    
    # Plot confidence intervals if provided
    if confidence_intervals is not None:
        ax.fill_between(forecast.index, 
                        confidence_intervals['lower'], 
                        confidence_intervals['upper'],
                        color=COLORS['forecast'], alpha=0.2, 
                        label='95% Confidence Interval')
    
    ax.set_xlabel('Date')
    ax.set_ylabel(variable_name)
    ax.set_title(f'{model_name} - Forecast vs Actual ({variable_name})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved forecast plot to {save_path}")
    
    return fig


def plot_residual_diagnostics(residuals: pd.Series,
                               model_name: str = "Model",
                               figsize: Tuple[int, int] = (14, 10),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create residual diagnostic plots.
    
    Includes:
    - Residuals over time
    - Histogram with KDE
    - Q-Q plot
    - ACF of residuals
    
    Parameters
    ----------
    residuals : pd.Series
        Model residuals
    model_name : str, optional
        Name of the model
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Example
    -------
    >>> fig = plot_residual_diagnostics(model.resid, model_name='ARIMA')
    """
    setup_plot_style()
    
    residuals_clean = residuals.dropna()
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)
    
    # Residuals over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(residuals_clean.index, residuals_clean.values, 
             color=COLORS['residual'], linewidth=0.8)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residual')
    ax1.set_title('Residuals Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals_clean, bins=50, density=True, alpha=0.7, 
             color=COLORS['primary'], edgecolor='white')
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(residuals_clean)
    x = np.linspace(residuals_clean.min(), residuals_clean.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
             label=f'Normal fit\n(μ={mu:.2f}, σ={sigma:.2f})')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Density')
    ax2.set_title('Residual Distribution')
    ax2.legend()
    
    # Q-Q plot
    ax3 = fig.add_subplot(gs[1, 0])
    stats.probplot(residuals_clean, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normality Check)')
    ax3.grid(True, alpha=0.3)
    
    # ACF of residuals
    ax4 = fig.add_subplot(gs[1, 1])
    plot_acf(residuals_clean, ax=ax4, lags=40, alpha=0.05)
    ax4.set_title('ACF of Residuals')
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('Autocorrelation')
    
    fig.suptitle(f'{model_name} - Residual Diagnostics', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved residual diagnostics to {save_path}")
    
    return fig


def plot_correlation_heatmap(df: pd.DataFrame,
                              columns: Optional[List[str]] = None,
                              method: str = 'pearson',
                              title: str = "Correlation Heatmap",
                              figsize: Tuple[int, int] = (12, 10),
                              annot: bool = True,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a correlation heatmap.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : List[str], optional
        Columns to include. If None, uses all numeric columns
    method : str, optional
        Correlation method: 'pearson', 'spearman', 'kendall' (default: 'pearson')
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size
    annot : bool, optional
        Whether to annotate cells with correlation values
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Example
    -------
    >>> fig = plot_correlation_heatmap(df, columns=['Kp', 'DST', 'Bz_GSM', 'plasma_speed'])
    """
    setup_plot_style()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        columns = [col for col in columns if col in df.columns]
    
    corr_matrix = df[columns].corr(method=method)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1,
                center=0, square=True, linewidths=0.5,
                annot=annot, fmt='.2f', annot_kws={'size': 8},
                cbar_kws={'shrink': 0.8, 'label': f'{method.capitalize()} Correlation'},
                ax=ax)
    
    ax.set_title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved correlation heatmap to {save_path}")
    
    return fig


def plot_cross_correlation(series1: pd.Series,
                            series2: pd.Series,
                            name1: str = "Series 1",
                            name2: str = "Series 2",
                            max_lag: int = 24,
                            figsize: Tuple[int, int] = (12, 5),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot cross-correlation function between two time series.
    
    Parameters
    ----------
    series1 : pd.Series
        First time series (typically the predictor)
    series2 : pd.Series
        Second time series (typically the target)
    name1 : str, optional
        Name of first series
    name2 : str, optional
        Name of second series
    max_lag : int, optional
        Maximum lag to compute (default: 24)
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Example
    -------
    >>> fig = plot_cross_correlation(df['Bz_GSM'], df['DST'], 'Bz', 'DST')
    """
    setup_plot_style()
    
    # Align and clean both series
    df_aligned = pd.DataFrame({name1: series1, name2: series2}).dropna()
    s1 = df_aligned[name1].values
    s2 = df_aligned[name2].values
    
    # Standardize
    s1 = (s1 - s1.mean()) / s1.std()
    s2 = (s2 - s2.mean()) / s2.std()
    
    # Compute cross-correlation
    lags = range(-max_lag, max_lag + 1)
    ccf_values = []
    
    for lag in lags:
        if lag < 0:
            ccf = np.corrcoef(s1[:lag], s2[-lag:])[0, 1]
        elif lag > 0:
            ccf = np.corrcoef(s1[lag:], s2[:-lag])[0, 1]
        else:
            ccf = np.corrcoef(s1, s2)[0, 1]
        ccf_values.append(ccf)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    colors = [COLORS['primary'] if v >= 0 else COLORS['forecast'] for v in ccf_values]
    ax.bar(lags, ccf_values, color=colors, alpha=0.7, width=0.8)
    
    # Confidence bounds (95%)
    n = len(df_aligned)
    conf_bound = 1.96 / np.sqrt(n)
    ax.axhline(y=conf_bound, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=-conf_bound, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Find optimal lag
    max_ccf_idx = np.argmax(np.abs(ccf_values))
    optimal_lag = list(lags)[max_ccf_idx]
    max_ccf = ccf_values[max_ccf_idx]
    
    ax.annotate(f'Max CCF: {max_ccf:.3f} at lag {optimal_lag}',
                xy=(optimal_lag, max_ccf),
                xytext=(optimal_lag + 5, max_ccf + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    ax.set_xlabel(f'Lag (positive = {name1} leads {name2})')
    ax.set_ylabel('Cross-Correlation')
    ax.set_title(f'Cross-Correlation: {name1} vs {name2}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved cross-correlation plot to {save_path}")
    
    return fig


def plot_model_comparison(metrics_df: pd.DataFrame,
                          metric: str = 'RMSE',
                          title: str = "Model Comparison",
                          figsize: Tuple[int, int] = (10, 6),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a bar chart comparing model performance.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with models as index and metrics as columns
    metric : str, optional
        Metric to compare (default: 'RMSE')
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metrics_df)))
    
    bars = ax.bar(metrics_df.index, metrics_df[metric], color=colors, 
                  edgecolor='white', linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * metrics_df[metric].max(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric)
    ax.set_title(f'{title} - {metric}')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved model comparison plot to {save_path}")
    
    return fig


def plot_storm_event(df: pd.DataFrame,
                     event_date: str,
                     window_hours: int = 72,
                     variables: Optional[List[str]] = None,
                     figsize: Tuple[int, int] = (14, 12),
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a specific storm event with multiple variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    event_date : str
        Center date of the storm event (e.g., '2003-10-29')
    window_hours : int, optional
        Hours before and after event to display
    variables : List[str], optional
        Variables to plot
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    setup_plot_style()
    
    if variables is None:
        variables = ['Bz_GSM', 'plasma_speed', 'proton_density', 'DST', 'Kp', 'AE']
    
    from datetime import timedelta
    event_dt = pd.to_datetime(event_date)
    start = event_dt - timedelta(hours=window_hours)
    end = event_dt + timedelta(hours=window_hours)
    
    df_event = df[(df.index >= start) & (df.index <= end)]
    variables = [v for v in variables if v in df_event.columns]
    
    fig, axes = plt.subplots(len(variables), 1, figsize=figsize, sharex=True)
    if len(variables) == 1:
        axes = [axes]
    
    for ax, var in zip(axes, variables):
        ax.plot(df_event.index, df_event[var], linewidth=1.5)
        ax.axvline(x=event_dt, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    fig.suptitle(f'Storm Event: {event_date} (±{window_hours}h window)', 
                 fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved storm event plot to {save_path}")
    
    return fig


def save_all_plots(plots: Dict[str, plt.Figure],
                   output_dir: str,
                   format: str = 'png') -> List[str]:
    """
    Save multiple plots to a directory.
    
    Parameters
    ----------
    plots : Dict[str, plt.Figure]
        Dictionary mapping plot names to figure objects
    output_dir : str
        Directory to save plots
    format : str, optional
        Image format: 'png', 'pdf', 'svg' (default: 'png')
    
    Returns
    -------
    List[str]
        List of saved file paths
    
    Example
    -------
    >>> saved_files = save_all_plots({'timeseries': fig1, 'acf': fig2}, 'outputs/plots/')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    for name, fig in plots.items():
        filepath = os.path.join(output_dir, f"{name}.{format}")
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', format=format)
        saved_paths.append(filepath)
        print(f"Saved: {filepath}")
    
    return saved_paths


def create_summary_dashboard(df: pd.DataFrame,
                             target_var: str = 'Kp',
                             predictor_vars: Optional[List[str]] = None,
                             figsize: Tuple[int, int] = (16, 12),
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a summary dashboard with multiple visualizations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_var : str, optional
        Target variable to analyze
    predictor_vars : List[str], optional
        Predictor variables
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    setup_plot_style()
    
    if predictor_vars is None:
        predictor_vars = ['Bz_GSM', 'plasma_speed', 'proton_density']
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig)
    
    # Time series of target variable
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df[target_var], linewidth=0.5)
    ax1.set_ylabel(target_var)
    ax1.set_title(f'{target_var} Time Series')
    ax1.grid(True, alpha=0.3)
    
    # Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    df[target_var].dropna().hist(bins=50, ax=ax2, edgecolor='white')
    ax2.set_xlabel(target_var)
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution')
    
    # ACF
    ax3 = fig.add_subplot(gs[1, 1])
    plot_acf(df[target_var].dropna(), ax=ax3, lags=30, alpha=0.05)
    ax3.set_title('ACF')
    
    # PACF
    ax4 = fig.add_subplot(gs[1, 2])
    plot_pacf(df[target_var].dropna(), ax=ax4, lags=30, alpha=0.05, method='ywm')
    ax4.set_title('PACF')
    
    # Correlations with predictors
    ax5 = fig.add_subplot(gs[2, :])
    all_vars = [target_var] + [v for v in predictor_vars if v in df.columns]
    corr = df[all_vars].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax5, 
                fmt='.2f', square=True)
    ax5.set_title('Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved summary dashboard to {save_path}")
    
    return fig
