# OMNI Solar Wind & Space Weather Analysis

A two-stage pipeline for **solar flare forecasting** and **Earth space-weather impact analysis**:

1. **Data Pipeline** — downloads multi-source solar observation data (SDO, JSOC, LASP EVE, Fenyi Observatory)
2. **Analysis Pipeline** — processes OMNI2 solar wind data with time-series forecasting, correlation analysis, and visualization

---

## Table of Contents

- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [OMNI2 Dataset](#omni2-dataset)
- [Setup](#setup)
- [Usage](#usage)
  - [Data Pipeline](#data-pipeline)
  - [Analysis Pipeline](#analysis-pipeline)
- [Analysis Modules](#analysis-modules)
- [Outputs](#outputs)
- [AIA Wavelength Channels](#aia-wavelength-channels-solar-flare-relevance)
- [Rate Limiting](#rate-limiting)
- [Space Weather Application](#space-weather-application)

---

## Project Structure

```
OMNIProject/
├── config/
│   └── pipeline_config.yaml        # Source URLs, rate limits, wavelengths
├── src/
│   ├── config.py                   # Configuration loader
│   ├── http_client.py              # Rate-limited HTTP client (token bucket)
│   ├── fits_analyzer.py            # FITS analysis with astropy.io.fits
│   ├── pipeline.py                 # Data pipeline orchestrator
│   ├── extractors/
│   │   ├── base_extractor.py       # Abstract base with shared logic
│   │   ├── sdo_nasa_extractor.py
│   │   ├── jsoc_extractor.py
│   │   ├── eve_extractor.py
│   │   └── fenyi_extractor.py
│   └── analysis/
│       ├── data_preprocessing.py   # Load, clean, storm detection, stationarity
│       ├── univariate_models.py    # ETS, Croston, ARIMA, SARIMA, LSTM
│       ├── multivariate_models.py  # ARIMAX, SARIMAX, VAR
│       ├── correlation_analysis.py # Pearson, cross-correlation, lag analysis
│       ├── model_evaluation.py     # Metrics, residual analysis, Ljung-Box test
│       └── visualization.py        # Time series, ACF/PACF, forecasts, heatmaps
├── data/
│   ├── raw/                        # Downloaded files by source
│   ├── processed/                  # Analysis outputs, collated datasets
│   ├── metadata/                   # Per-source metadata (JSON + CSV)
│   └── fits_cache/                 # FITS file cache
├── logs/                           # Pipeline logs (rotating)
├── outputs/
│   ├── plots/                      # Generated figures
│   └── results/                    # CSVs and summary reports
├── run_pipeline.py                 # Data extraction CLI
├── run_analysis.py                 # Time series analysis CLI
├── requirements.txt
└── README.md
```

---

## Data Sources

| Source | URL | Data Type |
|--------|-----|-----------|
| **NASA SDO** | `https://sdo.gsfc.nasa.gov/data/` | AIA/HMI browse images (JPEG/FITS) |
| **JSOC Stanford** | `http://jsoc2.stanford.edu/data/aia/synoptic/` | AIA synoptic FITS (science-grade) |
| **LASP EVE** | `https://lasp.colorado.edu/eve/data_access/` | EUV irradiance (FITS/NetCDF/CSV) |
| **Fenyi Observatory** | `http://fenyi.solarobs.epss.hun-ren.hu/en/databases/SDO/` | Flare catalogs, SDO image archives |

### Coverage Period

- **December 2025** (2025-12-01 → 2025-12-31)
- **January 2026** (2026-01-01 → 2026-01-31)
- **February 2026** (2026-02-01 → 2026-02-28)

---

## OMNI2 Dataset

The analysis pipeline consumes **OMNI2** hourly mean values of interplanetary magnetic field (IMF) and solar wind plasma parameters measured by spacecraft near Earth's orbit, along with geomagnetic and solar activity indices.

Key parameter groups:

| Group | Parameters |
|-------|-----------|
| **IMF** | \|B\|, Bx/By/Bz (GSE & GSM), latitude/longitude angles |
| **Solar Wind** | Proton temperature, density, flow speed, flow angles, α/p ratio, flow pressure |
| **Derived** | Electric field, plasma beta, Alfvén Mach number, magnetosonic Mach number |
| **Geomagnetic** | Kp, Dst, AE, AL, AU, ap |
| **Solar Activity** | Sunspot number (v2), F10.7 index, PC(N) index |
| **Energetic Protons** | Fluxes > 1, 2, 4, 10, 30, 60 MeV |

Each of the following geomagnetic indices is an **independent forecasting target** (models are trained and evaluated separately for each):

| Target | Description |
|--------|-------------|
| **Kp** | Planetary geomagnetic activity index (0–9 scale) |
| **Dst** | Disturbance Storm Time index — measures ring-current intensity (nT) |
| **AE** | Auroral Electrojet index — captures high-latitude current activity (nT) |

Solar wind predictor variables used in multivariate models: `Bz_GSM`, `Bz_GSE`, `plasma_speed`, `proton_density`, `flow_pressure`, `B_mag_avg`, `electric_field`, `plasma_beta`, `alfven_mach`

---

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install all dependencies
pip install -r requirements.txt
```

> **Note:** TensorFlow (listed in `requirements.txt`) is only required for LSTM forecasting. The analysis pipeline degrades gracefully and skips LSTM if TensorFlow is not installed.

---

## Usage

### Data Pipeline

Download solar observation data from all configured sources and run FITS analysis:

```bash
# Full pipeline (extract + FITS analysis)
python run_pipeline.py

# Download only (skip FITS analysis)
python run_pipeline.py --skip-analysis

# Analyse already-downloaded data only
python run_pipeline.py --skip-extract

# Inspect a single FITS file
python run_pipeline.py --analyze path/to/file.fits

# Use a custom config file
python run_pipeline.py --config path/to/custom_config.yaml
```

### Analysis Pipeline

Run the full time-series analysis on the OMNI2 dataset:

```bash
# Default run (reads omni2_full_dataset.csv, writes to outputs/)
python run_analysis.py

# Custom config file
python run_analysis.py --config config/analysis_config.yaml
```

The analysis pipeline runs five sequential steps:

| Step | Description |
|------|-------------|
| 1 | **Data Preparation** — load, clean, storm detection, resampling, stationarity tests |
| 2 | **Univariate Forecasting** — ETS, Croston, auto-ARIMA, SARIMA, LSTM |
| 3 | **Multivariate Modeling** — ARIMAX, SARIMAX, VAR |
| 4 | **Correlation Analysis** — Pearson, cross-correlation, lag identification |
| 5 | **Visualization & Reporting** — all plots + summary dashboard |

---

## Analysis Modules

### `src/analysis/data_preprocessing.py`

- Load OMNI2 CSV and replace fill values with `NaN`
- Clean and interpolate missing data
- Detect geomagnetic storm events using Dst threshold
- Resample to daily or custom frequency
- Normalize/standardize features
- ADF and KPSS stationarity tests

### `src/analysis/univariate_models.py`

| Model | Notes |
|-------|-------|
| **ETS** (Exponential Smoothing) | Holt-Winters additive/multiplicative |
| **Croston** | For intermittent / sparse series |
| **Auto-ARIMA** | Order selection via `pmdarima` |
| **SARIMA** | Seasonal extension with configurable period |
| **LSTM** | Sequence-to-one deep learning (TensorFlow/Keras, optional) |

### `src/analysis/multivariate_models.py`

| Model | Notes |
|-------|-------|
| **ARIMAX** | ARIMA with exogenous solar wind predictors |
| **SARIMAX** | Seasonal ARIMAX |
| **VAR** | Vector Autoregression for joint forecasting of multiple indices |

### `src/analysis/correlation_analysis.py`

- Pearson correlation matrix across all variables
- Cross-correlation functions between solar wind drivers and geomagnetic indices
- Lag relationship identification
- Storm predictor analysis

### `src/analysis/model_evaluation.py`

- Standard forecast metrics: MAE, RMSE, MAPE, R²
- Residual diagnostics (ACF of residuals, normality tests)
- Ljung-Box portmanteau test for serial correlation
- Comparative evaluation report across all fitted models

### `src/analysis/visualization.py`

- Time series plots (with optional date range filtering)
- ACF / PACF plots
- Forecast vs. actual overlays
- Residual diagnostic panels
- Correlation heatmaps
- Cross-correlation plots
- Model comparison bar charts
- Summary dashboard (multi-panel figure)
- Storm event highlights
- Batch export of all figures

---

## Outputs

### Data Pipeline

| File | Description |
|------|-------------|
| `data/processed/collated_dataset.csv` | Merged metadata + FITS analysis |
| `data/processed/all_sources_metadata.csv` | Download metadata from all sources |
| `data/processed/fits_analysis.csv` | FITS header & image statistics |
| `data/processed/pipeline_summary.txt` | Human-readable run summary |
| `data/metadata/<source>_metadata.json` | Per-source metadata (JSON) |
| `logs/pipeline.log` | Detailed pipeline log |

### Analysis Pipeline

| File | Description |
|------|-------------|
| `outputs/results/statistical_properties.csv` | Descriptive stats for all variables |
| `outputs/results/forecast_<model>_<var>.csv` | Forecast series per model and variable |
| `outputs/results/model_evaluation.csv` | Comparative metrics table |
| `outputs/results/correlation_summary.csv` | Pearson correlations |
| `outputs/plots/*.png` | All generated figures |

---

## AIA Wavelength Channels (Solar Flare Relevance)

| Channel (Å) | Ion | Temperature | Flare Relevance |
|-------------|-----|------------|-----------------|
| 94 | Fe XVIII | ~6.3 MK | Hot flare plasma |
| 131 | Fe VIII/XXI | ~0.4/10 MK | Flare / transition region |
| 171 | Fe IX | ~0.6 MK | Quiet corona baseline |
| 193 | Fe XII/XXIV | ~1.2/20 MK | Corona + hot flare plasma |
| 211 | Fe XIV | ~2.0 MK | Active region corona |
| 304 | He II | ~0.05 MK | Chromosphere / TR |
| 335 | Fe XVI | ~2.5 MK | Active region |
| 1600 | C IV + cont. | – | Transition region / photosphere |
| 1700 | Continuum | – | Photosphere |

---

## Rate Limiting

Each source has independent rate limits configured in `pipeline_config.yaml`:

| Source | Requests/sec | Burst | Retry | Backoff |
|--------|-------------|-------|-------|---------|
| SDO NASA | 2.0 | 5 | 3 | 2.0× |
| JSOC Stanford | 1.0 | 3 | 3 | 2.0× |
| LASP EVE | 1.0 | 2 | 3 | 3.0× |
| Fenyi Observatory | 0.5 | 2 | 3 | 3.0× |

The pipeline uses a **token-bucket** algorithm with exponential backoff retries. Already-downloaded files are automatically skipped (SHA-256 verified).

### FITS Analysis

Uses `astropy.io.fits` to extract:

- **Header metadata**: observation time, wavelength, instrument, pixel scale, solar radius, Carrington coordinates, data quality flags, etc.
- **Image statistics**: mean, std, min, max, median, 99th percentile
- **HDU inventory**: type, shape, dtype for each Header Data Unit

---

## Space Weather Application

This project collects and analyses data suitable for:

- **Solar flare prediction** using multi-wavelength AIA imagery
- **Geomagnetic storm forecasting** using Kp, Dst, and AE indices
- **EUV irradiance monitoring** (EVE) for ionospheric impact assessment
- **Magnetic field analysis** (HMI magnetograms) for active region evolution
- **Flare event labeling** from Fenyi observatory catalogs
- **Satellite drag and communication disruption** forecasting

---

*Data acknowledgment: OMNI2 data are provided by the NASA Space Physics Data Facility (SPDF). Contact: [Natalia.E.Papitashvili@nasa.gov](mailto:Natalia.E.Papitashvili@nasa.gov)*

