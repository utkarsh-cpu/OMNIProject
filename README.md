# OMNI Solar Flare Data Pipeline

A data pipeline for extracting solar observation data from multiple sources to support **solar flare forecasting** and **Earth space-weather impact analysis** (ionospheric effects, satellite system disruption).

## Data Sources

| Source | URL | Data Type |
|--------|-----|-----------|
| **NASA SDO** | `https://sdo.gsfc.nasa.gov/data/` | AIA/HMI browse images (JPEG/FITS) |
| **JSOC Stanford** | `http://jsoc2.stanford.edu/data/aia/synoptic/` | AIA synoptic FITS (science-grade) |
| **LASP EVE** | `https://lasp.colorado.edu/eve/data_access/` | EUV irradiance (FITS/NetCDF/CSV) |
| **Fenyi Observatory** | `http://fenyi.solarobs.epss.hun-ren.hu/en/databases/SDO/` | Flare catalogs, SDO image archives |

## Coverage Period

- **December 2025** (2025-12-01 → 2025-12-31)
- **January 2026** (2026-01-01 → 2026-01-31)
- **February 2026** (2026-02-01 → 2026-02-28)

## Project Structure

```
OMNIProject/
├── config/
│   └── pipeline_config.yaml    # All source URLs, rate limits, wavelengths
├── src/
│   ├── config.py               # Configuration loader
│   ├── http_client.py          # Rate-limited HTTP client (token bucket)
│   ├── fits_analyzer.py        # FITS analysis with astropy.io.fits
│   ├── pipeline.py             # Main orchestrator
│   └── extractors/
│       ├── base_extractor.py   # Abstract base with shared logic
│       ├── sdo_nasa_extractor.py
│       ├── jsoc_extractor.py
│       ├── eve_extractor.py
│       └── fenyi_extractor.py
├── data/
│   ├── raw/                    # Downloaded files by source
│   ├── processed/              # Analysis outputs, collated datasets
│   ├── metadata/               # Per-source metadata (JSON + CSV)
│   └── fits_cache/             # FITS file cache
├── logs/                       # Pipeline logs (rotating)
├── run_pipeline.py             # CLI entry point
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Full pipeline (extract + analyse)
```bash
python run_pipeline.py
```

### Download only (skip FITS analysis)
```bash
python run_pipeline.py --skip-analysis
```

### Analyse only (data already downloaded)
```bash
python run_pipeline.py --skip-extract
```

### Inspect a single FITS file
```bash
python run_pipeline.py --analyze path/to/file.fits
```

### Custom config
```bash
python run_pipeline.py --config path/to/custom_config.yaml
```

## Rate Limiting

Each source has independent rate limits configured in `pipeline_config.yaml`:

| Source | Requests/sec | Burst | Retry | Backoff |
|--------|-------------|-------|-------|---------|
| SDO NASA | 2.0 | 5 | 3 | 2.0× |
| JSOC Stanford | 1.0 | 3 | 3 | 2.0× |
| LASP EVE | 1.0 | 2 | 3 | 3.0× |
| Fenyi Observatory | 0.5 | 2 | 3 | 3.0× |

The pipeline uses a **token-bucket** algorithm with exponential backoff retries. Already-downloaded files are automatically skipped (SHA-256 verified).

## FITS Analysis

Uses `astropy.io.fits` to extract:

- **Header metadata**: observation time, wavelength, instrument, pixel scale, solar radius, Carrington coordinates, data quality flags, etc.
- **Image statistics**: mean, std, min, max, median, 99th percentile
- **HDU inventory**: type, shape, dtype for each Header Data Unit

## Outputs

After a successful run, find:

| File | Description |
|------|-------------|
| `data/processed/collated_dataset.csv` | Merged metadata + FITS analysis |
| `data/processed/all_sources_metadata.csv` | Download metadata from all sources |
| `data/processed/fits_analysis.csv` | FITS header & image statistics |
| `data/processed/pipeline_summary.txt` | Human-readable run summary |
| `data/metadata/<source>_metadata.json` | Per-source metadata (JSON) |
| `logs/pipeline.log` | Detailed pipeline log |

## AIA Wavelength Channels (Solar Flare Relevance)

| Channel (Å) | Ion | Temperature | Flare Relevance |
|-------------|-----|------------|-----------------|
| 94 | Fe XVIII | ~6.3 MK | Hot flare plasma |
| 131 | Fe VIII/XXI | ~0.4/10 MK | Flare/transition region |
| 171 | Fe IX | ~0.6 MK | Quiet corona baseline |
| 193 | Fe XII/XXIV | ~1.2/20 MK | Corona + hot flare plasma |
| 211 | Fe XIV | ~2.0 MK | Active region corona |
| 304 | He II | ~0.05 MK | Chromosphere/TR |
| 335 | Fe XVI | ~2.5 MK | Active region |
| 1600 | C IV + cont. | – | Transition region/photosphere |
| 1700 | Continuum | – | Photosphere |

## Space Weather Application

This pipeline collects data suitable for:
- **Solar flare prediction** using multi-wavelength AIA imagery
- **EUV irradiance monitoring** (EVE) for ionospheric impact assessment
- **Magnetic field analysis** (HMI magnetograms) for active region evolution
- **Flare event labeling** from Fenyi observatory catalogs
- **Satellite drag and communication disruption** forecasting
