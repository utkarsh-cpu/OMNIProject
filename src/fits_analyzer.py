"""
FITS File Analyzer
==================
Uses astropy.io.fits to open, inspect, and extract metadata and statistics
from downloaded FITS files.  Produces a consolidated analysis dataframe
suitable for downstream forecasting models.

Key capabilities:
  - Header keyword extraction (observation time, wavelength, instrument, etc.)
  - Image data statistics (mean, std, min, max, percentiles)
  - Batch processing of all FITS files in the raw data directory
  - Export of collated analysis to CSV / JSON for model ingestion
"""

import glob
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits as astropy_fits

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


class FITSAnalyzer:
    """Analyze FITS files using astropy.io.fits."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.header_keywords = config.fits_analysis.get("header_keywords", [])
        self.compute_stats = config.fits_analysis.get("compute_statistics", True)
        self.stat_names = config.fits_analysis.get("statistics", [])
        self.storage = config.storage

    # ================================================================== #
    #  Single-file analysis
    # ================================================================== #

    def analyze_file(self, fits_path: str) -> dict[str, Any]:
        """
        Open a FITS file, extract header metadata and image statistics.

        Parameters
        ----------
        fits_path : str
            Path to a .fits or .fits.gz file.

        Returns
        -------
        dict with keys: file, headers, stats, hdu_info, errors
        """
        result: dict[str, Any] = {
            "file": fits_path,
            "filename": os.path.basename(fits_path),
            "headers": {},
            "stats": {},
            "hdu_info": [],
            "errors": [],
        }

        try:
            with astropy_fits.open(fits_path, memmap=True) as hdul:
                result["num_hdus"] = len(hdul)

                # ---- HDU summary --------------------------------------- #
                for i, hdu in enumerate(hdul):
                    info = {
                        "index": i,
                        "name": hdu.name,
                        "type": type(hdu).__name__,
                    }
                    if hdu.data is not None:
                        info["shape"] = list(hdu.data.shape)
                        info["dtype"] = str(hdu.data.dtype)
                    result["hdu_info"].append(info)

                # ---- Header keywords from primary HDU ------------------ #
                primary = hdul[0]
                header = primary.header
                for kw in self.header_keywords:
                    try:
                        val = header.get(kw)
                        if val is not None:
                            result["headers"][kw] = _serialisable(val)
                    except Exception:
                        pass

                # Also capture HISTORY and COMMENT counts
                result["headers"]["_HISTORY_COUNT"] = len(header.get("HISTORY", []))
                result["headers"]["_COMMENT_COUNT"] = len(header.get("COMMENT", []))

                # ---- Image statistics ---------------------------------- #
                if self.compute_stats and primary.data is not None:
                    result["stats"] = self._compute_image_stats(primary.data)

                # If primary has no data, try the first ImageHDU
                if primary.data is None:
                    for hdu in hdul[1:]:
                        if isinstance(hdu, (astropy_fits.ImageHDU,
                                            astropy_fits.CompImageHDU)):
                            if hdu.data is not None:
                                # Grab headers from this HDU too
                                for kw in self.header_keywords:
                                    if kw not in result["headers"]:
                                        val = hdu.header.get(kw)
                                        if val is not None:
                                            result["headers"][kw] = _serialisable(val)
                                if self.compute_stats:
                                    result["stats"] = self._compute_image_stats(
                                        hdu.data
                                    )
                                break

        except Exception as exc:
            logger.error("FITS analysis failed for %s: %s", fits_path, exc)
            result["errors"].append(str(exc))

        return result

    # ================================================================== #
    #  Batch analysis
    # ================================================================== #

    def analyze_all(self, root_dir: str | None = None) -> pd.DataFrame:
        """
        Recursively find and analyse every FITS file under *root_dir*
        (defaults to the raw data directory).

        Returns a DataFrame with one row per file.
        """
        root = root_dir or self.storage.get("raw_dir", "data/raw")
        patterns = [
            os.path.join(root, "**", "*.fits"),
            os.path.join(root, "**", "*.fits.gz"),
            os.path.join(root, "**", "*.fit"),
        ]
        fits_files: list[str] = []
        for pat in patterns:
            fits_files.extend(glob.glob(pat, recursive=True))

        fits_files = sorted(set(fits_files))
        logger.info("Found %d FITS files to analyse under %s", len(fits_files), root)

        records = []
        for fpath in fits_files:
            analysis = self.analyze_file(fpath)
            flat = self._flatten(analysis)
            records.append(flat)

        if not records:
            logger.warning("No FITS files found for analysis")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logger.info("FITS analysis complete: %d files, %d columns",
                     len(df), len(df.columns))
        return df

    def save_analysis(self, df: pd.DataFrame, tag: str = "fits_analysis"):
        """Persist the analysis DataFrame to CSV and JSON."""
        if df.empty:
            logger.warning("Empty analysis DataFrame — nothing to save")
            return

        out_dir = self.storage.get("processed_dir", "data/processed")
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, f"{tag}.csv")
        json_path = os.path.join(out_dir, f"{tag}.json")

        df.to_csv(csv_path, index=False)
        logger.info("FITS analysis CSV: %s", csv_path)

        df.to_json(json_path, orient="records", indent=2, default_handler=str)
        logger.info("FITS analysis JSON: %s", json_path)

    # ================================================================== #
    #  Pretty-print a single FITS file (interactive use)
    # ================================================================== #

    @staticmethod
    def print_info(fits_path: str):
        """Print a human-readable summary of a FITS file."""
        with astropy_fits.open(fits_path, memmap=True) as hdul:
            print(f"\n{'='*60}")
            print(f"FITS File: {fits_path}")
            print(f"{'='*60}")
            hdul.info()

            for i, hdu in enumerate(hdul):
                print(f"\n--- HDU {i}: {hdu.name} ({type(hdu).__name__}) ---")
                header = hdu.header
                for card in header.cards:
                    if card.keyword and card.keyword not in ("HISTORY", "COMMENT", ""):
                        print(f"  {card.keyword:20s} = {card.value!r}")

                if hdu.data is not None:
                    data = np.asarray(hdu.data, dtype=np.float64)
                    finite = data[np.isfinite(data)]
                    if finite.size > 0:
                        print(f"\n  Data shape : {hdu.data.shape}")
                        print(f"  Data dtype : {hdu.data.dtype}")
                        print(f"  Mean       : {finite.mean():.4f}")
                        print(f"  Std        : {finite.std():.4f}")
                        print(f"  Min        : {finite.min():.4f}")
                        print(f"  Max        : {finite.max():.4f}")
                        print(f"  Median     : {np.median(finite):.4f}")
                        print(f"  P99        : {np.percentile(finite, 99):.4f}")
            print(f"{'='*60}\n")

    # ================================================================== #
    #  Internal helpers
    # ================================================================== #

    def _compute_image_stats(self, data: np.ndarray) -> dict[str, float]:
        """Compute configured statistics on image data."""
        arr = np.asarray(data, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {}

        stat_map = {
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "median": float(np.median(finite)),
            "percentile_99": float(np.percentile(finite, 99)),
        }

        if self.stat_names:
            return {k: v for k, v in stat_map.items() if k in self.stat_names}
        return stat_map

    @staticmethod
    def _flatten(analysis: dict) -> dict:
        """Flatten nested analysis dict into a single-level dict for DataFrame."""
        flat: dict[str, Any] = {
            "file": analysis["file"],
            "filename": analysis["filename"],
            "num_hdus": analysis.get("num_hdus", 0),
            "errors": "; ".join(analysis.get("errors", [])),
        }
        for k, v in analysis.get("headers", {}).items():
            flat[f"hdr_{k}"] = v
        for k, v in analysis.get("stats", {}).items():
            flat[f"stat_{k}"] = v
        if analysis.get("hdu_info"):
            flat["primary_shape"] = str(
                analysis["hdu_info"][0].get("shape", "")
            )
        return flat


# ===================================================================== #
#  Module-level helpers
# ===================================================================== #

def _serialisable(val: Any) -> Any:
    """Ensure a FITS header value is JSON-serialisable."""
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    return str(val)
