"""
Pipeline Orchestrator
=====================
Coordinates the full extraction → analysis → collation workflow:

1. Initialize logging and configuration
2. Run each enabled extractor (SDO, JSOC, EVE, Fenyi)
3. Analyse all downloaded FITS files with astropy
4. Collate metadata across all sources into a unified dataset
5. Save consolidated outputs for downstream forecasting models
"""

import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from typing import Any

import pandas as pd

from src.config import PipelineConfig
from src.extractors.sdo_nasa_extractor import SDONasaExtractor
from src.extractors.jsoc_extractor import JSOCExtractor
from src.extractors.eve_extractor import EVEExtractor
from src.extractors.fenyi_extractor import FenyiExtractor
from src.fits_analyzer import FITSAnalyzer

logger = logging.getLogger("pipeline")

# Extractor registry: (key_in_config, class)
_EXTRACTORS = [
    ("sdo_nasa", SDONasaExtractor),
    ("jsoc_stanford", JSOCExtractor),
    ("lasp_eve", EVEExtractor),
    ("fenyi_observatory", FenyiExtractor),
]


class SolarDataPipeline:
    """End-to-end solar data extraction and analysis pipeline."""

    def __init__(self, config_path: str | None = None):
        self.config = PipelineConfig(config_path)
        self._setup_logging()
        self.all_metadata: list[dict] = []

    # ================================================================== #
    #  Logging
    # ================================================================== #

    def _setup_logging(self):
        log_cfg = self.config.logging_config
        log_file = log_cfg.get("file", "logs/pipeline.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        root = logging.getLogger()
        root.setLevel(getattr(logging, log_cfg.get("level", "INFO")))

        fmt = logging.Formatter(log_cfg.get(
            "format",
            "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        ))

        # Rotating file handler
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_cfg.get("max_bytes", 10_485_760),
            backupCount=log_cfg.get("backup_count", 5),
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # ================================================================== #
    #  Run the full pipeline
    # ================================================================== #

    def run(self, *, skip_extraction: bool = False, skip_analysis: bool = False):
        """
        Execute the complete pipeline.

        Parameters
        ----------
        skip_extraction : bool
            If True, skip downloading and jump to FITS analysis
            (useful when data is already downloaded).
        skip_analysis : bool
            If True, skip the FITS analysis step.
        """
        t0 = time.monotonic()
        logger.info("=" * 70)
        logger.info("OMNI Solar Data Pipeline — run started at %s",
                     datetime.utcnow().isoformat())
        logger.info("Date ranges: %s", self.config.date_ranges)
        logger.info("=" * 70)

        # ---- Step 1: Extraction ---------------------------------------- #
        if not skip_extraction:
            self._run_extractors()
        else:
            logger.info("Extraction skipped (skip_extraction=True)")

        # ---- Step 2: FITS Analysis ------------------------------------- #
        fits_df = pd.DataFrame()
        if not skip_analysis:
            fits_df = self._run_fits_analysis()
        else:
            logger.info("FITS analysis skipped (skip_analysis=True)")

        # ---- Step 3: Collation ----------------------------------------- #
        self._collate_all(fits_df)

        elapsed = time.monotonic() - t0
        logger.info("=" * 70)
        logger.info("Pipeline complete in %.1f seconds", elapsed)
        logger.info("Total metadata records: %d", len(self.all_metadata))
        logger.info("=" * 70)

    # ================================================================== #
    #  Step 1 — Run extractors
    # ================================================================== #

    def _run_extractors(self):
        """Run each enabled extractor in sequence."""
        for key, cls in _EXTRACTORS:
            src_cfg = self.config.source(key)
            if not src_cfg.get("enabled", False):
                logger.info("Source '%s' is disabled — skipping", key)
                continue

            logger.info("-" * 50)
            logger.info("Starting extractor: %s", key)
            logger.info("-" * 50)

            try:
                with cls(self.config) as extractor:
                    records = extractor.extract()
                    self.all_metadata.extend(records)
                    logger.info("Extractor '%s' returned %d records", key, len(records))
            except Exception as exc:
                logger.error("Extractor '%s' failed: %s", key, exc, exc_info=True)

    # ================================================================== #
    #  Step 2 — FITS analysis
    # ================================================================== #

    def _run_fits_analysis(self) -> pd.DataFrame:
        """Analyse all FITS files found in the raw data directory."""
        logger.info("-" * 50)
        logger.info("Running FITS analysis with astropy.io.fits")
        logger.info("-" * 50)

        analyzer = FITSAnalyzer(self.config)
        df = analyzer.analyze_all()

        if not df.empty:
            analyzer.save_analysis(df)
            logger.info("FITS analysis: %d files processed", len(df))
        else:
            logger.info("No FITS files found for analysis")

        return df

    # ================================================================== #
    #  Step 3 — Collation
    # ================================================================== #

    def _collate_all(self, fits_df: pd.DataFrame):
        """
        Merge extraction metadata with FITS analysis results into a
        single consolidated dataset for downstream modelling.
        """
        logger.info("-" * 50)
        logger.info("Collating data across all sources")
        logger.info("-" * 50)

        out_dir = self.config.storage.get("processed_dir", "data/processed")
        os.makedirs(out_dir, exist_ok=True)

        # ---- Extraction metadata --------------------------------------- #
        if self.all_metadata:
            meta_df = pd.json_normalize(self.all_metadata)
            meta_csv = os.path.join(out_dir, "all_sources_metadata.csv")
            meta_json = os.path.join(out_dir, "all_sources_metadata.json")
            meta_df.to_csv(meta_csv, index=False)
            with open(meta_json, "w", encoding="utf-8") as f:
                json.dump(self.all_metadata, f, indent=2, default=str)
            logger.info("Collated metadata: %d records -> %s", len(meta_df), meta_csv)
        else:
            meta_df = pd.DataFrame()
            logger.warning("No extraction metadata to collate")

        # ---- Merge with FITS analysis ---------------------------------- #
        if not fits_df.empty and not meta_df.empty:
            # Join on filename where possible
            if "filename" in meta_df.columns and "filename" in fits_df.columns:
                merged = meta_df.merge(fits_df, on="filename", how="outer",
                                       suffixes=("_meta", "_fits"))
            else:
                merged = pd.concat([meta_df, fits_df], ignore_index=True)

            merged_csv = os.path.join(out_dir, "collated_dataset.csv")
            merged.to_csv(merged_csv, index=False)
            logger.info("Collated dataset: %d rows, %d columns -> %s",
                         len(merged), len(merged.columns), merged_csv)

        # ---- Summary report -------------------------------------------- #
        self._write_summary(out_dir, meta_df, fits_df)

    def _write_summary(self, out_dir: str, meta_df: pd.DataFrame,
                       fits_df: pd.DataFrame):
        """Write a human-readable summary of the pipeline run."""
        summary_path = os.path.join(out_dir, "pipeline_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("OMNI Solar Data Pipeline — Run Summary\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
            f.write("=" * 60 + "\n\n")

            f.write("Date Ranges:\n")
            for dr in self.config.date_ranges:
                f.write(f"  {dr['label']}: {dr['start']} to {dr['end']}\n")
            f.write("\n")

            if not meta_df.empty:
                f.write(f"Total downloaded artefacts: {len(meta_df)}\n")
                if "source" in meta_df.columns:
                    f.write("\nBy source:\n")
                    for src, cnt in meta_df["source"].value_counts().items():
                        f.write(f"  {src}: {cnt}\n")
                if "format" in meta_df.columns:
                    f.write("\nBy format:\n")
                    for fmt, cnt in meta_df["format"].value_counts().items():
                        f.write(f"  {fmt}: {cnt}\n")
            else:
                f.write("No artefacts downloaded.\n")

            f.write(f"\nFITS files analysed: {len(fits_df)}\n")

            if not fits_df.empty and "hdr_WAVELNTH" in fits_df.columns:
                f.write("\nFITS wavelengths covered:\n")
                for wl in sorted(fits_df["hdr_WAVELNTH"].dropna().unique()):
                    f.write(f"  {wl} Å\n")

        logger.info("Summary written to %s", summary_path)
