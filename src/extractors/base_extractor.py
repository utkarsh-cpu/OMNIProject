"""
Base extractor providing shared logic for all data source extractors.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.config import PipelineConfig
from src.http_client import RateLimitedClient, build_client

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Abstract base class for all solar data extractors."""

    # Subclasses must set this to the key under `sources:` in config
    SOURCE_KEY: str = ""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.source_cfg: dict[str, Any] = config.source(self.SOURCE_KEY)
        self.client: RateLimitedClient = build_client(self.source_cfg)
        self.base_url: str = self.source_cfg.get("base_url", "")
        self.storage = config.storage
        self._metadata_records: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Date iteration helpers
    # ------------------------------------------------------------------ #

    def iter_target_dates(self):
        """Yield every date covered by the configured date_ranges."""
        for dr in self.config.date_ranges:
            d = dr["start"]
            while d <= dr["end"]:
                yield d
                d += timedelta(days=1)

    # ------------------------------------------------------------------ #
    #  Metadata bookkeeping
    # ------------------------------------------------------------------ #

    def record_metadata(self, entry: dict):
        """Append a metadata record for later persistence."""
        self._metadata_records.append(entry)

    def save_metadata(self, filename: str | None = None):
        """Write collected metadata to a JSON + CSV in the metadata dir."""
        if not self._metadata_records:
            logger.warning("No metadata records to save for %s", self.SOURCE_KEY)
            return

        fname = filename or f"{self.SOURCE_KEY}_metadata"
        meta_dir = self.storage.get("metadata_dir", "data/metadata")
        os.makedirs(meta_dir, exist_ok=True)

        json_path = os.path.join(meta_dir, f"{fname}.json")
        csv_path = os.path.join(meta_dir, f"{fname}.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata_records, f, indent=2, default=str)
        logger.info("Metadata JSON saved: %s  (%d records)",
                     json_path, len(self._metadata_records))

        df = pd.json_normalize(self._metadata_records)
        df.to_csv(csv_path, index=False)
        logger.info("Metadata CSV saved: %s", csv_path)

    # ------------------------------------------------------------------ #
    #  Destination path builder
    # ------------------------------------------------------------------ #

    def dest_path(self, sub_dir: str, filename: str) -> str:
        """Return full path under raw_dir/{SOURCE_KEY}/{sub_dir}/{filename}."""
        p = os.path.join(
            self.storage.get("raw_dir", "data/raw"),
            self.SOURCE_KEY,
            sub_dir,
            filename,
        )
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    # ------------------------------------------------------------------ #
    #  Abstract contract
    # ------------------------------------------------------------------ #

    @abstractmethod
    def extract(self) -> list[dict]:
        """
        Run the full extraction for all configured date ranges.
        Returns list of metadata dicts for every downloaded artefact.
        """
        ...

    # ------------------------------------------------------------------ #
    #  Context manager
    # ------------------------------------------------------------------ #

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
