"""
Configuration loader for the Solar Data Pipeline.
Reads pipeline_config.yaml and provides typed access to settings.
"""

import os
import yaml
from datetime import datetime, date
from typing import Any


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG = os.path.join(_BASE_DIR, "config", "pipeline_config.yaml")


class PipelineConfig:
    """Loads and exposes pipeline configuration."""

    def __init__(self, config_path: str | None = None):
        path = config_path or _DEFAULT_CONFIG
        with open(path, "r", encoding="utf-8") as f:
            self._raw: dict[str, Any] = yaml.safe_load(f)
        self._ensure_directories()

    # -- helpers ---------------------------------------------------------- #
    def _ensure_directories(self):
        storage = self._raw.get("storage", {})
        for key in ("raw_dir", "processed_dir", "metadata_dir",
                     "fits_cache_dir", "logs_dir"):
            d = os.path.join(_BASE_DIR, storage.get(key, f"data/{key}"))
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def _parse_date(s: str) -> date:
        return datetime.strptime(s, "%Y-%m-%d").date()

    # -- public api ------------------------------------------------------- #
    @property
    def base_dir(self) -> str:
        return _BASE_DIR

    @property
    def date_ranges(self) -> list[dict]:
        """Return list of {start: date, end: date, label: str}."""
        out = []
        for dr in self._raw.get("date_ranges", []):
            out.append({
                "start": self._parse_date(dr["start"]),
                "end": self._parse_date(dr["end"]),
                "label": dr.get("label", ""),
            })
        return out

    def source(self, name: str) -> dict[str, Any]:
        return self._raw.get("sources", {}).get(name, {})

    @property
    def sources(self) -> dict[str, dict]:
        return self._raw.get("sources", {})

    @property
    def storage(self) -> dict[str, str]:
        storage = self._raw.get("storage", {})
        return {k: os.path.join(_BASE_DIR, v) for k, v in storage.items()}

    @property
    def fits_analysis(self) -> dict[str, Any]:
        return self._raw.get("fits_analysis", {})

    @property
    def logging_config(self) -> dict[str, Any]:
        cfg = self._raw.get("logging", {})
        if "file" in cfg:
            cfg["file"] = os.path.join(_BASE_DIR, cfg["file"])
        return cfg
