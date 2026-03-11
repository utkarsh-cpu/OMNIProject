"""
LASP EVE (EUV Variability Experiment) Data Extractor
=====================================================
Downloads solar EUV irradiance data from:
  https://lasp.colorado.edu/eve/data_access/evewebdata/interactive/

EVE measures the solar extreme-ultraviolet irradiance that directly drives
ionospheric variability and affects satellite drag/communications.

Data products:
  - ESP (EUV SpectroPhotometer): 0.1-7 nm quad-diode data
  - MEGS-A: spectra 6-37 nm
  - MEGS-B: spectra 37-106 nm
  - Extracted emission lines
  - Merged daily spectra (Level 2/3)

Directory layout (typical):
  .../evewebdata/interactive/download/level2/esp/{YYYY}/{DDD}/  (DOY-based)
  or
  .../evewebdata/interactive/plots/{YYYY}/{MM}/

The files may be FITS, NetCDF, or CSV depending on the product.
"""

import logging
import re
from datetime import date
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class EVEExtractor(BaseExtractor):
    SOURCE_KEY = "lasp_eve"

    # Known directory patterns for EVE web data
    _DOWNLOAD_BASE = "https://lasp.colorado.edu/eve/data_access/evewebdata/interactive/"

    def extract(self) -> list[dict]:
        logger.info("=== LASP EVE extraction starting ===")
        results: list[dict] = []
        products = self.source_cfg.get("products", [])

        for product in products:
            product_results = self._extract_product(product)
            results.extend(product_results)

        # Merged daily spectra
        if self.source_cfg.get("merged_spectra"):
            results.extend(self._extract_merged_spectra())

        self.save_metadata()
        logger.info("=== EVE extraction complete: %d files ===", len(results))
        return results

    # ------------------------------------------------------------------ #
    #  Product-level extraction
    # ------------------------------------------------------------------ #

    def _extract_product(self, product_cfg: dict) -> list[dict]:
        """Extract all dates for a single EVE data product."""
        name = product_cfg["name"]
        level = product_cfg.get("level", "level2")
        results = []

        for d in self.iter_target_dates():
            doy = d.timetuple().tm_yday
            year = d.strftime("%Y")
            doy_str = f"{doy:03d}"

            # Try common EVE directory patterns
            urls_to_try = [
                # DOY-based layout
                f"{self._DOWNLOAD_BASE}download/{level}/{name}/{year}/{doy_str}/",
                # Month-based layout
                f"{self._DOWNLOAD_BASE}download/{level}/{name}/{year}/{d.strftime('%m')}/",
                # Flat year layout
                f"{self._DOWNLOAD_BASE}download/{level}/{name}/{year}/",
            ]

            for dir_url in urls_to_try:
                files = self._list_data_files(dir_url, d)
                if files:
                    for fname in files:
                        file_url = urljoin(dir_url, fname)
                        sub = f"{name}/{year}/{d.strftime('%m')}/{d.strftime('%d')}"
                        dest = self.dest_path(sub, fname)
                        try:
                            meta = self.client.download_file(file_url, dest)
                            meta.update({
                                "source": self.SOURCE_KEY,
                                "product": name,
                                "level": level,
                                "date": str(d),
                                "doy": doy,
                                "filename": fname,
                                "description": product_cfg.get("description", ""),
                            })
                            self.record_metadata(meta)
                            results.append(meta)
                        except Exception as exc:
                            logger.error("EVE download failed %s: %s", file_url, exc)
                    break  # Got data from this URL pattern, move on

        logger.info("EVE product '%s': %d files", name, len(results))
        return results

    # ------------------------------------------------------------------ #
    #  Merged spectra (daily)
    # ------------------------------------------------------------------ #

    def _extract_merged_spectra(self) -> list[dict]:
        """Download merged daily spectra (Level 2 or Level 3)."""
        results = []

        for d in self.iter_target_dates():
            year = d.strftime("%Y")
            doy = d.timetuple().tm_yday
            doy_str = f"{doy:03d}"

            urls_to_try = [
                f"{self._DOWNLOAD_BASE}download/level2/merged/{year}/{doy_str}/",
                f"{self._DOWNLOAD_BASE}download/level3/merged/{year}/{doy_str}/",
                f"{self._DOWNLOAD_BASE}download/level2b/merged/{year}/{d.strftime('%m')}/",
            ]

            for dir_url in urls_to_try:
                files = self._list_data_files(dir_url, d)
                if files:
                    for fname in files:
                        file_url = urljoin(dir_url, fname)
                        sub = f"merged/{year}/{d.strftime('%m')}/{d.strftime('%d')}"
                        dest = self.dest_path(sub, fname)
                        try:
                            meta = self.client.download_file(file_url, dest)
                            meta.update({
                                "source": self.SOURCE_KEY,
                                "product": "merged_spectra",
                                "date": str(d),
                                "doy": doy,
                                "filename": fname,
                            })
                            self.record_metadata(meta)
                            results.append(meta)
                        except Exception as exc:
                            logger.error("EVE merged download failed %s: %s",
                                         file_url, exc)
                    break

        logger.info("EVE merged spectra: %d files", len(results))
        return results

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _list_data_files(self, directory_url: str, target_date: date) -> list[str]:
        """
        Parse a directory listing and return science data file names
        (.fits, .fit, .nc, .csv, .sav) that match the target date.
        """
        try:
            resp = self.client.get(directory_url)
        except Exception:
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        extensions = (".fits", ".fit", ".fits.gz", ".nc", ".csv", ".sav")
        date_str = target_date.strftime("%Y%j")  # YYYYDOY format
        date_str2 = target_date.strftime("%Y%m%d")  # YYYYMMDD

        files = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(href.endswith(ext) for ext in extensions):
                # Accept if date is in filename or if directory is date-specific
                if date_str in href or date_str2 in href or href.startswith("EVE"):
                    files.append(href)

        return files
