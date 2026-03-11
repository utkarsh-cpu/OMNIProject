"""
JSOC / Stanford AIA Synoptic Data Extractor
============================================
Downloads AIA synoptic FITS images from:
  http://jsoc2.stanford.edu/data/aia/synoptic/

Directory layout on the server:
  {base_url}/{YYYY}/{MM}/{DD}/H{HH}00/
  Files:  AIA{YYYYMMDD}_{HHMMSS}_{WAVELENGTH}.fits

This is the primary source of science-grade (synoptic) FITS data
used for flare detection and space-weather analysis.
"""

import logging
import re
from datetime import date
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class JSOCExtractor(BaseExtractor):
    SOURCE_KEY = "jsoc_stanford"

    def extract(self) -> list[dict]:
        logger.info("=== JSOC/Stanford AIA synoptic extraction starting ===")
        results: list[dict] = []
        wavelengths = self.source_cfg.get("wavelengths", [])
        preferred_hours = self.source_cfg.get("preferred_hours", ["00", "12"])

        for d in self.iter_target_dates():
            day_results = self._extract_day(d, wavelengths, preferred_hours)
            results.extend(day_results)

        self.save_metadata()
        logger.info("=== JSOC extraction complete: %d FITS files ===", len(results))
        return results

    # ------------------------------------------------------------------ #

    def _extract_day(
        self, d: date, wavelengths: list[str], preferred_hours: list[str]
    ) -> list[dict]:
        """Download FITS files for a single day at preferred hour slots."""
        results = []

        for hour in preferred_hours:
            hour_dir = f"H{hour}00"
            url = (
                f"{self.base_url}"
                f"{d.strftime('%Y')}/{d.strftime('%m')}/{d.strftime('%d')}/"
                f"{hour_dir}/"
            )

            fits_links = self._list_fits(url)
            if not fits_links:
                logger.debug("No FITS at %s", url)
                continue

            # Filter to requested wavelengths
            for link in fits_links:
                wl = self._parse_wavelength(link)
                if wl and wl in wavelengths:
                    file_url = urljoin(url, link)
                    sub = f"{d.strftime('%Y/%m/%d')}/{hour_dir}"
                    dest = self.dest_path(sub, link)
                    try:
                        meta = self.client.download_file(file_url, dest)
                        meta.update({
                            "source": self.SOURCE_KEY,
                            "date": str(d),
                            "hour": hour,
                            "wavelength": wl,
                            "filename": link,
                            "format": "FITS",
                        })
                        self.record_metadata(meta)
                        results.append(meta)
                    except Exception as exc:
                        logger.error("JSOC download failed %s: %s", file_url, exc)

        logger.info("JSOC %s: %d FITS files", d.isoformat(), len(results))
        return results

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _list_fits(self, directory_url: str) -> list[str]:
        """Parse an Apache-style directory listing for .fits links."""
        try:
            resp = self.client.get(directory_url)
        except Exception as exc:
            logger.debug("JSOC listing failed %s: %s", directory_url, exc)
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        return [
            a["href"]
            for a in soup.find_all("a", href=True)
            if a["href"].endswith(".fits")
        ]

    @staticmethod
    def _parse_wavelength(filename: str) -> str | None:
        """
        Extract the 4-digit zero-padded wavelength from a filename like:
        AIA20251201_000008_0171.fits  ->  '0171'
        """
        m = re.search(r"_(\d{4})\.fits", filename)
        return m.group(1) if m else None
