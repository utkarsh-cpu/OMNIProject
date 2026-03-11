"""
SDO / NASA Data Extractor
=========================
Downloads AIA and HMI browse-quality FITS/JPEG data from the NASA SDO portal.
URL: https://sdo.gsfc.nasa.gov/data/

The SDO data site exposes AIA synoptic imagery under:
  https://sdo.gsfc.nasa.gov/assets/img/browse/{YYYY}/{MM}/{DD}/
with filenames like:
  {YYYY}{MM}{DD}_{HHMMSS}_{AIA_WAVELENGTH|HMI_PRODUCT}.jpg

For higher-quality FITS data the pipeline delegates to the JSOC extractor.
This extractor focuses on the browse-quality images that are lightweight
enough for rapid scanning and flare event identification, plus any FITS
products directly hosted on the SDO portal.
"""

import logging
import re
from datetime import date
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class SDONasaExtractor(BaseExtractor):
    SOURCE_KEY = "sdo_nasa"

    # SDO browse image base (lightweight JPEG synoptic imagery)
    _BROWSE_BASE = "https://sdo.gsfc.nasa.gov/assets/img/browse/"

    def extract(self) -> list[dict]:
        logger.info("=== SDO/NASA extraction starting ===")
        results: list[dict] = []

        aia_channels = self.source_cfg.get("aia_channels", [])
        hmi_products = self.source_cfg.get("hmi_products", [])
        cadence = self.source_cfg.get("cadence_minutes", 720)

        for d in self.iter_target_dates():
            day_results = self._extract_day(d, aia_channels, hmi_products, cadence)
            results.extend(day_results)

        self.save_metadata()
        logger.info("=== SDO/NASA extraction complete: %d files ===", len(results))
        return results

    # ------------------------------------------------------------------ #

    def _extract_day(
        self, d: date, channels: list, hmi_products: list, cadence: int
    ) -> list[dict]:
        """Scrape the browse-image directory for a given day, download matching files."""
        day_str = d.strftime("%Y/%m/%d")
        url = f"{self._BROWSE_BASE}{day_str}/"

        try:
            resp = self.client.get(url)
        except Exception as exc:
            logger.warning("SDO browse listing failed for %s: %s", day_str, exc)
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        links = [
            a["href"] for a in soup.find_all("a", href=True)
            if a["href"].endswith((".jpg", ".fits", ".fits.gz"))
        ]

        if not links:
            logger.debug("No browse images found for %s", day_str)
            return []

        # Filter links to only requested channels/products
        wanted = set(str(c) for c in channels) | set(hmi_products)
        selected = self._filter_by_cadence_and_channel(links, wanted, cadence)

        results = []
        for fname in selected:
            file_url = urljoin(url, fname)
            sub_dir = d.strftime("%Y/%m/%d")
            dest = self.dest_path(sub_dir, fname)
            try:
                meta = self.client.download_file(file_url, dest)
                meta.update({
                    "source": self.SOURCE_KEY,
                    "date": str(d),
                    "filename": fname,
                    "channel": self._parse_channel(fname),
                })
                self.record_metadata(meta)
                results.append(meta)
            except Exception as exc:
                logger.error("Download failed %s: %s", file_url, exc)

        logger.info("SDO browse %s: downloaded %d / %d files",
                     day_str, len(results), len(selected))
        return results

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_channel(fname: str) -> str:
        """Extract wavelength/product from a filename like 20251201_000000_0171.jpg."""
        m = re.search(r"_(\d{3,4}|[a-zA-Z]+gram)\.", fname)
        return m.group(1) if m else "unknown"

    @staticmethod
    def _filter_by_cadence_and_channel(
        links: list[str], wanted: set[str], cadence_minutes: int
    ) -> list[str]:
        """
        From a day's listing, pick files matching wanted channels/products
        and thin them to approximately *cadence_minutes* spacing.
        """
        # Group by channel, then pick by time spacing
        by_channel: dict[str, list[tuple[int, str]]] = {}
        time_re = re.compile(r"(\d{8})_(\d{6})_")

        for link in links:
            m = time_re.search(link)
            if not m:
                continue
            hhmm = int(m.group(2)[:4])  # HHMM as integer for sorting
            ch = SDONasaExtractor._parse_channel(link)
            if ch in wanted or ch == "unknown":
                by_channel.setdefault(ch, []).append((hhmm, link))

        selected = []
        for ch, items in by_channel.items():
            items.sort()
            last_time = -cadence_minutes
            for hhmm, link in items:
                # convert HHMM to minutes-since-midnight
                minutes = (hhmm // 100) * 60 + (hhmm % 100)
                if minutes - last_time >= cadence_minutes:
                    selected.append(link)
                    last_time = minutes

        return selected
