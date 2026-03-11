"""
Fenyi Solar Observatory (Hungary) SDO Data Extractor
=====================================================
Extracts solar data and flare catalogs from:
  http://fenyi.solarobs.epss.hun-ren.hu/en/databases/SDO/

The Fenyi observatory provides:
  - SDO/AIA image archives with flare-associated annotations
  - Active region catalogs
  - Solar flare event catalogs (with GOES class, location, timing)

These catalogs are especially valuable for labeling flare events across
the other imaging data sources.
"""

import csv
import io
import json
import logging
import re
from datetime import date
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from src.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class FenyiExtractor(BaseExtractor):
    SOURCE_KEY = "fenyi_observatory"

    def extract(self) -> list[dict]:
        logger.info("=== Fenyi Observatory extraction starting ===")
        results: list[dict] = []
        data_types = self.source_cfg.get("data_types", [])

        if "flare_catalog" in data_types:
            results.extend(self._extract_flare_catalog())

        if "active_region_catalog" in data_types:
            results.extend(self._extract_active_region_catalog())

        if "sdo_images" in data_types:
            results.extend(self._extract_sdo_images())

        self.save_metadata()
        logger.info("=== Fenyi extraction complete: %d items ===", len(results))
        return results

    # ------------------------------------------------------------------ #
    #  Flare Catalog
    # ------------------------------------------------------------------ #

    def _extract_flare_catalog(self) -> list[dict]:
        """
        Scrape the Fenyi flare catalog pages.
        The observatory often provides flare lists by month/year
        with GOES class, timing, heliographic coordinates, etc.
        """
        results = []
        catalog_url = urljoin(self.base_url, "flare_catalog/")

        # Attempt to fetch the main catalog index
        try:
            resp = self.client.get(self.base_url)
        except Exception as exc:
            logger.warning("Fenyi main page unreachable: %s", exc)
            return results

        soup = BeautifulSoup(resp.text, "lxml")

        # Look for links to flare databases, catalogs, download pages
        catalog_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True).lower()
            if any(kw in text for kw in ("flare", "catalog", "database", "list")):
                catalog_links.append(urljoin(self.base_url, href))
            elif any(kw in href.lower() for kw in ("flare", "catalog", "event")):
                catalog_links.append(urljoin(self.base_url, href))

        # Deduplicate
        catalog_links = list(dict.fromkeys(catalog_links))
        logger.info("Fenyi: found %d catalog-related links", len(catalog_links))

        for link in catalog_links:
            try:
                page_results = self._scrape_catalog_page(link)
                results.extend(page_results)
            except Exception as exc:
                logger.warning("Failed to scrape %s: %s", link, exc)

        # Also try direct date-based queries
        for d in self.iter_target_dates():
            results.extend(self._query_flares_for_date(d))

        logger.info("Fenyi flare catalog: %d records", len(results))
        return results

    def _scrape_catalog_page(self, url: str) -> list[dict]:
        """Try to extract tabular data from a catalog page."""
        resp = self.client.get(url)
        soup = BeautifulSoup(resp.text, "lxml")
        records = []

        # Try to find HTML tables with flare data
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            # Parse header
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            if not headers:
                continue

            for row in rows[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) == len(headers):
                    record = dict(zip(headers, cells))
                    record["source"] = self.SOURCE_KEY
                    record["catalog_url"] = url
                    # Check if this record falls within our date ranges
                    if self._record_in_date_range(record):
                        records.append(record)
                        self.record_metadata(record)

        # Also look for downloadable CSV/text files
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith((".csv", ".txt", ".dat", ".fits")):
                file_url = urljoin(url, href)
                fname = href.split("/")[-1]
                dest = self.dest_path("catalogs", fname)
                try:
                    meta = self.client.download_file(file_url, dest)
                    meta["source"] = self.SOURCE_KEY
                    meta["type"] = "catalog_file"
                    self.record_metadata(meta)
                    records.append(meta)
                except Exception as exc:
                    logger.debug("Fenyi file download failed %s: %s", file_url, exc)

        return records

    def _query_flares_for_date(self, d: date) -> list[dict]:
        """
        Try date-parameterized queries if the site supports them.
        Some observatory sites expose query parameters for date filtering.
        """
        results = []
        date_str = d.strftime("%Y-%m-%d")

        # Common query patterns for solar observatories
        query_urls = [
            f"{self.base_url}?date={date_str}",
            f"{self.base_url}?start={date_str}&end={date_str}",
        ]

        for qurl in query_urls:
            try:
                resp = self.client.get(qurl)
                if resp.status_code == 200 and len(resp.text) > 200:
                    page_results = self._parse_query_response(resp.text, d)
                    results.extend(page_results)
                    if page_results:
                        break
            except Exception:
                continue

        return results

    def _parse_query_response(self, html: str, d: date) -> list[dict]:
        """Parse a query response page for flare/event data."""
        soup = BeautifulSoup(html, "lxml")
        records = []

        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            for row in rows[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) == len(headers):
                    record = dict(zip(headers, cells))
                    record["source"] = self.SOURCE_KEY
                    record["query_date"] = str(d)
                    records.append(record)
                    self.record_metadata(record)

        return records

    # ------------------------------------------------------------------ #
    #  Active Region Catalog
    # ------------------------------------------------------------------ #

    def _extract_active_region_catalog(self) -> list[dict]:
        """Attempt to extract active region catalog data."""
        results = []
        try:
            resp = self.client.get(self.base_url)
        except Exception as exc:
            logger.warning("Fenyi AR catalog unreachable: %s", exc)
            return results

        soup = BeautifulSoup(resp.text, "lxml")
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True).lower()
            href = a["href"]
            if "active" in text or "region" in text or "ar" in href.lower():
                ar_url = urljoin(self.base_url, href)
                try:
                    page_results = self._scrape_catalog_page(ar_url)
                    results.extend(page_results)
                except Exception as exc:
                    logger.debug("AR catalog page failed %s: %s", ar_url, exc)

        logger.info("Fenyi active region catalog: %d records", len(results))
        return results

    # ------------------------------------------------------------------ #
    #  SDO Images from Fenyi mirror
    # ------------------------------------------------------------------ #

    def _extract_sdo_images(self) -> list[dict]:
        """Download SDO images hosted on the Fenyi mirror."""
        results = []

        try:
            resp = self.client.get(self.base_url)
        except Exception as exc:
            logger.warning("Fenyi SDO images unreachable: %s", exc)
            return results

        soup = BeautifulSoup(resp.text, "lxml")
        image_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith((".fits", ".fits.gz", ".jpg", ".png")):
                image_links.append(urljoin(self.base_url, href))

        for img_url in image_links:
            fname = img_url.split("/")[-1]
            dest = self.dest_path("sdo_images", fname)
            try:
                meta = self.client.download_file(img_url, dest)
                meta["source"] = self.SOURCE_KEY
                meta["type"] = "sdo_image"
                self.record_metadata(meta)
                results.append(meta)
            except Exception as exc:
                logger.debug("Fenyi image download failed %s: %s", img_url, exc)

        logger.info("Fenyi SDO images: %d files", len(results))
        return results

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _record_in_date_range(self, record: dict) -> bool:
        """Check if a catalog record falls within the configured date ranges."""
        # Try common date field names
        for key in ("Date", "date", "DATE", "Start", "start", "Event Date",
                     "DATE_OBS", "date_obs"):
            val = record.get(key, "")
            if val:
                try:
                    # Try parsing common date formats
                    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y",
                                "%Y-%m-%dT%H:%M:%S", "%Y%m%d"):
                        try:
                            parsed = __import__("datetime").datetime.strptime(
                                val[:10], fmt
                            ).date()
                            for dr in self.config.date_ranges:
                                if dr["start"] <= parsed <= dr["end"]:
                                    return True
                            return False
                        except ValueError:
                            continue
                except Exception:
                    pass
        # If no date field found, include it (we'll filter later)
        return True
