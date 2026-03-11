"""
Rate-limited HTTP client with retry logic for solar data downloads.

Implements a token-bucket rate limiter and exponential backoff retries,
respecting per-source rate limits defined in the pipeline config.
"""

import hashlib
import logging
import os
import threading
import time
from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ========================================================================= #
#  Token-bucket rate limiter (thread-safe)
# ========================================================================= #

class TokenBucket:
    """Simple token-bucket rate limiter."""

    def __init__(self, rate: float, burst: int):
        """
        Parameters
        ----------
        rate  : tokens added per second (= max sustained requests/s)
        burst : maximum tokens in the bucket
        """
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 120.0) -> bool:
        """Block until a token is available (up to *timeout* seconds)."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self._burst,
                                   self._tokens + elapsed * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            # Back off a bit before checking again
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)


# ========================================================================= #
#  Rate-limited requests session
# ========================================================================= #

class RateLimitedClient:
    """HTTP client with built-in rate limiting and retries."""

    def __init__(
        self,
        rate: float = 1.0,
        burst: int = 3,
        retry_max: int = 3,
        retry_backoff: float = 2.0,
        timeout: int = 60,
        user_agent: str = "OMNISolarPipeline/1.0 (research; space-weather)",
    ):
        self._bucket = TokenBucket(rate, burst)
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": user_agent,
            "Accept": "*/*",
        })

        # urllib3 automatic retries for transient errors
        retries = Retry(
            total=retry_max,
            backoff_factor=retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    # ------------------------------------------------------------------ #
    #  Core methods
    # ------------------------------------------------------------------ #

    def get(self, url: str, **kwargs) -> requests.Response:
        """Perform a rate-limited GET request."""
        self._bucket.acquire()
        kwargs.setdefault("timeout", self._timeout)
        logger.debug("GET %s", url)
        resp = self._session.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    def head(self, url: str, **kwargs) -> requests.Response:
        self._bucket.acquire()
        kwargs.setdefault("timeout", self._timeout)
        return self._session.head(url, **kwargs)

    def download_file(
        self,
        url: str,
        dest_path: str,
        *,
        chunk_size: int = 1024 * 256,
        skip_existing: bool = True,
    ) -> dict[str, Any]:
        """
        Download a file respecting rate limits.

        Returns metadata dict: {url, path, size_bytes, sha256, elapsed_sec}.
        """
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        if skip_existing and os.path.isfile(dest_path):
            size = os.path.getsize(dest_path)
            logger.info("Skipping (exists): %s  (%d bytes)", dest_path, size)
            return {
                "url": url,
                "path": dest_path,
                "size_bytes": size,
                "sha256": _file_sha256(dest_path),
                "elapsed_sec": 0.0,
                "skipped": True,
            }

        self._bucket.acquire()
        logger.info("Downloading: %s -> %s", url, dest_path)
        t0 = time.monotonic()

        resp = self._session.get(url, stream=True, timeout=self._timeout)
        resp.raise_for_status()

        sha = hashlib.sha256()
        total = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                sha.update(chunk)
                total += len(chunk)

        elapsed = time.monotonic() - t0
        logger.info("Downloaded %d bytes in %.1fs", total, elapsed)

        return {
            "url": url,
            "path": dest_path,
            "size_bytes": total,
            "sha256": sha.hexdigest(),
            "elapsed_sec": round(elapsed, 2),
            "skipped": False,
        }

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ========================================================================= #
#  Helpers
# ========================================================================= #

def _file_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            sha.update(block)
    return sha.hexdigest()


def build_client(source_cfg: dict) -> RateLimitedClient:
    """Factory: create a RateLimitedClient from a source config block."""
    rl = source_cfg.get("rate_limit", {})
    return RateLimitedClient(
        rate=rl.get("requests_per_second", 1.0),
        burst=rl.get("burst_size", 3),
        retry_max=rl.get("retry_max", 3),
        retry_backoff=rl.get("retry_backoff_factor", 2.0),
        timeout=source_cfg.get("timeout_seconds", 60),
    )
