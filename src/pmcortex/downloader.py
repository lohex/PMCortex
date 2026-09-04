"""Download PMC articles as local JATS XML files."""

from collections.abc import Iterable, Iterator
import os
from pathlib import Path
import random
import re
import tempfile
import time
from types import TracebackType

import httpx
from lxml import etree
from loguru import logger
from pmcortex.models import DownloadResult



class PMCJATSDownloader:
    """
    A synchronous downloader for PMC JATS XML files using the OAI-PMH API.

    This class provides methods to fetch and extract JATS XML files for articles
    from PubMed Central (PMC) using the OAI-PMH API.

    Attributes:
        OAI_BASE (str): The base URL for the OAI-PMH API.
    """

    OAI_BASE = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"

    def __init__(self,
        out_dir: str | Path,
        *,
        user_agent: str = "PMCJATSDownloader/1.0 (contact: you@example.org)",
        timeout_s: float = 30.0,
        min_interval_s: float = 0.6,
        max_retries: int = 5,
        backoff_base_s: float = 0.8,
        jitter_s: float = 0.25,
        overwrite: bool = False,
    ) -> None:
        """
        Initializes the PMCJATSDownloader.

        Args:
            out_dir (str | Path): The directory where downloaded files will be saved.
            user_agent (str): The User-Agent header for HTTP requests.
            timeout_s (float): The timeout for HTTP requests in seconds.
            min_interval_s (float): The minimum interval between requests to avoid rate-limiting.
            max_retries (int): The maximum number of retries for failed requests.
            backoff_base_s (float): The base duration for exponential backoff.
            jitter_s (float): The random jitter added to backoff duration.
            overwrite (bool): Whether to overwrite existing files.
        """
        if not user_agent.strip():
            raise ValueError("user_agent must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if min_interval_s < 0:
            raise ValueError("min_interval_s must not be negative")
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if backoff_base_s < 0:
            raise ValueError("backoff_base_s must not be negative")
        if jitter_s < 0:
            raise ValueError("jitter_s must not be negative")

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.jitter_s = jitter_s
        self.overwrite = overwrite

        self._next_allowed = 0.0

        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            follow_redirects=True,
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.1",
            },
        )
        logger.info("Initialized PMCJATSDownloader")

    def close(self) -> None:
        """Closes the HTTP client."""
        self.client.close()
        logger.info("Closed HTTP client")

    def __enter__(self) -> "PMCJATSDownloader":
        """Enters the context manager."""
        logger.info("Entering context manager")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exits the context manager and closes the HTTP client."""
        self.close()
        logger.info("Exiting context manager")

    # ---------- Helpers ----------

    @staticmethod
    def normalize_pmcid(pmcid: str) -> str:
        """
        Normalizes a PMC ID to ensure it is in the correct format.

        Args:
            pmcid (str): The PMC ID to normalize.

        Returns:
            str: The normalized PMC ID.

        Raises:
            ValueError: If the PMC ID is invalid.
        """
        pmcid = pmcid.strip().upper()
        if not pmcid.startswith("PMC"):
            pmcid = "PMC" + pmcid
        if not re.fullmatch(r"PMC\d+", pmcid):
            raise ValueError(f"Invalid PMCID: {pmcid}")
        logger.debug(f"Normalized PMCID: {pmcid}")
        return pmcid

    @staticmethod
    def oai_identifier(pmcid: str) -> str:
        """
        Generates an OAI identifier for a given PMC ID.

        Args:
            pmcid (str): The PMC ID.

        Returns:
            str: The OAI identifier.
        """
        pmcid = PMCJATSDownloader.normalize_pmcid(pmcid)
        num = pmcid.removeprefix("PMC")
        identifier = f"oai:pubmedcentral.nih.gov:{num}"
        logger.debug(f"Generated OAI identifier: {identifier}")
        return identifier

    def out_path(self, pmcid: str) -> Path:
        """
        Constructs the output file path for a given PMC ID.

        Args:
            pmcid (str): The PMC ID.

        Returns:
            Path: The output file path.
        """
        pmcid = self.normalize_pmcid(pmcid)
        path = self.out_dir / f"{pmcid}.nxml"
        logger.debug(f"Output path for {pmcid}: {path}")
        return path

    def _rate_limit(self) -> None:
        """
        Enforces rate limiting between requests.

        If the current time is before the next allowed time, the method sleeps
        until the next allowed time.
        """
        now = time.monotonic()
        if now < self._next_allowed:
            sleep_time = self._next_allowed - now
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self._next_allowed = time.monotonic() + self.min_interval_s

    def _backoff_sleep(self, attempt: int) -> float:
        """
        Calculates the backoff sleep duration for a given attempt.

        Args:
            attempt (int): The attempt number.

        Returns:
            float: The backoff sleep duration in seconds.
        """
        backoff_time = self.backoff_base_s * (2 ** (attempt - 1)) + random.random() * self.jitter_s
        logger.debug(f"Backoff sleep for attempt {attempt}: {backoff_time:.2f} seconds")
        return backoff_time

    def _build_url(self, identifier: str) -> str:
        """
        Constructs the URL for fetching an OAI record.

        Args:
            identifier (str): The OAI identifier.

        Returns:
            str: The constructed URL.
        """
        url = (
            f"{self.OAI_BASE}"
            f"?verb=GetRecord"
            f"&identifier={identifier}"
            f"&metadataPrefix=pmc"
        )
        logger.debug(f"Built URL: {url}")
        return url

    @staticmethod
    def _write_atomic(path: Path, content: bytes) -> None:
        """Publish bytes atomically through a process-unique neighboring file."""
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    # ---------- Core ----------

    def fetch_oai_xml(self, pmcid: str) -> bytes:
        """
        Fetches the OAI XML for a given PMC ID.

        Args:
            pmcid (str): The PMC ID.

        Returns:
            bytes: The fetched OAI XML.

        Raises:
            RuntimeError: If the OAI record cannot be fetched after retries.
        """
        identifier = self.oai_identifier(pmcid)
        url = self._build_url(identifier)

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit()
                resp = self.client.get(url)
                resp.raise_for_status()
                logger.info(f"Successfully fetched OAI XML for {pmcid}")
                return resp.content

            except Exception as e:
                last_err = e
                logger.error(f"Error fetching {pmcid} (attempt {attempt}): {repr(e)}")
                if attempt == self.max_retries:
                    break
                time.sleep(self._backoff_sleep(attempt))

        raise RuntimeError(f"Failed to fetch OAI record for {pmcid}") from last_err

    @staticmethod
    def extract_jats(oai_xml: bytes) -> bytes:
        """
        Extracts the JATS XML from the OAI XML.

        Args:
            oai_xml (bytes): The OAI XML.

        Returns:
            bytes: The extracted JATS XML.

        Raises:
            ValueError: If the OAI XML contains errors or no <article> element.
        """
        parser = etree.XMLParser(recover=True, resolve_entities=False, huge_tree=True)
        root = etree.fromstring(oai_xml, parser=parser)

        err = root.find(".//{*}error")
        if err is not None:
            code = err.get("code")
            msg = (err.text or "").strip()
            logger.error(f"OAI error code={code} msg={msg}")
            raise ValueError(f"OAI error code={code} msg={msg}")

        article = root.find(".//{*}metadata//{*}article")
        if article is None:
            logger.error("No <article> found in OAI metadata")
            raise ValueError("No <article> found in OAI metadata")

        logger.info("Successfully extracted JATS XML")
        return etree.tostring(
            article,
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=True,
        )

    def download_one(self, pmcid: str) -> DownloadResult:
        """
        Downloads and extracts the JATS XML for a single PMC ID.

        Args:
            pmcid (str): The PMC ID.

        Returns:
            DownloadResult: The result of the download operation.
        """
        pmcid = self.normalize_pmcid(pmcid)
        path = self.out_path(pmcid)

        if path.exists() and not self.overwrite:
            logger.info(f"File already exists for {pmcid}, skipping download")
            return DownloadResult(pmcid, path, "ok", "cached")

        try:
            logger.info(f"Downloading {pmcid}...")
            oai_xml = self.fetch_oai_xml(pmcid)
            logger.info(f"Extracting JATS XML for {pmcid}...")
            jats_xml = self.extract_jats(oai_xml)

            self._write_atomic(path, jats_xml)

            logger.info(f"Successfully downloaded and saved {pmcid}")
            return DownloadResult(pmcid, path, "ok")

        except ValueError as e:
            msg = str(e)
            logger.warning(f"ValueError for {pmcid}: {msg}")
            if "idDoesNotExist" in msg:
                return DownloadResult(pmcid, None, "not_found", msg)
            if "cannotDisseminateFormat" in msg or "No <article>" in msg:
                return DownloadResult(pmcid, None, "no_fulltext", msg)
            return DownloadResult(pmcid, None, "error", msg)

        except Exception as e:
            logger.error(f"Unexpected error for {pmcid}: {repr(e)}")
            return DownloadResult(pmcid, None, "error", repr(e))

    def iter_downloads(
        self,
        pmcids: Iterable[str],
        limit: int | None = None,
    ) -> Iterator[DownloadResult]:
        """
        Yield download results as soon as each serial request finishes.

        Args:
            pmcids: PMC IDs to download in iteration order.
            limit (int, optional): The maximum number of PMC IDs to download.

        Yields:
            One download result per selected PMC ID.

        Raises:
            TypeError: If ``pmcids`` is a string instead of an ID iterable.
            ValueError: If ``limit`` is not positive.
        """
        if isinstance(pmcids, (str, bytes)):
            raise TypeError("pmcids must be an iterable of PMC ID strings")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")

        for index, pmcid in enumerate(pmcids):
            if limit is not None and index >= limit:
                break
            result = self.download_one(pmcid)
            result.log()
            yield result

    def download_many(
        self,
        pmcids: list[str],
        limit: int | None = None,
    ) -> list[DownloadResult]:
        """Download multiple PMC IDs and return their completed results."""
        return list(self.iter_downloads(pmcids, limit=limit))
