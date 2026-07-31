"""Search PubMed Central through the NCBI E-utilities API."""

from collections.abc import Callable, Mapping
import time

import httpx
from loguru import logger
import pandas as pd

from pmcortex.http_context_manager import HTTPContextManager, RequestRateLimiter


class PMCSearch(HTTPContextManager):
    """Search PMC with ESearch and retrieve matching titles with ESummary."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    MAX_RESULTS = 10_000
    SUMMARY_BATCH_SIZE = 200

    def __init__(
        self,
        *,
        user_agent: str = "PMCSearch/1.0 (contact: you@example.org)",
        timeout_s: float = 30.0,
        max_requests_per_second: float = 2.9,
        max_attempts: int = 3,
        retry_backoff_s: float = 1.0,
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        """
        Initialize the E-utilities HTTP client and request rate limit.

        The default stays slightly below NCBI's three-requests-per-second limit.
        Transient transport failures, HTTP 429 responses, and server errors are
        retried with exponential backoff. ``time_fn`` and ``sleep_fn`` allow
        deterministic tests.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if retry_backoff_s < 0:
            raise ValueError("retry_backoff_s must not be negative")

        self._max_attempts = max_attempts
        self._retry_backoff_s = retry_backoff_s
        self._sleep_fn = time.sleep if sleep_fn is None else sleep_fn
        self._rate_limiter = RequestRateLimiter(
            max_requests_per_second,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )
        super().__init__(
            user_agent=user_agent,
            timeout_s=timeout_s,
            accept="application/json,*/*;q=0.1",
        )

    def search(self, query: str, max_results: int = 10) -> pd.DataFrame:
        """
        Return up to ``max_results`` relevant PMC hits for a query.

        ESearch determines the ordered PMC identifiers. ESummary retrieves the
        corresponding titles in batches while preserving the ESearch order.

        Args:
            query: Non-empty PMC search expression.
            max_results: Requested number of hits, from 1 through 10,000.

        Returns:
            DataFrame with ``pmcid`` and ``title`` columns. It may contain fewer
            rows when PMC has fewer matching records.

        Raises:
            TypeError: If ``max_results`` is not an integer.
            ValueError: If the query is empty or the result limit is invalid.
            httpx.HTTPError: If an API request still fails after all attempts.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise TypeError("max_results must be an integer")
        if not 1 <= max_results <= self.MAX_RESULTS:
            raise ValueError(
                f"max_results must be between 1 and {self.MAX_RESULTS}"
            )

        pmc_uids = self._search_pmc_uids(query, max_results)
        titles = self._fetch_titles(pmc_uids)
        rows = [
            {
                "pmcid": f"PMC{uid}",
                "title": titles.get(uid, f"PMC{uid}"),
            }
            for uid in pmc_uids
        ]
        logger.info(
            "Retrieved {} of up to {} requested PMC hits for query: {}",
            len(rows),
            max_results,
            query,
        )
        return pd.DataFrame(rows, columns=["pmcid", "title"])

    def _search_pmc_uids(self, query: str, max_results: int) -> list[str]:
        """Retrieve relevance-ordered PMC UIDs from ESearch."""
        response = self._post_with_retries(
            self.ESEARCH_URL,
            data={
                "db": "pmc",
                "term": query,
                "retmax": str(max_results),
                "retmode": "json",
                "sort": "relevance",
            },
        )
        payload: object = response.json()
        if not isinstance(payload, dict):
            raise ValueError("ESearch returned an invalid JSON document")

        search_result = payload.get("esearchresult")
        if not isinstance(search_result, dict):
            raise ValueError("ESearch response is missing 'esearchresult'")

        id_list = search_result.get("idlist")
        ids_are_valid = isinstance(id_list, list) and all(
            isinstance(uid, str) and uid.isdigit()
            for uid in id_list
        )
        if not ids_are_valid:
            raise ValueError("ESearch response contains an invalid 'idlist'")

        return [uid for uid in id_list if isinstance(uid, str)]

    def _fetch_titles(self, pmc_uids: list[str]) -> dict[str, str]:
        """Retrieve ESummary titles for PMC UIDs in API-friendly batches."""
        titles: dict[str, str] = {}
        for start in range(0, len(pmc_uids), self.SUMMARY_BATCH_SIZE):
            batch = pmc_uids[start : start + self.SUMMARY_BATCH_SIZE]
            response = self._post_with_retries(
                self.ESUMMARY_URL,
                data={
                    "db": "pmc",
                    "id": ",".join(batch),
                    "retmode": "json",
                    "version": "2.0",
                },
            )
            self._extract_titles_inplace(response.json(), batch, titles)
        return titles

    @staticmethod
    def _extract_titles_inplace(
        payload: object,
        requested_uids: list[str],
        titles: dict[str, str],
    ) -> None:
        """Add valid ESummary titles from one batch to ``titles`` in place."""
        if not isinstance(payload, dict):
            raise ValueError("ESummary returned an invalid JSON document")

        summary_result = payload.get("result")
        if not isinstance(summary_result, dict):
            raise ValueError("ESummary response is missing 'result'")

        for uid in requested_uids:
            record = summary_result.get(uid)
            if not isinstance(record, dict):
                continue
            title = record.get("title")
            if isinstance(title, str) and title.strip():
                titles[uid] = title.strip()

    def _post_with_retries(
        self,
        url: str,
        *,
        data: Mapping[str, str],
    ) -> httpx.Response:
        """POST to E-utilities, retrying only failures that may be transient."""
        for attempt in range(1, self._max_attempts + 1):
            try:
                self._rate_limiter.wait()
                response = self.client.post(url, data=data)
                response.raise_for_status()
                return response
            except httpx.TransportError as error:
                if attempt == self._max_attempts:
                    raise
                self._wait_before_retry(attempt, error)
            except httpx.HTTPStatusError as error:
                status_code = error.response.status_code
                is_transient_status = status_code == 429 or 500 <= status_code < 600
                if not is_transient_status or attempt == self._max_attempts:
                    raise
                self._wait_before_retry(attempt, error)

        raise RuntimeError("Unreachable retry state.")

    def _wait_before_retry(
        self,
        failed_attempt: int,
        error: httpx.HTTPError,
    ) -> None:
        """Log a transient failure and apply exponential retry backoff."""
        delay_s = self._retry_backoff_s * (2 ** (failed_attempt - 1))
        logger.warning(
            "Transient PMC search failure on attempt {}/{}: {}. "
            "Retrying in {:.2f} seconds.",
            failed_attempt,
            self._max_attempts,
            repr(error),
            delay_s,
        )
        self._sleep_fn(delay_s)
