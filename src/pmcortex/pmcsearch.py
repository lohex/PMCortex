"""Search the PMC website for article identifiers and titles."""

import re
from urllib.parse import quote_plus

from loguru import logger
from lxml import etree
import pandas as pd

from pmcortex.http_context_manager import HTTPContextManager


class PMCSearch(HTTPContextManager):
    """Search PMC and collect matching PMCID/title pairs."""

    SEARCH_BASE = "https://www.ncbi.nlm.nih.gov/pmc/"

    def __init__(
        self,
        *,
        user_agent: str = "PMCSearch/1.0 (contact: you@example.org)",
        timeout_s: float = 30.0,
    ) -> None:
        """Initialize search state and HTTP client."""
        self.hit_list: list[dict[str, str]] = []
        self.query: str | None = None
        self.doc: etree._Element | None = None
        super().__init__(
            user_agent=user_agent,
            timeout_s=timeout_s,
            accept="text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
        )

    def search(self, search: str) -> None:
        """Populate this instance with hits for a non-empty search query."""
        query = search.strip()
        if not query:
            raise ValueError("search must not be empty")
        self.query = query
        self.get_results()
        self.extract_pmcids()

    def get_results(self) -> etree._Element:
        """Fetch and parse the PMC search result page."""
        if not self.query:
            raise ValueError("Search query is not set.")

        query = quote_plus(self.query)
        url = f"{self.SEARCH_BASE}?term={query}"
        response = self.client.get(url)
        response.raise_for_status()

        parser = etree.HTMLParser(recover=True)
        self.doc = etree.fromstring(response.content, parser=parser)
        logger.info(f"Fetched search results for query: {self.query}")
        return self.doc

    def extract_pmcids(self) -> list[dict[str, str]]:
        """Extract PMCID/title pairs from the current search document."""
        if self.doc is None:
            raise ValueError("No parsed document available. Call get_results() first.")

        hits: list[dict[str, str]] = []
        for node in self.doc.xpath("//a[contains(@href, '/pmc.ncbi.nlm.nih.gov/articles/PMC')]"):
            pmcid = node.get('data-ga-label')

            title = " ".join(node.itertext()).strip()
            title = re.sub(' +', ' ', title)
            if not title:
                title = pmcid

            hits.append({"pmcid": pmcid, "title": title})

        self.hit_list = hits
        logger.info(f"Extracted {len(self.hit_list)} PMCID hits")
        return self.hit_list

    def get_all(self) -> pd.DataFrame:
        """Return extracted hits as DataFrame."""
        return pd.DataFrame(self.hit_list)
