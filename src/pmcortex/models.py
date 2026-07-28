from dataclasses import dataclass
from pathlib import Path
from loguru import logger


@dataclass(frozen=True)
class DownloadResult:
    """
    Represents the result of a download operation.

    Attributes:
        pmcid (str): The PubMed Central ID of the article.
        path (Path | None): The file path where the article is saved, or None if not applicable.
        status (str): The status of the download (e.g., "ok", "not_found", "no_fulltext", "error").
        message (str | None): Additional information about the download status.
    """
    pmcid: str
    path: Path | None
    status: str  # "ok" | "not_found" | "no_fulltext" | "error"
    message: str | None = None

    def log(self):
        """Logs the details of the download result using loguru."""
        logger.info(f"DownloadResult - PMCID: {self.pmcid}, Status: {self.status}, Path: {self.path}, Message: {self.message}")


@dataclass(frozen=True)
class Reference:
    rid: str | None           # JATS ref id, e.g. "R12"
    label: str | None         # JATS label, e.g. "1" if the reference is labeled "[1]", otherwise None
    doi: str | None
    pmid: str | None
    pmcid: str | None
    year: str | None
    title: str | None
    journal: str | None        # journal / book / etc.
    authors: list[str]        # "Surname Given"


@dataclass(frozen=True)
class JATSArticle:
    pmcid: str | None
    pmid: str | None
    title: str | None
    abstract: str | None
    authors: list[str]  # "Surname Given"
    references: list[Reference]
    sections: dict[str, str]

@dataclass(frozen=True)
class Context:
    query: str
    hits: list[str]
    query_length: int
    n_hits: int
    # source_pmcid: int
    context: str
    position: tuple[int, int]
