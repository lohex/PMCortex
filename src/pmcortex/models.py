"""Data models shared by the PMC download and JATS parsing pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger


DownloadStatus = Literal["ok", "not_found", "no_fulltext", "error"]


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of downloading one PMC article."""

    pmcid: str
    path: Path | None
    status: DownloadStatus
    message: str | None = None

    def log(self) -> None:
        """Log the download outcome."""
        logger.info(
            "DownloadResult - PMCID: {}, Status: {}, Path: {}, Message: {}",
            self.pmcid,
            self.status,
            self.path,
            self.message,
        )


@dataclass(frozen=True, slots=True)
class Reference:
    """Bibliographic reference extracted from a JATS article."""

    rid: str | None
    label: str | None
    doi: str | None
    pmid: str | None
    pmcid: str | None
    year: int | None
    title: str | None
    journal: str | None
    authors: list[str]


@dataclass(frozen=True, slots=True)
class JATSArticle:
    """Structured content extracted from one JATS article."""

    pmcid: str | None
    pmid: str | None
    title: str | None
    abstract: str | None
    authors: list[str]
    references: list[Reference]
    sections: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class Context:
    """Retrieval query and cited positive documents extracted from a sentence."""

    query: str
    hits: list[str]
    query_length: int
    n_hits: int
    context: str | None
    position: tuple[int, int]
