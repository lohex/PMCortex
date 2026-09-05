"""Data models shared by the PMC download and JATS parsing pipeline."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from loguru import logger

from pmcortex.citation_normalizer import CitationDiagnostic


DownloadStatus = Literal["ok", "not_found", "no_fulltext", "error"]


def _validate_optional_string(name: str, value: str | None) -> None:
    """Validate one optional string field without changing its value."""
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")


def _validate_string_tuple(name: str, values: tuple[str, ...]) -> None:
    """Validate an immutable sequence whose members must all be strings."""
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not all(isinstance(value, str) for value in values):
        raise TypeError(f"{name} must contain only strings")


def _validate_nonnegative_integer(name: str, value: int) -> None:
    """Validate an integer index or count that cannot be negative."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")


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
    authors: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate immutable bibliographic metadata."""
        for name, value in (
            ("rid", self.rid),
            ("label", self.label),
            ("doi", self.doi),
            ("pmid", self.pmid),
            ("pmcid", self.pmcid),
            ("title", self.title),
            ("journal", self.journal),
        ):
            _validate_optional_string(name, value)
        if self.year is not None:
            _validate_nonnegative_integer("year", self.year)
        _validate_string_tuple("authors", self.authors)


@dataclass(frozen=True, slots=True)
class JATSSection:
    """Immutable public representation of one extracted article section."""

    title: str
    full_title: str
    source_blocks: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate section titles and immutable source blocks."""
        if not isinstance(self.title, str):
            raise TypeError("title must be a string")
        if not isinstance(self.full_title, str):
            raise TypeError("full_title must be a string")
        _validate_string_tuple("source_blocks", self.source_blocks)


@dataclass(frozen=True, slots=True)
class JATSArticle:
    """Structured content extracted from one JATS article."""

    pmcid: str | None
    pmid: str | None
    title: str | None
    abstract: str | None
    authors: tuple[str, ...]
    references: tuple[Reference, ...]
    sections: tuple[JATSSection, ...]

    def __post_init__(self) -> None:
        """Validate immutable article metadata and component collections."""
        for name, value in (
            ("pmcid", self.pmcid),
            ("pmid", self.pmid),
            ("title", self.title),
            ("abstract", self.abstract),
        ):
            _validate_optional_string(name, value)
        _validate_string_tuple("authors", self.authors)
        if not isinstance(self.references, tuple):
            raise TypeError("references must be a tuple")
        if not all(isinstance(reference, Reference) for reference in self.references):
            raise TypeError("references must contain only Reference values")
        if not isinstance(self.sections, tuple):
            raise TypeError("sections must be a tuple")
        if not all(isinstance(section, JATSSection) for section in self.sections):
            raise TypeError("sections must contain only JATSSection values")


@dataclass(frozen=True, slots=True)
class SentencePosition:
    """Globally identifiable structural position of one article sentence."""

    source_pmcid: str
    section_index: int
    paragraph_index: int
    sentence_index: int

    def __post_init__(self) -> None:
        """Validate the source identifier and nonnegative structural indices."""
        if not isinstance(self.source_pmcid, str):
            raise TypeError("source_pmcid must be a string")
        if not self.source_pmcid.strip():
            raise ValueError("source_pmcid must not be empty")
        _validate_nonnegative_integer("section_index", self.section_index)
        _validate_nonnegative_integer("paragraph_index", self.paragraph_index)
        _validate_nonnegative_integer("sentence_index", self.sentence_index)

    @property
    def sentence_id(self) -> str:
        """Return the canonical PMCID and structural-index identifier."""
        return (
            f"{self.source_pmcid}/{self.section_index}/"
            f"{self.paragraph_index}/{self.sentence_index}"
        )


@dataclass(frozen=True, slots=True)
class PositionedSentence:
    """One normalized full-text sentence with its structural article position."""

    position: SentencePosition
    text: str

    def __post_init__(self) -> None:
        """Validate one immutable positioned sentence."""
        if not isinstance(self.position, SentencePosition):
            raise TypeError("position must be a SentencePosition")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


class CitationCleanupAction(StrEnum):
    """Closed set of citation-removal actions applied to a retrieval query."""

    REMOVED_LABEL_CITATION = "removed_label_citation"
    REMOVED_PARENTHETICAL_CITATION = "removed_parenthetical_citation"


@dataclass(frozen=True, slots=True)
class Context:
    """Retrieval query and cited positive documents extracted from a sentence."""

    position: SentencePosition
    query_raw: str | None
    query: str
    citation_forms: tuple[str, ...]
    citation_cleanup_action: CitationCleanupAction
    hits: tuple[str, ...]
    query_length: int
    n_hits: int
    context: str | None

    def __post_init__(self) -> None:
        """Validate retrieval text, immutable hits, counts, and cleanup action."""
        if not isinstance(self.position, SentencePosition):
            raise TypeError("position must be a SentencePosition")
        _validate_optional_string("query_raw", self.query_raw)
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        _validate_string_tuple("citation_forms", self.citation_forms)
        if not isinstance(self.citation_cleanup_action, CitationCleanupAction):
            raise TypeError(
                "citation_cleanup_action must be a CitationCleanupAction"
            )
        _validate_string_tuple("hits", self.hits)
        _validate_nonnegative_integer("query_length", self.query_length)
        _validate_nonnegative_integer("n_hits", self.n_hits)
        if self.n_hits != len(self.hits):
            raise ValueError("n_hits must equal the number of hits")
        _validate_optional_string("context", self.context)


@dataclass(frozen=True, slots=True)
class NormalizedAuthorLists:
    """Immutable normalized article and cited-author names for YAML export."""

    authors: tuple[str, ...]
    cited_authors: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        """Validate both immutable levels of the normalized author lists."""
        _validate_string_tuple("authors", self.authors)
        if not isinstance(self.cited_authors, tuple):
            raise TypeError("cited_authors must be a tuple")
        for authors in self.cited_authors:
            _validate_string_tuple("cited_authors entry", authors)


@dataclass(frozen=True, slots=True)
class ContextExtractionResult:
    """Contexts, optional sentences, and diagnostics extracted from one article."""

    contexts: tuple[Context, ...]
    sentences: tuple[PositionedSentence, ...]
    diagnostics: tuple[CitationDiagnostic, ...]
    unsafe_citation_rejection_count: int
    alignment_rejection_count: int = 0

    def __post_init__(self) -> None:
        """Validate immutable extraction collections and rejection counters."""
        if not isinstance(self.contexts, tuple):
            raise TypeError("contexts must be a tuple")
        if not all(isinstance(context, Context) for context in self.contexts):
            raise TypeError("contexts must contain only Context values")
        if not isinstance(self.sentences, tuple):
            raise TypeError("sentences must be a tuple")
        if not all(
            isinstance(sentence, PositionedSentence) for sentence in self.sentences
        ):
            raise TypeError("sentences must contain only PositionedSentence values")
        if not isinstance(self.diagnostics, tuple):
            raise TypeError("diagnostics must be a tuple")
        if not all(
            isinstance(diagnostic, CitationDiagnostic)
            for diagnostic in self.diagnostics
        ):
            raise TypeError("diagnostics must contain only CitationDiagnostic values")
        _validate_nonnegative_integer(
            "unsafe_citation_rejection_count",
            self.unsafe_citation_rejection_count,
        )
        _validate_nonnegative_integer(
            "alignment_rejection_count",
            self.alignment_rejection_count,
        )


@dataclass(frozen=True, slots=True)
class ParsedJATSResult:
    """Complete immutable result of parsing one JATS article."""

    article: JATSArticle
    extraction: ContextExtractionResult

    def __post_init__(self) -> None:
        """Validate the article and context-extraction result."""
        if not isinstance(self.article, JATSArticle):
            raise TypeError("article must be a JATSArticle")
        if not isinstance(self.extraction, ContextExtractionResult):
            raise TypeError("extraction must be a ContextExtractionResult")

    @property
    def contexts(self) -> tuple[Context, ...]:
        """Return extracted retrieval contexts."""
        return self.extraction.contexts

    @property
    def sentences(self) -> tuple[PositionedSentence, ...]:
        """Return optionally retained positioned sentences."""
        return self.extraction.sentences

    @property
    def diagnostics(self) -> tuple[CitationDiagnostic, ...]:
        """Return citation diagnostics accumulated during parsing."""
        return self.extraction.diagnostics

    @property
    def unsafe_citation_rejection_count(self) -> int:
        """Return the number of rejected semantically unsafe citations."""
        return self.extraction.unsafe_citation_rejection_count

    @property
    def alignment_rejection_count(self) -> int:
        """Return the number of rejected sentence-alignment failures."""
        return self.extraction.alignment_rejection_count
