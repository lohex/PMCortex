"""Public API for PMCortex retrieval-dataset construction."""

import sys

from loguru import logger


logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    filter={"pmcortex.pipeline": "INFO", "": "WARNING"},
)
logger.add("logs/pmcortex.log", level="INFO")

from .citation_normalizer import (
    CitationDiagnostic,
    CitationDiagnosticCode,
    CitationForm,
    CitationNormalizer,
    CitationOccurrence,
    NormalizedCitationText,
)
from .jatsparser import JATSParseError, JATSParser
from .models import (
    CitationCleanupAction,
    Context,
    ContextExtractionResult,
    JATSArticle,
    JATSDiagnostic,
    JATSSection,
    NormalizedAuthorLists,
    ParserDiagnostic,
    ParserDiagnosticCode,
    ParsedJATSResult,
    PositionedSentence,
    Reference,
    SentencePosition,
)
from .mesh import (
    PubMedMeSHClient,
    extract_abstracts_from_pubmed_xml,
    extract_mesh_terms_from_pubmed_xml,
)
from .pmcsearch import PMCSearch, PMCSearchResponseError
from .pipeline import (
    DatasetLayout,
    MaterializedDataset,
    PMCIngestionPipeline,
    ProcessingRecord,
)
from .query_filters import (
    CitationSpoilerAssessment,
    CitationSpoilerError,
    CitationSpoilerFilterResult,
    CitationSpoilerReason,
    CrossContextCoreferenceFilter,
    QueryFilterPipeline,
    assess_residual_citation_spoiler,
    assert_no_residual_citation_spoilers,
    filter_citation_spoilers,
)
from .serialization import (
    author_lists_to_payload,
    contexts_to_dataframe,
    diagnostics_to_payload,
    metadata_to_dataframe,
    normalize_author_lists,
    references_to_dataframe,
    render_author_yaml,
    render_diagnostics_json,
    render_positioned_sentences,
)
from .sentence_segmenter import (
    DEFAULT_SENTENCE_SEGMENTATION_CONFIG,
    SentenceSegmentationConfig,
    SentenceSegmenter,
)

__all__ = [
    "CitationDiagnostic",
    "CitationDiagnosticCode",
    "CitationCleanupAction",
    "CitationForm",
    "CitationNormalizer",
    "CitationOccurrence",
    "CitationSpoilerAssessment",
    "CitationSpoilerError",
    "CitationSpoilerFilterResult",
    "CitationSpoilerReason",
    "Context",
    "ContextExtractionResult",
    "JATSArticle",
    "JATSDiagnostic",
    "JATSParseError",
    "JATSParser",
    "JATSSection",
    "DatasetLayout",
    "MaterializedDataset",
    "NormalizedCitationText",
    "NormalizedAuthorLists",
    "ParserDiagnostic",
    "ParserDiagnosticCode",
    "PMCIngestionPipeline",
    "PMCSearch",
    "PMCSearchResponseError",
    "ProcessingRecord",
    "ParsedJATSResult",
    "PositionedSentence",
    "PubMedMeSHClient",
    "CrossContextCoreferenceFilter",
    "QueryFilterPipeline",
    "Reference",
    "SentencePosition",
    "SentenceSegmentationConfig",
    "SentenceSegmenter",
    "DEFAULT_SENTENCE_SEGMENTATION_CONFIG",
    "author_lists_to_payload",
    "assess_residual_citation_spoiler",
    "assert_no_residual_citation_spoilers",
    "filter_citation_spoilers",
    "contexts_to_dataframe",
    "diagnostics_to_payload",
    "extract_abstracts_from_pubmed_xml",
    "extract_mesh_terms_from_pubmed_xml",
    "metadata_to_dataframe",
    "normalize_author_lists",
    "references_to_dataframe",
    "render_author_yaml",
    "render_diagnostics_json",
    "render_positioned_sentences",
]
