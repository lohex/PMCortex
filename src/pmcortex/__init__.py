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
from .jatsparser import JATSParser
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

__all__ = [
    "CitationDiagnostic",
    "CitationDiagnosticCode",
    "CitationForm",
    "CitationNormalizer",
    "CitationOccurrence",
    "CitationSpoilerAssessment",
    "CitationSpoilerError",
    "CitationSpoilerFilterResult",
    "CitationSpoilerReason",
    "JATSParser",
    "DatasetLayout",
    "MaterializedDataset",
    "NormalizedCitationText",
    "PMCIngestionPipeline",
    "PMCSearch",
    "PMCSearchResponseError",
    "ProcessingRecord",
    "PubMedMeSHClient",
    "CrossContextCoreferenceFilter",
    "QueryFilterPipeline",
    "assess_residual_citation_spoiler",
    "assert_no_residual_citation_spoilers",
    "filter_citation_spoilers",
    "extract_abstracts_from_pubmed_xml",
    "extract_mesh_terms_from_pubmed_xml",
]
