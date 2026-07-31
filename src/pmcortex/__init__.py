"""Public API for PMCortex retrieval-dataset construction."""

from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, level="WARNING")
logger.add("logs/pmcortex.log", level="INFO")

from .jatsparser import JATSParser
from .mesh import (
    PubMedMeSHClient,
    extract_abstracts_from_pubmed_xml,
    extract_mesh_terms_from_pubmed_xml,
)
from .pmcsearch import PMCSearch
from .query_filters import CrossContextCoreferenceFilter, QueryFilterPipeline

__all__ = [
    "JATSParser",
    "PMCSearch",
    "PubMedMeSHClient",
    "CrossContextCoreferenceFilter",
    "QueryFilterPipeline",
    "extract_abstracts_from_pubmed_xml",
    "extract_mesh_terms_from_pubmed_xml",
]
