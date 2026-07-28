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
from .query_filters import QueryFilterPipeline

__all__ = [
    "JATSParser",
    "PubMedMeSHClient",
    "QueryFilterPipeline",
    "extract_abstracts_from_pubmed_xml",
    "extract_mesh_terms_from_pubmed_xml",
]
