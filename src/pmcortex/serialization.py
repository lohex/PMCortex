"""Serialize immutable parser results into the current dataset artifacts."""

from dataclasses import asdict
from typing import TypedDict

import pandas as pd
import yaml

from pmcortex.models import (
    Context,
    JATSArticle,
    NormalizedAuthorLists,
    PositionedSentence,
    Reference,
)


METADATA_COLUMNS: tuple[str, ...] = (
    "pmcid",
    "pmid",
    "title",
    "abstract",
    "authors",
)
SOURCE_COLUMNS: tuple[str, ...] = (
    "rid",
    "label",
    "doi",
    "pmid",
    "pmcid",
    "year",
    "title",
    "journal",
    "authors",
    "source_pmcid",
)
CONTEXT_COLUMNS: tuple[str, ...] = (
    "source_pmcid",
    "section_index",
    "paragraph_index",
    "sentence_index",
    "query_raw",
    "query",
    "citation_forms",
    "citation_cleanup_action",
    "hits",
    "query_length",
    "n_hits",
    "context",
)


class AuthorListsPayload(TypedDict):
    """YAML-compatible mutable representation of normalized author lists."""

    authors: list[str]
    cited_authors: list[list[str]]


def metadata_to_dataframe(article: JATSArticle) -> pd.DataFrame:
    """Return one article's metadata in the baseline CSV column order."""
    if not isinstance(article, JATSArticle):
        raise TypeError("article must be a JATSArticle")
    row = {
        "pmcid": article.pmcid,
        "pmid": article.pmid,
        "title": article.title,
        "abstract": article.abstract,
        "authors": list(article.authors),
    }
    return pd.DataFrame([row], columns=METADATA_COLUMNS)


def references_to_dataframe(
    source_pmcid: str,
    references: tuple[Reference, ...],
) -> pd.DataFrame:
    """Return bibliographic references in the baseline source CSV schema."""
    if not isinstance(source_pmcid, str):
        raise TypeError("source_pmcid must be a string")
    if not source_pmcid.strip():
        raise ValueError("source_pmcid must not be empty")
    if not isinstance(references, tuple):
        raise TypeError("references must be a tuple")
    if not all(isinstance(reference, Reference) for reference in references):
        raise TypeError("references must contain only Reference values")

    rows = []
    for reference in references:
        row = asdict(reference)
        row["authors"] = list(reference.authors)
        row["source_pmcid"] = source_pmcid
        rows.append(row)
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def contexts_to_dataframe(contexts: tuple[Context, ...]) -> pd.DataFrame:
    """Return contexts flattened into the baseline context CSV schema."""
    if not isinstance(contexts, tuple):
        raise TypeError("contexts must be a tuple")
    if not all(isinstance(context, Context) for context in contexts):
        raise TypeError("contexts must contain only Context values")

    rows = [
        {
            "source_pmcid": context.position.source_pmcid,
            "section_index": context.position.section_index,
            "paragraph_index": context.position.paragraph_index,
            "sentence_index": context.position.sentence_index,
            "query_raw": context.query_raw,
            "query": context.query,
            "citation_forms": list(context.citation_forms),
            "citation_cleanup_action": context.citation_cleanup_action.value,
            "hits": list(context.hits),
            "query_length": context.query_length,
            "n_hits": context.n_hits,
            "context": context.context,
        }
        for context in contexts
    ]
    return pd.DataFrame(rows, columns=CONTEXT_COLUMNS)


def render_positioned_sentences(
    sentences: tuple[PositionedSentence, ...],
) -> str:
    """Render sentences in the pre-refactoring fulltext line format."""
    if not isinstance(sentences, tuple):
        raise TypeError("sentences must be a tuple")
    if not all(isinstance(sentence, PositionedSentence) for sentence in sentences):
        raise TypeError("sentences must contain only PositionedSentence values")

    return "\n".join(
        (
            f"{sentence.position.section_index}/"
            f"{sentence.position.paragraph_index}/"
            f"{sentence.position.sentence_index}\t"
            f"{sentence.text}"
        )
        for sentence in sentences
    )


def _short_author_name(name: str) -> str:
    """Convert a normalized `Surname Given` name into `Surname G`."""
    parts = [part for part in name.split() if part]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} {parts[1][0]}"


def normalize_author_lists(article: JATSArticle) -> NormalizedAuthorLists:
    """Build immutable shortened article and cited-author name collections."""
    if not isinstance(article, JATSArticle):
        raise TypeError("article must be a JATSArticle")
    authors = tuple(_short_author_name(name) for name in article.authors)
    cited_authors = tuple(
        tuple(_short_author_name(name) for name in reference.authors)
        for reference in article.references
    )
    return NormalizedAuthorLists(authors=authors, cited_authors=cited_authors)


def author_lists_to_payload(author_lists: NormalizedAuthorLists) -> AuthorListsPayload:
    """Convert immutable author lists into a YAML-compatible payload."""
    if not isinstance(author_lists, NormalizedAuthorLists):
        raise TypeError("author_lists must be a NormalizedAuthorLists")
    return {
        "authors": list(author_lists.authors),
        "cited_authors": [list(authors) for authors in author_lists.cited_authors],
    }


def render_author_yaml(author_lists: NormalizedAuthorLists) -> str:
    """Render normalized author lists using the baseline YAML representation."""
    payload = author_lists_to_payload(author_lists)
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=False,
    )
