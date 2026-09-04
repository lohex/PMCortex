"""Filter citation-derived retrieval queries and select hard negatives."""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from enum import StrEnum
import heapq
from pathlib import Path
import re

import pandas as pd
from loguru import logger


ET_AL_PATTERN = re.compile(r"\bet al\.", flags=re.IGNORECASE)
NON_ALNUM_SPACE_PATTERN = re.compile(r"[^a-zA-Z0-9 ]")
PARENTHETICAL_AUTHOR_YEAR_PATTERN = re.compile(
    r"\((?=[^()\n]{0,240}\b(?:18|19|20)\d{2}[a-z]?\b)"
    r"(?=[^()\n]{0,240}(?:[,;&]|\band\b|\bet\s+al\.))"
    r"[^()\n]{1,240}\)",
    flags=re.IGNORECASE,
)
NARRATIVE_AUTHOR_YEAR_PATTERN = re.compile(
    r"\b[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’‐‑–-]+"
    r"(?:\s+(?:et\s+al\.|and|&)\s*"
    r"[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’‐‑–-]+)?"
    r"\s*\(\s*(?:18|19|20)\d{2}[a-z]?\s*\)",
)
UNWRAPPED_AUTHOR_YEAR_PATTERN = re.compile(
    r"\b[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’‐‑–-]+"
    r"(?:\s+(?:et\s+al\.|and|&)\s*"
    r"[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'’‐‑–-]+)?"
    r"\s*,\s*(?:18|19|20)\d{2}[a-z]?\b",
)
NUMERIC_CITATION_PATTERN = re.compile(
    r"\[\s*\d{1,4}(?:[a-z])?"
    r"(?:\s*(?:[,;/]|[-–—−]|\band\b)\s*\d{1,4}(?:[a-z])?)*\s*\]",
    flags=re.IGNORECASE,
)
BIBLIOGRAPHIC_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:doi|pmid|pmcid)\s*:\s*(?:10\.)?[a-z0-9./_-]+",
    flags=re.IGNORECASE,
)
NONPUBLIC_CITATION_PATTERN = re.compile(
    r"\b(?:unpublished\s+data|personal\s+communication|in\s+press)\b",
    flags=re.IGNORECASE,
)
EMPTY_CITATION_PLACEHOLDER_PATTERN = re.compile(
    r"\(\s*(?:e\.g\.|i\.e\.)\s*,?\s*\)",
    flags=re.IGNORECASE,
)
PUBLISHER_CITATION_ARTIFACT_PATTERN = re.compile(r"▸")
INTERNAL_XREF_PATTERN = re.compile(r"\[xref:[^\]\s]+\]", flags=re.IGNORECASE)
DEFAULT_ANAPHORIC_STARTS = (
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "the same",
    "the former",
    "the latter",
    "former",
    "latter",
    "other",
    "another",
    "such",
    "similar",
    "both",
    "each",
    "either",
    "neither",
)
DEFAULT_CONNECTIVES = (
    "therefore",
    "thus",
    "hence",
    "consequently",
    "accordingly",
    "as a result",
    "however",
    "nevertheless",
    "nonetheless",
    "still",
    "instead",
    "moreover",
    "furthermore",
    "further",
    "in addition",
    "additionally",
    "also",
    "for example",
    "for instance",
    "likewise",
    "similarly",
    "in contrast",
    "by contrast",
    "conversely",
    "specifically",
    "notably",
    "indeed",
    "meanwhile",
    "recently",
    "overall",
    "in summary",
    "in conclusion",
    "importantly",
    "interestingly",
    "surprisingly",
    "unfortunately",
    "it has been found that",
    "it was found that",
    "it is noteworthy that",
    "it should be noted that",
)
DEFAULT_REFERENCE_FRAMING_STARTS = (
    "according to",
    "as shown in",
    "as described in",
    "as reported in",
    "as mentioned in",
    "as stated in",
    "as summarized in",
    "as detailed in",
    "as outlined in",
    "as illustrated in",
    "as demonstrated in",
    "as evidenced by",
    "as revealed by",
    "as indicated by",
    "as documented in",
    "as highlighted in",
    "as emphasized in",
)
DEFAULT_LOCAL_CONTEXT_STARTS = (
    "figure",
    "fig.",
    "table",
    "section",
    "appendix",
    "supplementary figure",
    "supplementary fig.",
    "supplementary table",
    "see",
)
DEFAULT_SUMMARY_STARTS = (
    "taken together",
    "in summary",
    "overall",
    "in conclusion",
    "in brief",
    "in short",
    "to sum up",
    "to summarize",
    "in essence",
    "in a nutshell",
    "in a word",
    "all in all",
    "on the whole",
    "in general",
    "generally speaking",
)
LOCAL_CONTEXT_REFERENCE_PATTERN = re.compile(
    r"\b(?:figure|fig\.|table|section|appendix|supplementary\s+(?:figure|fig\.|table))\s+[a-z0-9ivx.-]+\b"
    r"|\b(?:as\s+(?:described|discussed|mentioned|reported|shown)|see)\s+(?:above|below)\b",
    flags=re.IGNORECASE,
)


class CitationSpoilerReason(StrEnum):
    """Closed set of reasons why a query exposes bibliographic information."""

    POSITIVE_AUTHOR = "positive_author"
    PARENTHETICAL_AUTHOR_YEAR = "parenthetical_author_year"
    NARRATIVE_AUTHOR_YEAR = "narrative_author_year"
    UNWRAPPED_AUTHOR_YEAR = "unwrapped_author_year"
    ET_AL = "et_al"
    NUMERIC_CITATION = "numeric_citation"
    BIBLIOGRAPHIC_IDENTIFIER = "bibliographic_identifier"
    NONPUBLIC_CITATION = "nonpublic_citation"
    EMPTY_CITATION_PLACEHOLDER = "empty_citation_placeholder"
    PUBLISHER_ARTIFACT = "publisher_artifact"
    INTERNAL_XREF_MARKER = "internal_xref_marker"
    INCOMPLETE_POSITIVE_AUTHOR_METADATA = "incomplete_positive_author_metadata"


@dataclass(frozen=True, slots=True)
class CitationSpoilerAssessment:
    """Residual citation-spoiler assessment for one query string."""

    is_spoiler: bool
    reasons: tuple[CitationSpoilerReason, ...]


@dataclass(frozen=True, slots=True)
class CitationSpoilerFilterResult:
    """Filtered benchmark rows and their auditable rejected complement."""

    filtered: pd.DataFrame
    rejected: pd.DataFrame


class CitationSpoilerError(RuntimeError):
    """Raised when a final benchmark export still contains citation spoilers."""


class CrossContextCoreferenceFilter:
    """Remove queries with anaphoric mentions linked to their preceding context."""

    def __init__(
        self,
        nlp: object,
        *,
        batch_size: int = 512,
        query_markers: Iterable[str] = DEFAULT_ANAPHORIC_STARTS,
    ) -> None:
        """Configure the filter with a spaCy-compatible pipeline containing FastCoref."""
        pipe = getattr(nlp, "pipe", None)
        if not callable(pipe):
            raise TypeError("nlp must provide a callable pipe method")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        markers = tuple(query_markers)
        if not markers:
            raise ValueError("query_markers must not be empty")

        self._nlp = nlp
        self.batch_size = batch_size
        self.query_markers = markers

    def filter_cross_context_coreferences(
        self,
        queries: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a copy without queries coreferentially linked to `context`."""
        required_columns = {"context", "query"}
        missing_columns = required_columns.difference(queries.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing coreference columns: {missing}")

        candidates = [
            (query_id, context, query)
            for query_id, context, query in queries[
                ["context", "query"]
            ].itertuples()
            if isinstance(context, str)
            and context.strip()
            and isinstance(query, str)
            and query.strip()
        ]
        if not candidates:
            return queries.copy()

        texts = [f"{context} {query}" for _, context, query in candidates]
        pipe = getattr(self._nlp, "pipe")
        documents = list(pipe(texts, batch_size=self.batch_size))
        if len(documents) != len(candidates):
            raise RuntimeError("The NLP pipeline returned an unexpected document count")
        dependent_query_ids = {
            query_id
            for document, (query_id, context, query) in zip(documents, candidates)
            if self._depends_on_context(document, context, query)
        }
        return queries.drop(index=dependent_query_ids).copy()

    def _depends_on_context(
        self,
        document: object,
        context: str,
        query: str,
    ) -> bool:
        """Return whether one coreference cluster crosses into the query."""
        extension = getattr(document, "_", None)
        clusters = getattr(extension, "coref_clusters", ())
        if clusters is None:
            return False
        full_text = f"{context} {query}"
        context_end = len(context)
        query_start = context_end + 1

        for cluster in clusters:
            spans = [(int(start), int(end)) for start, end in cluster]
            has_context_mention = any(end <= context_end for _, end in spans)
            query_mentions = [
                full_text[start:end]
                for start, end in spans
                if start >= query_start
            ]
            has_anaphoric_query_mention = any(
                _starts_with_any_phrase(mention, self.query_markers)
                for mention in query_mentions
            )
            if has_context_mention and has_anaphoric_query_mention:
                return True

        return False


class QueryFilterPipeline:
    """Filter extracted contexts by mutating `filtered_df` and returning `self`."""

    def __init__(self, data_dir: str | Path) -> None:
        """Load the three extraction CSV files from `data_dir`."""
        self.data_dir = Path(data_dir)
        required_files = (
            "extracted_contexts.csv",
            "extracted_metadata.csv",
            "extracted_sources.csv",
        )
        missing_files = [
            name for name in required_files if not (self.data_dir / name).is_file()
        ]
        if missing_files:
            missing = ", ".join(missing_files)
            raise FileNotFoundError(f"Missing pipeline input files: {missing}")

        logger.info("Initializing QueryFilterPipeline from {}", self.data_dir)

        self.df_contexts = pd.read_csv(
            self.data_dir / "extracted_contexts.csv",
            dtype={"hits": "string", "source_pmcid": "string"},
        )
        logger.info(
            "Loaded contexts from {} with {} rows",
            self.data_dir / "extracted_contexts.csv",
            len(self.df_contexts),
        )
        self.df_metainfo = pd.read_csv(
            self.data_dir / "extracted_metadata.csv",
            dtype={"pmcid": "string", "pmid": "string"},
        )
        logger.info(
            "Loaded metadata from {} with {} rows",
            self.data_dir / "extracted_metadata.csv",
            len(self.df_metainfo),
        )
        self.df_sources = pd.read_csv(
            self.data_dir / "extracted_sources.csv",
            dtype={"source_pmcid": "string", "pmid": "string"},
        )
        logger.info(
            "Loaded sources from {} with {} rows",
            self.data_dir / "extracted_sources.csv",
            len(self.df_sources),
        )

        self.annotated_df = self.df_contexts.copy()
        self.filtered_df = self.df_contexts.copy()
        self.citation_spoiler_audit = pd.DataFrame()
        self._metadata_with_authors: pd.DataFrame | None = None
        self._sources_with_authors: pd.DataFrame | None = None
        self._positive_author_patterns: (
            dict[tuple[str, str], re.Pattern[str] | None] | None
        ) = None
        self._self_citation_lookup: dict[str, set[str]] | None = None
        self._source_count_lookup: dict[str, int] | None = None
        logger.info(
            "Pipeline ready with {} raw queries; expensive annotations will be materialized lazily",
            len(self.filtered_df),
        )

    def reset(self) -> "QueryFilterPipeline":
        """Restore the unfiltered contexts and return this pipeline."""
        self.filtered_df = self.df_contexts.copy()
        self.citation_spoiler_audit = pd.DataFrame()
        logger.info("Reset filtered query set to {} rows", len(self.filtered_df))
        return self

    def filter_min_hit_queries(self, n: int = 1) -> "QueryFilterPipeline":
        """Keep queries containing exactly `n` cited positive documents."""
        if n <= 0:
            raise ValueError("n must be positive")
        self.filtered_df = self._apply_filter(
            f"filter_min_hit_queries(n={n})",
            self.filtered_df["n_hits"].eq(n),
        )
        return self

    def filter_no_et_al_queries(self) -> "QueryFilterPipeline":
        """Remove queries containing the phrase `et al.`."""
        self.filtered_df = self._apply_filter(
            "filter_no_et_al_queries",
            ~self.filtered_df["query"].fillna("").str.contains(ET_AL_PATTERN),
        )
        return self

    def filter_no_explicit_spoilers(self) -> "QueryFilterPipeline":
        """Remove queries exposing positive authors or residual citation syntax."""
        self._ensure_explicit_spoiler()
        self.filtered_df = self._apply_filter(
            "filter_no_explicit_spoilers",
            ~self.filtered_df["explicit_spoiler"].fillna(False),
        )
        return self

    def filter_no_citation_spoilers(self) -> "QueryFilterPipeline":
        """Remove every query flagged by metadata or residual citation checks."""
        self._ensure_explicit_spoiler()
        spoiler_mask = self.filtered_df["citation_spoiler"].fillna(False)
        audit_columns = [
            column
            for column in (
                "source_pmcid",
                "section_index",
                "paragraph_index",
                "sentence_index",
                "query_raw",
                "query",
                "hits_parsed",
                "citation_spoiler_reasons",
                "positive_author_metadata_complete",
            )
            if column in self.filtered_df.columns
        ]
        self.citation_spoiler_audit = self.filtered_df.loc[
            spoiler_mask,
            audit_columns,
        ].copy()
        self.citation_spoiler_audit["rejection_stage"] = (
            "filter_no_citation_spoilers"
        )
        self.filtered_df = self._apply_filter(
            "filter_no_citation_spoilers",
            ~spoiler_mask,
        )
        return self

    def assert_no_citation_spoilers(self) -> "QueryFilterPipeline":
        """Raise when the current result violates the no-spoiler invariant.

        Returns:
            This pipeline when no query contains a citation spoiler.

        Raises:
            CitationSpoilerError: If at least one current query is flagged.
        """
        self._ensure_explicit_spoiler()
        spoiler_rows = self.filtered_df.loc[
            self.filtered_df["citation_spoiler"].fillna(False),
            ["query", "citation_spoiler_reasons"],
        ]
        if spoiler_rows.empty:
            return self

        examples = spoiler_rows.head(3).to_dict(orient="records")
        raise CitationSpoilerError(
            f"Refusing benchmark export with {len(spoiler_rows)} citation "
            f"spoilers; examples={examples!r}"
        )

    def filter_no_self_citations(self) -> "QueryFilterPipeline":
        """Remove queries whose cited papers include a self-citation."""
        self._ensure_self_citation()
        self.filtered_df = self._apply_filter(
            "filter_no_self_citations",
            self.filtered_df["self_citation"].eq(0),
        )
        return self

    def filter_min_source_count(self, n: int) -> "QueryFilterPipeline":
        """Keep queries from articles with at least `n` indexed references."""
        if n <= 0:
            raise ValueError("n must be positive")
        self._ensure_n_sources()
        self.filtered_df = self._apply_filter(
            f"filter_min_source_count(n={n})",
            self.filtered_df["n_sources"].ge(n),
        )
        return self

    def filter_max_strangeness(self, threshold: float = 0.3) -> "QueryFilterPipeline":
        """Keep queries whose non-alphanumeric character ratio is at most `threshold`."""
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self._ensure_strangeness()
        self.filtered_df = self._apply_filter(
            f"filter_max_strangeness(threshold={threshold})",
            self.filtered_df["strangeness"].le(threshold),
        )
        return self

    def filter_min_query_length(self, n: int) -> "QueryFilterPipeline":
        """Keep queries containing at least `n` words."""
        if n <= 0:
            raise ValueError("n must be positive")
        self._ensure_query_length()
        self.filtered_df = self._apply_filter(
            f"filter_min_query_length(n={n})",
            self.filtered_df["query_length"].ge(n),
        )
        return self

    def filter_max_query_length(self, n: int) -> "QueryFilterPipeline":
        """Keep queries containing at most `n` words."""
        if n <= 0:
            raise ValueError("n must be positive")
        self._ensure_query_length()
        self.filtered_df = self._apply_filter(
            f"filter_max_query_length(n={n})",
            self.filtered_df["query_length"].le(n),
        )
        return self

    def filter_no_anaphoric_starts(
        self,
        starts: Iterable[str] | None = None,
    ) -> "QueryFilterPipeline":
        """Remove queries that begin with an anaphoric expression."""
        selected_starts = DEFAULT_ANAPHORIC_STARTS if starts is None else tuple(starts)
        return self._filter_query_starts(
            selected_starts,
            "filter_no_anaphoric_starts",
        )

    def filter_no_reference_framing_starts(
        self,
        starts: Iterable[str] | None = None,
    ) -> "QueryFilterPipeline":
        """Remove queries that begin with explicit reference framing."""
        selected_starts = (
            DEFAULT_REFERENCE_FRAMING_STARTS if starts is None else tuple(starts)
        )
        return self._filter_query_starts(
            selected_starts,
            "filter_no_reference_framing_starts",
        )

    def filter_no_summary_starts(
        self,
        starts: Iterable[str] | None = None,
    ) -> "QueryFilterPipeline":
        """Remove queries that begin with a discourse-summary expression."""
        selected_starts = DEFAULT_SUMMARY_STARTS if starts is None else tuple(starts)
        return self._filter_query_starts(
            selected_starts,
            "filter_no_summary_starts",
        )

    def filter_self_contained_queries(self) -> "QueryFilterPipeline":
        """Remove explicit document links while leaving anaphora to Coreference."""
        starts = (
            *DEFAULT_REFERENCE_FRAMING_STARTS,
            *DEFAULT_LOCAL_CONTEXT_STARTS,
        )
        starts_mask = self.filtered_df["query"].fillna("").apply(
            lambda query: _starts_with_any_phrase(query, starts)
        )
        local_reference_mask = self.filtered_df["query"].fillna("").str.contains(
            LOCAL_CONTEXT_REFERENCE_PATTERN
        )
        self.filtered_df = self._apply_filter(
            "filter_self_contained_queries",
            ~(starts_mask | local_reference_mask),
        )
        return self

    def filter_cross_context_coreferences(
        self,
        coreference_filter: CrossContextCoreferenceFilter,
    ) -> "QueryFilterPipeline":
        """Remove queries whose anaphoric mentions resolve into `context`."""
        filtered = coreference_filter.filter_cross_context_coreferences(
            self.filtered_df
        )
        self.filtered_df = self._apply_filter(
            "filter_cross_context_coreferences",
            self.filtered_df.index.isin(filtered.index),
        )
        return self

    def cleanup_queries(
        self,
        connectives: Iterable[str] | None = None,
    ) -> "QueryFilterPipeline":
        """Remove leading discourse connectives from the current queries."""
        selected_connectives = (
            DEFAULT_CONNECTIVES if connectives is None else tuple(connectives)
        )
        self.filtered_df = self.filtered_df.copy()
        self.filtered_df["query"] = self.filtered_df["query"].apply(
            lambda query: _cleanup_query_text(query, selected_connectives)
        )
        self._invalidate_query_dependent_annotations()
        logger.info("Cleaned leading connectives for {} rows", len(self.filtered_df))
        return self

    def select_shared_reference_subset(
        self,
        n: int,
        relevant_pmcids: Iterable[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Select exactly `n` decoy references per query while reusing references
        globally when possible.

        The decoys are drawn from the source article's reference list but must
        not overlap with that query's `hits`. The implementation uses a greedy
        multicover heuristic: repeatedly pick the reference that still helps
        the largest number of queries, then prune redundant references
        afterwards.

        If `relevant_pmcids` is provided, decoys are computed only for queries
        whose `source_pmcid` is in that subset. The internal `filtered_df`
        remains unchanged.

        Returns:
            tuple[pd.DataFrame, list[str]]:
                A copy of the selected query subset with `selected_decoys` and
                `n_selected_decoys`, plus the globally selected reference PMIDs.
        """
        if n <= 0:
            raise ValueError("n must be positive")

        relevant_pmcid_set = None
        if relevant_pmcids is not None:
            relevant_pmcid_set = {
                str(pmcid)
                for pmcid in relevant_pmcids
                if pmcid is not None and not pd.isna(pmcid)
            }

        working_df = self.filtered_df
        if relevant_pmcid_set is not None:
            working_df = working_df.loc[
                working_df["source_pmcid"].isin(relevant_pmcid_set)
            ].copy()
        else:
            working_df = working_df.copy()

        if "hits_parsed" not in working_df.columns:
            working_df["hits_parsed"] = working_df["hits"].apply(parse_hits)

        query_decoy_options = _build_query_decoy_options(
            working_df,
            self.df_sources,
            n,
        )
        selected_refs = _select_shared_references_greedy(query_decoy_options, n)
        selected_ref_set = _prune_redundant_selected_references(query_decoy_options, selected_refs, n)
        selected_by_query = {
            query_id: [ref for ref in refs if ref in selected_ref_set][:n]
            for query_id, refs in query_decoy_options.items()
        }

        selected_df = working_df.copy()
        selected_df["selected_decoys"] = [
            list(selected_by_query[query_id])
            for query_id in selected_df.index
        ]
        selected_df["n_selected_decoys"] = selected_df["selected_decoys"].apply(len)

        if not selected_df["n_selected_decoys"].eq(n).all():
            raise RuntimeError("Failed to assign exactly n decoy references to every query")

        return selected_df, sorted(selected_ref_set)

    def _filter_query_starts(
        self,
        starts: Iterable[str],
        filter_name: str,
    ) -> "QueryFilterPipeline":
        """Remove queries beginning with one of `starts`."""
        selected_starts = tuple(starts)
        keep_mask = ~self.filtered_df["query"].fillna("").apply(
            lambda query: _starts_with_any_phrase(query, selected_starts)
        )
        self.filtered_df = self._apply_filter(
            f"{filter_name}(starts={selected_starts})",
            keep_mask,
        )
        return self

    def _apply_filter(self, filter_name: str, mask: pd.Series) -> pd.DataFrame:
        """Return a filtered copy and log the number of removed rows."""
        before = len(self.filtered_df)
        filtered = self.filtered_df.loc[mask].copy()
        after = len(filtered)
        logger.info(
            "{} reduced queries from {} to {} rows",
            filter_name,
            before,
            after,
        )
        return filtered

    def _ensure_hits_parsed(self) -> None:
        """Materialize normalized citation-hit lists when absent."""
        if "hits_parsed" in self.filtered_df.columns:
            return

        logger.info("Materializing hits_parsed for {} rows", len(self.filtered_df))
        self.filtered_df["hits_parsed"] = self.filtered_df["hits"].apply(parse_hits)

    def _ensure_n_sources(self) -> None:
        """Materialize the number of unique references per source article."""
        if "n_sources" in self.filtered_df.columns:
            return

        source_count_lookup = self._get_source_count_lookup()
        logger.info("Materializing n_sources for {} rows", len(self.filtered_df))
        self.filtered_df["n_sources"] = (
            self.filtered_df["source_pmcid"].map(source_count_lookup).fillna(0).astype(int)
        )

    def _ensure_query_length(self) -> None:
        """Materialize or normalize query word counts."""
        if "query_length" in self.filtered_df.columns:
            self.filtered_df["query_length"] = (
                pd.to_numeric(self.filtered_df["query_length"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            return

        logger.info("Materializing query_length for {} rows", len(self.filtered_df))
        self.filtered_df["query_length"] = self.filtered_df["query"].apply(_count_query_words)

    def _ensure_strangeness(self) -> None:
        """Materialize or normalize the query strangeness score."""
        if "strangeness" in self.filtered_df.columns:
            self.filtered_df["strangeness"] = pd.to_numeric(
                self.filtered_df["strangeness"],
                errors="coerce",
            ).fillna(1.0)
            return

        logger.info("Materializing strangeness for {} rows", len(self.filtered_df))
        self.filtered_df["strangeness"] = self.filtered_df["query"].apply(_compute_strangeness)

    def _ensure_self_citation(self) -> None:
        """Materialize the number of positive documents that are self-citations."""
        if "self_citation" in self.filtered_df.columns:
            return

        self._ensure_hits_parsed()
        self_citation_lookup = self._get_self_citation_lookup()
        logger.info("Materializing self_citation for {} rows", len(self.filtered_df))
        self.filtered_df["self_citation"] = self.filtered_df.apply(
            lambda row: len(
                set(row["hits_parsed"]).intersection(
                    self_citation_lookup.get(row["source_pmcid"], set())
                )
            ),
            axis=1,
        )

    def _ensure_explicit_spoiler(self) -> None:
        """Materialize metadata and residual citation-spoiler assessments."""
        if "explicit_spoiler" in self.filtered_df.columns:
            return

        self._ensure_hits_parsed()
        positive_author_patterns = self._get_positive_author_patterns()
        logger.info(
            "Materializing citation-spoiler assessments for {} rows",
            len(self.filtered_df),
        )
        assessments = self.filtered_df.apply(
            lambda row: _assess_query_with_positive_authors(
                row["query"],
                row["source_pmcid"],
                row["hits_parsed"],
                positive_author_patterns,
            ),
            axis=1,
        )
        self.filtered_df["author_spoiling"] = assessments.apply(
            lambda assessment: CitationSpoilerReason.POSITIVE_AUTHOR
            in assessment.reasons
        )
        self.filtered_df["citation_spoiler"] = assessments.apply(
            lambda assessment: assessment.is_spoiler
        )
        self.filtered_df["citation_spoiler_reasons"] = assessments.apply(
            lambda assessment: [reason.value for reason in assessment.reasons]
        )
        self.filtered_df["positive_author_metadata_complete"] = self.filtered_df.apply(
            lambda row: _positive_author_metadata_is_complete(
                row["source_pmcid"],
                row["hits_parsed"],
                positive_author_patterns,
            ),
            axis=1,
        )
        self.filtered_df["explicit_spoiler"] = self.filtered_df["citation_spoiler"]

    def _get_metadata_with_authors(self) -> pd.DataFrame:
        """Return cached article metadata with normalized author names."""
        if self._metadata_with_authors is None:
            logger.info("Building normalized metadata author cache")
            self._metadata_with_authors = self.df_metainfo.assign(
                author_normalized=self.df_metainfo["authors"].apply(_normalize_metadata_authors)
            )
        return self._metadata_with_authors

    def _get_sources_with_authors(self) -> pd.DataFrame:
        """Return cached references with normalized author names."""
        if self._sources_with_authors is None:
            logger.info("Building normalized source author cache")
            self._sources_with_authors = self.df_sources.assign(
                author_normalized=self.df_sources["authors"].apply(_normalize_source_authors)
            )
        return self._sources_with_authors

    def _get_positive_author_patterns(
        self,
    ) -> dict[tuple[str, str], re.Pattern[str] | None]:
        """Return cached author patterns keyed by source article and positive PMID."""
        if self._positive_author_patterns is None:
            logger.info("Building positive-document author pattern cache")
            positive_author_lookup = _build_positive_author_lookup(
                self._get_sources_with_authors()
            )
            self._positive_author_patterns = {
                key: _compile_cited_author_pattern(surnames)
                for key, surnames in positive_author_lookup.items()
            }
        return self._positive_author_patterns

    def _get_self_citation_lookup(self) -> dict[str, set[str]]:
        """Return cached self-cited PMIDs keyed by source PMCID."""
        if self._self_citation_lookup is None:
            logger.info("Building self-citation lookup cache")
            self._self_citation_lookup = _build_self_citation_lookup(
                self._get_sources_with_authors(),
                self._get_metadata_with_authors(),
            )
        return self._self_citation_lookup

    def _get_source_count_lookup(self) -> dict[str, int]:
        """Return cached unique-reference counts keyed by source PMCID."""
        if self._source_count_lookup is None:
            logger.info("Building source-count lookup cache")
            self._source_count_lookup = _build_source_count_lookup(self.df_sources)
        return self._source_count_lookup

    def _invalidate_query_dependent_annotations(self) -> None:
        """Drop annotations invalidated by editing query text."""
        annotation_columns = (
            "query_length",
            "strangeness",
            "author_spoiling",
            "explicit_spoiler",
            "citation_spoiler",
            "citation_spoiler_reasons",
            "positive_author_metadata_complete",
        )
        for column in annotation_columns:
            if column in self.filtered_df.columns:
                self.filtered_df = self.filtered_df.drop(columns=column)


def normalize_author_name(author: str) -> str:
    """Reduce an author name to a clean surname followed by initials.

    Args:
        author: One author name in surname-first JATS order.

    Returns:
        Normalized surname and compact initials, or an empty string.

    Raises:
        TypeError: If ``author`` is not a string.
    """
    if not isinstance(author, str):
        raise TypeError("author must be a string")

    parts = [part for part in author.split() if part]
    if not parts:
        return ""

    surname = parts[0].strip("[]'\"")
    initials = "".join(part[0] for part in parts[1:] if part)
    return f"{surname} {initials}".strip()


def parse_hits(raw_hits: object) -> list[str]:
    """Parse a serialized citation-hit list and normalize its PMIDs."""
    if isinstance(raw_hits, list):
        values = raw_hits
    elif pd.isna(raw_hits):
        return []
    else:
        values = literal_eval(str(raw_hits))

    parsed_hits: list[str] = []
    for value in values:
        normalized = _coerce_pmid(value)
        if normalized is None:
            continue
        parsed_hits.append(normalized)
    return parsed_hits


def _normalize_metadata_authors(raw_authors: object) -> list[str]:
    """Normalize an article-author list loaded from CSV."""
    return [
        normalized
        for normalized in (
            normalize_author_name(author)
            for author in _parse_author_values(raw_authors)
        )
        if normalized
    ]


def _normalize_source_authors(raw_authors: object) -> list[str]:
    """Normalize reference authors loaded from a serialized CSV field."""
    return _normalize_metadata_authors(raw_authors)


def _parse_author_values(raw_authors: object) -> list[str]:
    """Parse list-serialized or legacy comma-separated author values."""
    if raw_authors is None:
        return []

    if isinstance(raw_authors, float) and pd.isna(raw_authors):
        return []

    values: object = raw_authors
    if isinstance(raw_authors, str):
        serialized = raw_authors.strip()
        if not serialized:
            return []
        is_serialized_collection = serialized.startswith(("[", "("))
        if is_serialized_collection:
            try:
                values = literal_eval(serialized)
            except (SyntaxError, ValueError) as error:
                raise ValueError("invalid serialized author list") from error
        else:
            values = [part.strip() for part in serialized.split(",")]

    if not isinstance(values, (list, tuple)):
        raise TypeError("authors must be a list, tuple, or serialized string")

    authors: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("author entries must be strings")
        normalized = value.strip()
        if normalized:
            authors.append(normalized)
    return authors


def _build_positive_author_lookup(
    df_sources: pd.DataFrame,
) -> dict[tuple[str, str], set[str]]:
    """Collect author surnames for each source-article/reference pair."""
    normalized_sources = df_sources[[
        "source_pmcid",
        "pmid",
        "author_normalized",
    ]].copy()
    normalized_sources["pmid"] = normalized_sources["pmid"].apply(_coerce_pmid)
    normalized_sources = normalized_sources.dropna(
        subset=["source_pmcid", "pmid"]
    )
    grouped = (
        normalized_sources.groupby(["source_pmcid", "pmid"])[
            "author_normalized"
        ]
        .agg(_collect_surnames)
    )
    return grouped.to_dict()


def _build_self_citation_lookup(
    df_sources: pd.DataFrame,
    df_metainfo: pd.DataFrame,
) -> dict[str, set[str]]:
    """Collect self-cited PMIDs for each source article."""
    merged = pd.merge(
        df_sources[["pmid", "source_pmcid", "author_normalized"]],
        df_metainfo[["pmcid", "author_normalized"]],
        left_on="source_pmcid",
        right_on="pmcid",
    ).rename(
        columns={
            "author_normalized_x": "source_authors",
            "author_normalized_y": "main_authors",
        }
    )

    merged = merged.dropna(subset=["pmid"]).copy()
    merged["pmid"] = merged["pmid"].apply(_coerce_pmid)
    merged["self_citation"] = merged.apply(
        lambda row: bool(set(row["source_authors"]).intersection(row["main_authors"])),
        axis=1,
    )

    self_citing = merged.loc[merged["self_citation"], ["source_pmcid", "pmid"]]
    grouped = self_citing.groupby("source_pmcid")["pmid"].agg(
        lambda values: {value for value in values if value is not None}
    )
    return grouped.to_dict()


def _build_source_count_lookup(df_sources: pd.DataFrame) -> dict[str, int]:
    """Count unique valid PMIDs for each source article."""
    valid_sources = (
        df_sources[["source_pmcid", "pmid"]]
        .assign(pmid=lambda frame: frame["pmid"].apply(_coerce_pmid))
        .dropna(subset=["pmid", "source_pmcid"])
    )
    valid_sources = valid_sources.drop_duplicates(subset=["source_pmcid", "pmid"])
    grouped = valid_sources.groupby("source_pmcid")["pmid"].size()
    return grouped.astype(int).to_dict()


def _collect_surnames(authors_per_reference: Iterable[list[str]]) -> set[str]:
    """Collect unique surnames from grouped reference-author lists."""
    surnames: set[str] = set()
    for author_list in authors_per_reference:
        for author in author_list:
            parts = str(author).split()
            if parts:
                surnames.add(parts[0])
    return surnames


def _contains_cited_author(
    query: object,
    cited_author_pattern: re.Pattern[str] | None,
) -> bool:
    """Return whether a query contains a cited-author surname."""
    if not isinstance(query, str) or cited_author_pattern is None:
        return False

    return bool(cited_author_pattern.search(query))


def assess_residual_citation_spoiler(query: str) -> CitationSpoilerAssessment:
    """Detect high-confidence bibliographic residue in one query.

    Args:
        query: Final or intermediate retrieval-query text.

    Returns:
        Immutable spoiler flag and all matched reason categories.

    Raises:
        TypeError: If ``query`` is not a string.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    reasons: list[CitationSpoilerReason] = []
    if PARENTHETICAL_AUTHOR_YEAR_PATTERN.search(query):
        reasons.append(CitationSpoilerReason.PARENTHETICAL_AUTHOR_YEAR)
    elif NARRATIVE_AUTHOR_YEAR_PATTERN.search(query):
        reasons.append(CitationSpoilerReason.NARRATIVE_AUTHOR_YEAR)
    elif UNWRAPPED_AUTHOR_YEAR_PATTERN.search(query):
        reasons.append(CitationSpoilerReason.UNWRAPPED_AUTHOR_YEAR)

    reason_patterns = (
        (CitationSpoilerReason.ET_AL, ET_AL_PATTERN),
        (CitationSpoilerReason.NUMERIC_CITATION, NUMERIC_CITATION_PATTERN),
        (
            CitationSpoilerReason.BIBLIOGRAPHIC_IDENTIFIER,
            BIBLIOGRAPHIC_IDENTIFIER_PATTERN,
        ),
        (
            CitationSpoilerReason.NONPUBLIC_CITATION,
            NONPUBLIC_CITATION_PATTERN,
        ),
        (
            CitationSpoilerReason.EMPTY_CITATION_PLACEHOLDER,
            EMPTY_CITATION_PLACEHOLDER_PATTERN,
        ),
        (
            CitationSpoilerReason.PUBLISHER_ARTIFACT,
            PUBLISHER_CITATION_ARTIFACT_PATTERN,
        ),
        (
            CitationSpoilerReason.INTERNAL_XREF_MARKER,
            INTERNAL_XREF_PATTERN,
        ),
    )
    reasons.extend(
        reason for reason, pattern in reason_patterns if pattern.search(query)
    )
    return CitationSpoilerAssessment(
        is_spoiler=bool(reasons),
        reasons=tuple(reasons),
    )


def assert_no_residual_citation_spoilers(
    queries: pd.DataFrame,
    *,
    query_column: str = "query",
) -> None:
    """Enforce the final no-residual-citation invariant for a table.

    Args:
        queries: Candidate benchmark rows.
        query_column: Column containing query strings.

    Raises:
        TypeError: If ``queries`` is not a DataFrame or the column name is invalid.
        ValueError: If the requested query column is absent or contains non-strings.
        CitationSpoilerError: If at least one query contains a residual spoiler.
    """
    if not isinstance(queries, pd.DataFrame):
        raise TypeError("queries must be a pandas DataFrame")
    if not isinstance(query_column, str):
        raise TypeError("query_column must be a string")
    if not query_column.strip():
        raise ValueError("query_column must not be empty")
    if query_column not in queries.columns:
        raise ValueError(f"Missing query column: {query_column}")

    invalid_mask = ~queries[query_column].apply(lambda value: isinstance(value, str))
    if invalid_mask.any():
        raise ValueError("query column must contain only strings")

    assessments = queries[query_column].apply(assess_residual_citation_spoiler)
    spoiler_mask = assessments.apply(lambda assessment: assessment.is_spoiler)
    if not spoiler_mask.any():
        return

    examples = [
        {
            "index": index,
            "query": queries.at[index, query_column],
            "reasons": [reason.value for reason in assessments.at[index].reasons],
        }
        for index in queries.index[spoiler_mask][:3]
    ]
    raise CitationSpoilerError(
        f"Refusing benchmark export with {int(spoiler_mask.sum())} residual "
        f"citation spoilers; examples={examples!r}"
    )


def filter_citation_spoilers(
    queries: pd.DataFrame,
    sources: pd.DataFrame,
    *,
    query_column: str = "query",
    hits_column: str = "hits_parsed",
    source_column: str = "source_pmcid",
    require_complete_author_metadata: bool = False,
) -> CitationSpoilerFilterResult:
    """Filter an existing benchmark with residual and positive-author checks.

    Args:
        queries: Existing benchmark rows to assess without mutation.
        sources: Reference metadata containing source PMCID, PMID, and authors.
        query_column: Query-text column in ``queries``.
        hits_column: Serialized or materialized positive-PMID column in ``queries``.
        source_column: Source-article column shared by both tables.
        require_complete_author_metadata: Reject rows whose positive-reference
            authors cannot all be checked. This is recommended for legacy data
            lacking structural citation-cleanup metadata.

    Returns:
        Independent filtered and rejected DataFrames with audit annotations.

    Raises:
        TypeError: If either table or a column-name argument has the wrong type.
        ValueError: If required columns are absent or author fields are malformed.
    """
    if not isinstance(queries, pd.DataFrame):
        raise TypeError("queries must be a pandas DataFrame")
    if not isinstance(sources, pd.DataFrame):
        raise TypeError("sources must be a pandas DataFrame")
    if not isinstance(require_complete_author_metadata, bool):
        raise TypeError("require_complete_author_metadata must be a bool")
    column_names = (query_column, hits_column, source_column)
    if any(not isinstance(column, str) for column in column_names):
        raise TypeError("column names must be strings")
    if any(not column.strip() for column in column_names):
        raise ValueError("column names must not be empty")

    missing_query_columns = {
        query_column,
        hits_column,
        source_column,
    }.difference(queries.columns)
    if missing_query_columns:
        missing = ", ".join(sorted(missing_query_columns))
        raise ValueError(f"Missing query columns: {missing}")
    missing_source_columns = {
        source_column,
        "pmid",
        "authors",
    }.difference(sources.columns)
    if missing_source_columns:
        missing = ", ".join(sorted(missing_source_columns))
        raise ValueError(f"Missing source columns: {missing}")

    normalized_sources = sources.assign(
        author_normalized=sources["authors"].apply(_normalize_source_authors)
    )
    positive_author_lookup = _build_positive_author_lookup(normalized_sources)
    positive_author_patterns = {
        key: _compile_cited_author_pattern(surnames)
        for key, surnames in positive_author_lookup.items()
    }

    annotated = queries.copy()
    parsed_hits = annotated[hits_column].apply(parse_hits)
    assessments = pd.Series(
        (
            _assess_query_with_positive_authors(
                query,
                source_pmcid,
                hit_pmids,
                positive_author_patterns,
            )
            for query, source_pmcid, hit_pmids in zip(
                annotated[query_column],
                annotated[source_column],
                parsed_hits,
            )
        ),
        index=annotated.index,
    )
    annotated["citation_spoiler"] = assessments.apply(
        lambda assessment: assessment.is_spoiler
    )
    annotated["citation_spoiler_reasons"] = assessments.apply(
        lambda assessment: [reason.value for reason in assessment.reasons]
    )
    annotated["positive_author_metadata_complete"] = [
        _positive_author_metadata_is_complete(
            source_pmcid,
            hit_pmids,
            positive_author_patterns,
        )
        for source_pmcid, hit_pmids in zip(
            annotated[source_column],
            parsed_hits,
        )
    ]

    if require_complete_author_metadata:
        incomplete_metadata_mask = ~annotated[
            "positive_author_metadata_complete"
        ]
        annotated.loc[incomplete_metadata_mask, "citation_spoiler"] = True
        annotated.loc[
            incomplete_metadata_mask,
            "citation_spoiler_reasons",
        ] = annotated.loc[
            incomplete_metadata_mask,
            "citation_spoiler_reasons",
        ].apply(
            lambda reasons: [
                *reasons,
                CitationSpoilerReason.INCOMPLETE_POSITIVE_AUTHOR_METADATA.value,
            ]
        )

    spoiler_mask = annotated["citation_spoiler"]
    rejected = annotated.loc[spoiler_mask].copy()
    rejected["rejection_stage"] = "filter_citation_spoilers"
    filtered = annotated.loc[~spoiler_mask].copy()
    assert_no_residual_citation_spoilers(filtered, query_column=query_column)
    return CitationSpoilerFilterResult(filtered=filtered, rejected=rejected)


def _assess_query_with_positive_authors(
    query: object,
    source_pmcid: object,
    positive_pmids: Iterable[str],
    positive_author_patterns: dict[tuple[str, str], re.Pattern[str] | None],
) -> CitationSpoilerAssessment:
    """Combine residual syntax checks with query-specific positive authors."""
    if not isinstance(query, str):
        return CitationSpoilerAssessment(is_spoiler=False, reasons=())

    residual = assess_residual_citation_spoiler(query)
    reasons = list(residual.reasons)
    if isinstance(source_pmcid, str):
        contains_positive_author = any(
            _contains_cited_author(
                query,
                positive_author_patterns.get((source_pmcid, pmid)),
            )
            for pmid in positive_pmids
        )
        if contains_positive_author:
            reasons.insert(0, CitationSpoilerReason.POSITIVE_AUTHOR)

    unique_reasons = tuple(dict.fromkeys(reasons))
    return CitationSpoilerAssessment(
        is_spoiler=bool(unique_reasons),
        reasons=unique_reasons,
    )


def _positive_author_metadata_is_complete(
    source_pmcid: object,
    positive_pmids: Iterable[str],
    positive_author_patterns: dict[tuple[str, str], re.Pattern[str] | None],
) -> bool:
    """Return whether every positive document has at least one parsed author."""
    if not isinstance(source_pmcid, str):
        return False
    pmids = tuple(positive_pmids)
    if not pmids:
        return False
    return all(
        positive_author_patterns.get((source_pmcid, pmid)) is not None
        for pmid in pmids
    )


def _compile_cited_author_pattern(
    cited_surnames: set[str],
) -> re.Pattern[str] | None:
    """Compile one case-insensitive pattern for a set of surnames."""
    if not cited_surnames:
        return None

    parts = sorted(
        (re.escape(surname) for surname in cited_surnames),
        key=len,
        reverse=True,
    )
    return re.compile(rf"\b(?:{'|'.join(parts)})\b", flags=re.IGNORECASE)


def _coerce_pmid(value: object) -> str | None:
    """Normalize a CSV PMID value to a non-empty string."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, str) and not value.strip():
        return None

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, "g")

    value_str = str(value).strip()
    if not value_str:
        return None

    if re.fullmatch(r"\d+\.0+", value_str):
        return value_str.split(".", 1)[0]

    return value_str


def _compute_strangeness(query: object) -> float:
    """Return the fraction of non-alphanumeric, non-space characters."""
    if not isinstance(query, str) or not query:
        return 1.0

    strange_chars = len(NON_ALNUM_SPACE_PATTERN.findall(query))
    return strange_chars / len(query)


def _count_query_words(query: object) -> int:
    """Count whitespace-delimited words in a string query."""
    if not isinstance(query, str):
        return 0

    return len(query.split())


def _starts_with_any_phrase(query: object, phrases: Iterable[str]) -> bool:
    """Return whether a query starts with a complete phrase."""
    if not isinstance(query, str):
        return False

    normalized = query.strip().lower()
    for phrase in phrases:
        phrase_normalized = str(phrase).strip().lower()
        if not phrase_normalized:
            continue
        if normalized == phrase_normalized:
            return True
        if normalized.startswith(f"{phrase_normalized} "):
            return True
        if normalized.startswith(f"{phrase_normalized},"):
            return True
        if normalized.startswith(f"{phrase_normalized}:"):
            return True
        if normalized.startswith(f"{phrase_normalized};"):
            return True
    return False

def _cleanup_query_text(query: object, connectives: Iterable[str]) -> object:
    """Remove repeated leading connectives from a string query."""
    if not isinstance(query, str):
        return query

    cleaned = query.strip()
    while cleaned and _starts_with_any_phrase(cleaned, connectives):
        matched_prefix = _matching_start_phrase(cleaned, connectives)
        if matched_prefix is None:
            break
        cleaned = cleaned[len(matched_prefix):].lstrip(" ,;:-")

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return cleaned

    return cleaned[0].upper() + cleaned[1:]


def _matching_start_phrase(query: str, phrases: Iterable[str]) -> str | None:
    """Return the longest phrase matching the start of a query."""
    query_stripped = query.strip()
    query_lower = query_stripped.lower()
    for phrase in sorted(
        {str(phrase).strip() for phrase in phrases if str(phrase).strip()},
        key=len,
        reverse=True,
    ):
        phrase_lower = phrase.lower()
        if query_lower == phrase_lower:
            return query_stripped[:len(phrase)]
        if (
            query_lower.startswith(f"{phrase_lower} ")
            or query_lower.startswith(f"{phrase_lower},")
            or query_lower.startswith(f"{phrase_lower}:")
            or query_lower.startswith(f"{phrase_lower};")
        ):
            return query_stripped[:len(phrase)]
    return None


def _build_source_reference_lookup(
    filtered_df: pd.DataFrame,
    df_sources: pd.DataFrame,
) -> dict[Hashable, list[str]]:
    """Map each relevant source article to its unique reference PMIDs."""
    required_pmcs = filtered_df["source_pmcid"].dropna().drop_duplicates().tolist()
    source_refs = (
        df_sources[["source_pmcid", "pmid"]]
        .assign(pmid=lambda frame: frame["pmid"].apply(_coerce_pmid))
        .dropna(subset=["pmid", "source_pmcid"])
        .loc[lambda frame: frame["source_pmcid"].isin(required_pmcs)]
        .drop_duplicates()
    )

    source_ref_lookup: dict[Hashable, list[str]] = {}
    for source_pmcid in required_pmcs:
        refs = source_refs.loc[source_refs["source_pmcid"].eq(source_pmcid), "pmid"].tolist()
        source_ref_lookup[source_pmcid] = list(dict.fromkeys(refs))
    return source_ref_lookup


def _build_query_decoy_options(
    filtered_df: pd.DataFrame,
    df_sources: pd.DataFrame,
    n_required: int,
) -> dict[Hashable, list[str]]:
    """Build a mapping from query ID to list of decoy reference PMIDs that could be selected for that query."""
    if "hits_parsed" not in filtered_df.columns:
        raise ValueError("filtered_df must contain hits_parsed before decoy selection")

    source_ref_lookup = _build_source_reference_lookup(filtered_df, df_sources)
    query_decoy_options: dict[Hashable, list[str]] = {}

    for query_id, row in filtered_df.iterrows():
        source_pmcid = row["source_pmcid"]
        hits = {hit for hit in row["hits_parsed"] if hit is not None}
        decoys = [ref for ref in source_ref_lookup.get(source_pmcid, []) if ref not in hits]
        if len(decoys) < n_required:
            raise ValueError(
                f"Query {query_id} from source {source_pmcid} has only {len(decoys)} decoy references but needs {n_required}"
            )
        query_decoy_options[query_id] = decoys

    return query_decoy_options


def _select_shared_references_greedy(
    query_hits: dict[Hashable, list[str]],
    n_required: int,
) -> list[str]:
    """Select a set of reference PMIDs that covers at least n_required decoys for each query, reusing references globally when possible."""
    ref_to_queries: dict[str, set[Hashable]] = {}
    for query_id, hits in query_hits.items():
        for hit in hits:
            ref_to_queries.setdefault(hit, set()).add(query_id)

    remaining = {query_id: n_required for query_id in query_hits}
    selected_refs: list[str] = []
    selected_ref_set: set[str] = set()
    heap: list[tuple[int, int, str]] = [
        (-len(query_ids), -len(query_ids), ref_id)
        for ref_id, query_ids in ref_to_queries.items()
    ]
    heapq.heapify(heap)

    while any(value > 0 for value in remaining.values()):
        if not heap:
            raise RuntimeError("No references left that can satisfy the remaining query constraints")

        _, _, ref_id = heapq.heappop(heap)
        if ref_id in selected_ref_set:
            continue

        current_cover = sum(1 for query_id in ref_to_queries[ref_id] if remaining[query_id] > 0)
        if current_cover == 0:
            continue

        if heap and -heap[0][0] > current_cover:
            heapq.heappush(heap, (-current_cover, -len(ref_to_queries[ref_id]), ref_id))
            continue

        selected_refs.append(ref_id)
        selected_ref_set.add(ref_id)
        for query_id in ref_to_queries[ref_id]:
            if remaining[query_id] > 0:
                remaining[query_id] -= 1

    return selected_refs


def _prune_redundant_selected_references(
    query_hits: dict[Hashable, list[str]],
    selected_refs: list[str],
    n_required: int,
) -> set[str]:
    """Prune references from the selected set that are redundant, i.e. whose removal would still leave at least n_required decoys for every query."""
    ref_to_queries: dict[str, set[Hashable]] = {}
    for query_id, hits in query_hits.items():
        for hit in hits:
            ref_to_queries.setdefault(hit, set()).add(query_id)

    selected_ref_set = set(selected_refs)
    support_count = {
        query_id: sum(1 for hit in hits if hit in selected_ref_set)
        for query_id, hits in query_hits.items()
    }

    for ref_id in reversed(selected_refs):
        query_ids = ref_to_queries.get(ref_id, set())
        if all(support_count[query_id] - 1 >= n_required for query_id in query_ids):
            selected_ref_set.remove(ref_id)
            for query_id in query_ids:
                support_count[query_id] -= 1

    return selected_ref_set
