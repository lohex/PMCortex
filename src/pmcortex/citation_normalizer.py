"""Normalize bibliographic JATS cross-references without rewriting prose."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import re

from lxml import etree


_MARKER_PATTERN = re.compile(r"\[xref:([^\]\s]+)\]")
_MARKER_TEXT = r"\[xref:[^\]\s]+\]"
_MARKER_CLUSTER_TEXT = (
    rf"{_MARKER_TEXT}"
    rf"(?:\s*(?:(?:[,;/&–—−-]|\band\b|\bto\b)\s*)?{_MARKER_TEXT})*"
)
_MARKER_CLUSTER_PATTERN = re.compile(
    _MARKER_CLUSTER_TEXT,
    flags=re.IGNORECASE,
)
_WRAPPED_MARKER_CLUSTER_PATTERN = re.compile(
    rf"(?:\(\s*{_MARKER_CLUSTER_TEXT}\s*\)|\[\s*{_MARKER_CLUSTER_TEXT}\s*\])",
    flags=re.IGNORECASE,
)
_LABEL_ATOM_PATTERN = re.compile(r"^(\d{1,4})([A-Za-z]?)$")
_LABEL_SEPARATOR_PATTERN = re.compile(
    r"\s*(,|;|/|&|\band\b|\bto\b|[-–—−])\s*",
    flags=re.IGNORECASE,
)
_RANGE_SEPARATOR_PATTERN = re.compile(
    r"^\s*(?:[-–—−]|to)\s*$",
    flags=re.IGNORECASE,
)
_LIST_SEPARATOR_PATTERN = re.compile(
    r"^\s*(?:(?:[,;/]|&)|(?:,\s*)?\band\b)?\s*$",
    flags=re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(?:18|19|20)\d{2}[A-Za-z]?\b")
_TRAILING_LABEL_PUNCTUATION_PATTERN = re.compile(r"([.,;:!?]+)\s*$")
_BLOCK_BOUNDARY_TAGS = frozenset({"p", "title"})


class CitationDiagnosticCode(StrEnum):
    """Closed set of recoverable citation-normalization problems."""

    MISSING_RID = "missing_rid"
    UNKNOWN_RID = "unknown_rid"
    INVALID_RANGE = "invalid_range"
    UNSAFE_SEMANTIC_CITATION = "unsafe_semantic_citation"


class CitationForm(StrEnum):
    """Closed set of structurally observed bibliographic citation forms."""

    LABEL_ONLY = "label_only"
    PARENTHETICAL_AUTHOR_YEAR = "parenthetical_author_year"
    NARRATIVE_AUTHOR_YEAR = "narrative_author_year"
    MIXED_OR_MALFORMED = "mixed_or_malformed"


@dataclass(frozen=True, slots=True)
class CitationDiagnostic:
    """One recoverable problem encountered while normalizing a citation."""

    code: CitationDiagnosticCode
    rids: tuple[str, ...]
    source_line: int | None


@dataclass(frozen=True, slots=True)
class TextToken:
    """Literal mixed-content text that must retain its relative order."""

    text: str


@dataclass(frozen=True, slots=True)
class CitationToken:
    """Atomic bibliographic xref with its visible label and target IDs."""

    rids: tuple[str, ...]
    display_text: str
    source_line: int | None


InlineToken = TextToken | CitationToken


@dataclass(frozen=True, slots=True)
class CitationOccurrence:
    """One classified bibliographic xref occurrence in normalized source text."""

    form: CitationForm
    display_text: str
    rids: tuple[str, ...]
    source_line: int | None


@dataclass(frozen=True, slots=True)
class NormalizedCitationText:
    """Full-text and query renderings produced from one mixed-content block."""

    text_with_markers: str
    query_text_with_markers: str
    query_text: str
    cited_rids: tuple[str, ...]
    citation_occurrences: tuple[CitationOccurrence, ...]
    diagnostics: tuple[CitationDiagnostic, ...]

    @property
    def unsafe_citation_rids(self) -> tuple[str, ...]:
        """Return stable IDs belonging to narrative or malformed citations."""
        unsafe_forms = {
            CitationForm.NARRATIVE_AUTHOR_YEAR,
            CitationForm.MIXED_OR_MALFORMED,
        }
        return tuple(
            dict.fromkeys(
                rid
                for occurrence in self.citation_occurrences
                if occurrence.form in unsafe_forms
                for rid in occurrence.rids
            )
        )


@dataclass(frozen=True, slots=True)
class _RenderedCitationRun:
    """Rendered form and metadata for one adjacent citation-token run."""

    full_text: str
    query_text_with_markers: str
    cited_rids: tuple[str, ...]
    occurrences: tuple[CitationOccurrence, ...]
    diagnostics: tuple[CitationDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class _ResolvedCitationRun:
    """Resolved citations, connectors, and diagnostics for one token run."""

    citations: tuple[CitationToken, ...]
    connectors: tuple[str, ...]
    rids_by_citation: tuple[tuple[str, ...], ...]
    label_only_flags: tuple[bool, ...]
    cited_rids: tuple[str, ...]
    diagnostics: tuple[CitationDiagnostic, ...]


class CitationNormalizer:
    """Normalize JATS bibliographic xrefs using XML structure and ref labels."""

    def __init__(
        self,
        reference_order: Sequence[str],
        reference_labels: Mapping[str, str | None],
        *,
        skipped_tags: Iterable[str] = (),
    ) -> None:
        """
        Configure citation resolution for one article.

        Args:
            reference_order: Bibliographic IDs in their JATS document order.
            reference_labels: Visible reference-list labels keyed by ID.
            skipped_tags: Inline element names whose complete content is omitted.

        Raises:
            TypeError: If an argument or contained value has the wrong type.
            ValueError: If a reference ID or skipped tag is empty.
        """
        if isinstance(reference_order, (str, bytes)):
            raise TypeError("reference_order must be a sequence of strings")
        if not isinstance(reference_labels, Mapping):
            raise TypeError("reference_labels must be a mapping")
        if isinstance(skipped_tags, (str, bytes)):
            raise TypeError("skipped_tags must be an iterable of strings")

        order = tuple(reference_order)
        for rid in order:
            if not isinstance(rid, str):
                raise TypeError("reference IDs must be strings")
            if not rid.strip():
                raise ValueError("reference IDs must not be empty")

        self._reference_order = self._stable_unique(order)
        self._reference_positions = {
            rid: index for index, rid in enumerate(self._reference_order)
        }

        skipped_tag_values = tuple(skipped_tags)
        for tag in skipped_tag_values:
            if not isinstance(tag, str):
                raise TypeError("skipped tags must be strings")
            if not tag.strip():
                raise ValueError("skipped tags must not be empty")
        self._skipped_tags = frozenset(skipped_tag_values)

        rid_to_label: dict[str, str] = {}
        label_candidates: defaultdict[str, list[str]] = defaultdict(list)
        for rid, raw_label in reference_labels.items():
            if not isinstance(rid, str):
                raise TypeError("reference-label IDs must be strings")
            if not rid.strip():
                raise ValueError("reference-label IDs must not be empty")
            if raw_label is None:
                continue
            if not isinstance(raw_label, str):
                raise TypeError("reference labels must be strings or None")
            label_key = self._normalize_reference_label(raw_label)
            if label_key is None:
                continue
            rid_to_label[rid] = label_key
            label_candidates[label_key].append(rid)

        self._known_rids = frozenset((*self._reference_order, *reference_labels))
        self._rid_to_label = rid_to_label
        self._label_to_rid = {
            label: candidates[0]
            for label, candidates in label_candidates.items()
            if len(candidates) == 1
        }
        self._ambiguous_labels = frozenset(
            label
            for label, candidates in label_candidates.items()
            if len(candidates) > 1
        )

    def tokenize(self, element: etree._Element) -> tuple[InlineToken, ...]:
        """
        Convert one element's mixed content into ordered text and citation tokens.

        The element's own tail is deliberately excluded; callers that traverse
        siblings are responsible for adding it exactly once.

        Args:
            element: JATS element whose content should be tokenized.

        Returns:
            Immutable ordered inline-token sequence.

        Raises:
            TypeError: If ``element`` is not an lxml element.
        """
        if not isinstance(element, etree._Element):
            raise TypeError("element must be an lxml element")

        tokens: list[InlineToken] = []
        self._tokenize_element_inplace(element, tokens)
        return tuple(tokens)

    def normalize(self, tokens: Sequence[InlineToken]) -> NormalizedCitationText:
        """
        Render a token sequence as marker-bearing full text and a clean query.

        The source-like rendering retains semantic display text for diagnostics.
        The query rendering removes all bibliographic xref display text while
        retaining markers long enough to associate each sentence with its cited
        documents. Unsafe narrative citations are classified for downstream
        rejection instead of being silently rewritten.

        Args:
            tokens: Ordered output from :meth:`tokenize` or equivalent tokens.

        Returns:
            Both normalized renderings, cited IDs, and recoverable diagnostics.

        Raises:
            TypeError: If ``tokens`` is not a sequence or contains an invalid item.
        """
        if not isinstance(tokens, Sequence):
            raise TypeError("tokens must be a sequence")
        for token in tokens:
            if not isinstance(token, (TextToken, CitationToken)):
                raise TypeError("tokens must contain TextToken or CitationToken values")

        repaired_tokens = self._repair_split_delimiters(tokens)
        wrapper_normalized_tokens = self._strip_label_only_wrappers(repaired_tokens)
        full_parts: list[str] = []
        query_parts: list[str] = []
        cited_rids: list[str] = []
        occurrences: list[CitationOccurrence] = []
        diagnostics: list[CitationDiagnostic] = []

        index = 0
        while index < len(wrapper_normalized_tokens):
            token = wrapper_normalized_tokens[index]
            if isinstance(token, TextToken):
                full_parts.append(token.text)
                query_parts.append(token.text)
                index += 1
                continue

            run_end = self._citation_run_end(wrapper_normalized_tokens, index)
            is_parenthetically_wrapped = self._run_has_external_wrapper(
                wrapper_normalized_tokens,
                index,
                run_end,
            )
            rendered_run = self._render_citation_run(
                wrapper_normalized_tokens[index : run_end + 1],
                is_parenthetically_wrapped=is_parenthetically_wrapped,
            )
            full_parts.append(rendered_run.full_text)
            query_parts.append(rendered_run.query_text_with_markers)
            cited_rids.extend(rendered_run.cited_rids)
            occurrences.extend(rendered_run.occurrences)
            diagnostics.extend(rendered_run.diagnostics)
            index = run_end + 1

        full_text = self._normalize_text("".join(full_parts))
        query_text_with_markers = self._normalize_text("".join(query_parts))
        query_text = self.remove_markers(query_text_with_markers)

        return NormalizedCitationText(
            text_with_markers=full_text,
            query_text_with_markers=query_text_with_markers,
            query_text=query_text,
            cited_rids=self._stable_unique(cited_rids),
            citation_occurrences=tuple(occurrences),
            diagnostics=tuple(diagnostics),
        )

    def normalize_fragments(
        self,
        fragments: Iterable[str | etree._Element],
    ) -> NormalizedCitationText:
        """
        Tokenize and normalize an ordered sequence of text and XML fragments.

        XML fragment tails are excluded so the caller can pass them separately
        without duplicating text.

        Args:
            fragments: Literal strings or JATS inline elements in source order.

        Returns:
            Normalized text and citation metadata for the combined fragments.

        Raises:
            TypeError: If a fragment is neither text nor an lxml element.
        """
        tokens: list[InlineToken] = []
        for fragment in fragments:
            if isinstance(fragment, str):
                self._append_text_inplace(tokens, fragment)
            elif isinstance(fragment, etree._Element):
                self._tokenize_element_inplace(fragment, tokens)
            else:
                raise TypeError("fragments must contain strings or lxml elements")
        return self.normalize(tokens)

    @classmethod
    def remove_markers(cls, text: str) -> str:
        """
        Remove citation markers without consuming following grammatical punctuation.

        Separators are removed only when they occur between two markers. A comma
        following the final marker therefore remains part of the surrounding prose.

        Args:
            text: Marker-bearing normalized text.

        Returns:
            Marker-free text with normalized local whitespace and punctuation.

        Raises:
            TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        without_wrapped_clusters = _WRAPPED_MARKER_CLUSTER_PATTERN.sub(" ", text)
        without_markers = _MARKER_CLUSTER_PATTERN.sub(" ", without_wrapped_clusters)
        return cls._normalize_text(without_markers)

    @staticmethod
    def extract_rids(text: str) -> tuple[str, ...]:
        """
        Return marker IDs in first-occurrence order.

        Args:
            text: Marker-bearing normalized text.

        Returns:
            Stable-deduplicated citation IDs.

        Raises:
            TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return CitationNormalizer._stable_unique(_MARKER_PATTERN.findall(text))

    def _tokenize_element_inplace(
        self,
        element: etree._Element,
        tokens: list[InlineToken],
    ) -> None:
        """Append an element's content to ``tokens`` without appending its own tail."""
        element_name = etree.QName(element).localname
        if element_name in self._skipped_tags:
            return

        is_bibliographic_xref = (
            element_name == "xref" and element.get("ref-type") == "bibr"
        )
        if is_bibliographic_xref:
            rid_value = element.get("rid")
            rids = tuple(rid_value.split()) if rid_value else ()
            display_text = "".join(element.itertext())
            tokens.append(
                CitationToken(
                    rids=rids,
                    display_text=display_text,
                    source_line=element.sourceline,
                )
            )
            return

        self._append_text_inplace(tokens, element.text)
        for child in element:
            self._tokenize_element_inplace(child, tokens)
            child_name = etree.QName(child).localname
            if child_name in _BLOCK_BOUNDARY_TAGS:
                self._append_text_inplace(tokens, " ")
            self._append_text_inplace(tokens, child.tail)

    @staticmethod
    def _append_text_inplace(
        tokens: list[InlineToken],
        text: str | None,
    ) -> None:
        """Append literal text while coalescing adjacent text tokens in place."""
        if not text:
            return
        if tokens and isinstance(tokens[-1], TextToken):
            tokens[-1] = TextToken(tokens[-1].text + text)
            return
        tokens.append(TextToken(text))

    def _repair_split_delimiters(
        self,
        tokens: Sequence[InlineToken],
    ) -> tuple[InlineToken, ...]:
        """Move a citation-year tail through its matching closing delimiter."""
        repaired = list(tokens)
        for index in range(len(repaired) - 1):
            citation = repaired[index]
            following = repaired[index + 1]
            if not isinstance(citation, CitationToken):
                continue
            if not isinstance(following, TextToken):
                continue

            unmatched = self._last_unmatched_opener(citation.display_text)
            if unmatched is None:
                continue
            opener, opener_index = unmatched
            close_index = self._matching_close_index(following.text, opener)
            if close_index is None:
                continue

            moved_prefix = following.text[: close_index + 1]
            enclosed_text = (
                citation.display_text[opener_index + 1 :] + moved_prefix[:-1]
            )
            is_author_year = opener == "(" and _YEAR_PATTERN.search(enclosed_text)
            is_numeric_group = (
                opener in "([" and self._parse_label_expression(enclosed_text)
            )
            if not is_author_year and not is_numeric_group:
                continue

            repaired[index] = CitationToken(
                rids=citation.rids,
                display_text=citation.display_text + moved_prefix,
                source_line=citation.source_line,
            )
            repaired[index + 1] = TextToken(following.text[close_index + 1 :])

        return tuple(repaired)

    @staticmethod
    def _last_unmatched_opener(text: str) -> tuple[str, int] | None:
        """Return the final unmatched round or square opener and its position."""
        matching_open = {")": "(", "]": "["}
        stack: list[tuple[str, int]] = []
        for index, character in enumerate(text):
            if character in "([":
                stack.append((character, index))
            elif (
                character in matching_open
                and stack
                and stack[-1][0] == matching_open[character]
            ):
                stack.pop()
        return stack[-1] if stack else None

    @staticmethod
    def _matching_close_index(text: str, opener: str) -> int | None:
        """Find the matching delimiter in following text, respecting nesting."""
        closer = ")" if opener == "(" else "]"
        depth = 1
        for index, character in enumerate(text):
            if character == opener:
                depth += 1
            elif character == closer:
                depth -= 1
                if depth == 0:
                    return index
        return None

    def _strip_label_only_wrappers(
        self,
        tokens: Sequence[InlineToken],
    ) -> tuple[InlineToken, ...]:
        """Remove outer brackets only when they enclose a pure label-only run."""
        normalized = list(tokens)
        index = 0
        while index < len(normalized):
            if not isinstance(normalized[index], CitationToken):
                index += 1
                continue

            run_end = self._citation_run_end(normalized, index)
            citation_tokens = [
                token
                for token in normalized[index : run_end + 1]
                if isinstance(token, CitationToken)
            ]
            is_label_only_run = all(
                self._is_label_only(token) for token in citation_tokens
            )
            has_neighbors = index > 0 and run_end + 1 < len(normalized)
            if not is_label_only_run or not has_neighbors:
                index = run_end + 1
                continue

            previous = normalized[index - 1]
            following = normalized[run_end + 1]
            if not isinstance(previous, TextToken):
                index = run_end + 1
                continue
            if not isinstance(following, TextToken):
                index = run_end + 1
                continue

            opener_match = re.search(r"([\(\[])\s*$", previous.text)
            closer_match = re.match(r"^\s*([\)\]])", following.text)
            delimiters_match = (
                opener_match is not None
                and closer_match is not None
                and {opener_match.group(1), closer_match.group(1)}
                in ({"(", ")"}, {"[", "]"})
            )
            if (
                delimiters_match
                and opener_match is not None
                and closer_match is not None
            ):
                normalized[index - 1] = TextToken(
                    previous.text[: opener_match.start()]
                )
                normalized[run_end + 1] = TextToken(
                    following.text[closer_match.end() :]
                )

            index = run_end + 1

        return tuple(normalized)

    @staticmethod
    def _run_has_external_wrapper(
        tokens: Sequence[InlineToken],
        start: int,
        end: int,
    ) -> bool:
        """Return whether neighboring prose wraps one citation run in delimiters."""
        has_neighbors = start > 0 and end + 1 < len(tokens)
        if not has_neighbors:
            return False

        previous = tokens[start - 1]
        following = tokens[end + 1]
        if not isinstance(previous, TextToken) or not isinstance(following, TextToken):
            return False

        opener_match = re.search(r"([\(\[])\s*$", previous.text)
        closer_match = re.match(r"^\s*([\)\]])", following.text)
        if opener_match is None or closer_match is None:
            return False
        return {opener_match.group(1), closer_match.group(1)} in (
            {"(", ")"},
            {"[", "]"},
        )

    @staticmethod
    def _citation_run_end(tokens: Sequence[InlineToken], start: int) -> int:
        """Return the inclusive end index of an adjacent citation run."""
        end = start
        while end + 2 < len(tokens):
            current = tokens[end]
            connector = tokens[end + 1]
            following = tokens[end + 2]
            ends_sentence = (
                isinstance(current, CitationToken)
                and re.search(r"[.!?]\s*$", current.display_text) is not None
            )
            if ends_sentence:
                break
            if not isinstance(connector, TextToken):
                break
            if not isinstance(following, CitationToken):
                break
            is_connector = bool(
                _LIST_SEPARATOR_PATTERN.fullmatch(connector.text)
                or _RANGE_SEPARATOR_PATTERN.fullmatch(connector.text)
            )
            if not is_connector:
                break
            end += 2
        return end

    def _render_citation_run(
        self,
        run_tokens: Sequence[InlineToken],
        *,
        is_parenthetically_wrapped: bool,
    ) -> _RenderedCitationRun:
        """Resolve and render an alternating citation/connector token run."""
        resolved_run = self._resolve_citation_run(run_tokens)
        is_label_only_run = bool(resolved_run.citations) and all(
            resolved_run.label_only_flags
        )
        if is_label_only_run:
            return self._render_label_only_run(resolved_run)
        return self._render_mixed_run(
            resolved_run,
            is_parenthetically_wrapped=is_parenthetically_wrapped,
        )

    def _resolve_citation_run(
        self,
        run_tokens: Sequence[InlineToken],
    ) -> _ResolvedCitationRun:
        """Resolve explicit targets and numeric ranges without rendering text."""
        citations = tuple(
            token for token in run_tokens if isinstance(token, CitationToken)
        )
        connectors = tuple(
            token.text for token in run_tokens if isinstance(token, TextToken)
        )
        rids_by_citation = tuple(
            self._resolve_token_rids(citation) for citation in citations
        )
        label_only_flags = tuple(
            self._is_label_only(citation) for citation in citations
        )
        diagnostics = [
            diagnostic
            for citation in citations
            for diagnostic in self._diagnostics_for_token(citation)
        ]

        cited_rids: list[str] = []
        if citations:
            cited_rids.extend(rids_by_citation[0])
        for index, connector in enumerate(connectors):
            next_rids = rids_by_citation[index + 1]
            is_range = bool(_RANGE_SEPARATOR_PATTERN.fullmatch(connector))
            has_numeric_endpoints = (
                label_only_flags[index] and label_only_flags[index + 1]
            )
            if is_range and has_numeric_endpoints and cited_rids and next_rids:
                start_labels = self._label_keys_for_display(
                    citations[index].display_text,
                    citations[index].rids,
                )
                end_labels = self._label_keys_for_display(
                    citations[index + 1].display_text,
                    citations[index + 1].rids,
                )
                start_label_hint = start_labels[-1] if start_labels else None
                end_label_hint = end_labels[0] if end_labels else None
                expanded, range_diagnostic = self._expand_range(
                    cited_rids[-1],
                    next_rids[0],
                    citations[index].source_line,
                    start_label_hint=start_label_hint,
                    end_label_hint=end_label_hint,
                )
                cited_rids.extend(expanded)
                if range_diagnostic is not None:
                    diagnostics.append(range_diagnostic)
            cited_rids.extend(next_rids)

        return _ResolvedCitationRun(
            citations=citations,
            connectors=connectors,
            rids_by_citation=rids_by_citation,
            label_only_flags=label_only_flags,
            cited_rids=self._stable_unique(cited_rids),
            diagnostics=tuple(diagnostics),
        )

    def _render_label_only_run(
        self,
        run: _ResolvedCitationRun,
    ) -> _RenderedCitationRun:
        """Render a numeric citation run while preserving terminal punctuation."""
        trailing_punctuation = self._label_trailing_punctuation(
            run.citations[-1].display_text
        )
        has_invalid_range = any(
            diagnostic.code == CitationDiagnosticCode.INVALID_RANGE
            for diagnostic in run.diagnostics
        )
        if has_invalid_range:
            full_parts: list[str] = []
            for index, token_rids in enumerate(run.rids_by_citation):
                punctuation = self._label_trailing_punctuation(
                    run.citations[index].display_text
                )
                full_parts.append(
                    f" {self._render_markers(token_rids)}{punctuation} "
                )
                if index < len(run.connectors):
                    full_parts.append(run.connectors[index])
            full_text = "".join(full_parts)
        else:
            marker_text = self._render_markers(run.cited_rids)
            full_text = (
                f" {marker_text}{trailing_punctuation} "
                if marker_text
                else f" {trailing_punctuation} "
            )

        return _RenderedCitationRun(
            full_text=full_text,
            query_text_with_markers=full_text,
            cited_rids=run.cited_rids,
            occurrences=tuple(
                CitationOccurrence(
                    form=CitationForm.LABEL_ONLY,
                    display_text=citation.display_text,
                    rids=token_rids,
                    source_line=citation.source_line,
                )
                for citation, token_rids in zip(
                    run.citations,
                    run.rids_by_citation,
                )
            ),
            diagnostics=run.diagnostics,
        )

    def _render_mixed_run(
        self,
        run: _ResolvedCitationRun,
        *,
        is_parenthetically_wrapped: bool,
    ) -> _RenderedCitationRun:
        """Render semantic citation text and any numeric members in source order."""
        full_parts: list[str] = []
        query_parts: list[str] = []
        occurrences: list[CitationOccurrence] = []
        diagnostics = list(run.diagnostics)
        for index, citation in enumerate(run.citations):
            display_text = citation.display_text
            token_rids = run.rids_by_citation[index]
            marker_text = self._render_markers(token_rids)
            if run.label_only_flags[index]:
                citation_form = CitationForm.LABEL_ONLY
                punctuation = self._label_trailing_punctuation(display_text)
                rendered_citation = (
                    f" {marker_text}{punctuation} "
                    if marker_text
                    else f" {punctuation} "
                )
                query_citation = rendered_citation
            else:
                citation_form = self._classify_semantic_citation(
                    citation,
                    is_parenthetically_wrapped=is_parenthetically_wrapped,
                )
                rendered_citation = f" {display_text} "
                if marker_text:
                    rendered_citation = f" {display_text} {marker_text} "
                punctuation = self._semantic_trailing_punctuation(display_text)
                query_citation = (
                    f" {marker_text}{punctuation} "
                    if marker_text
                    else f" {punctuation} "
                )
                is_unsafe = citation_form in {
                    CitationForm.NARRATIVE_AUTHOR_YEAR,
                    CitationForm.MIXED_OR_MALFORMED,
                }
                if is_unsafe:
                    diagnostics.append(
                        CitationDiagnostic(
                            code=CitationDiagnosticCode.UNSAFE_SEMANTIC_CITATION,
                            rids=token_rids,
                            source_line=citation.source_line,
                        )
                    )
            full_parts.append(rendered_citation)
            query_parts.append(query_citation)
            occurrences.append(
                CitationOccurrence(
                    form=citation_form,
                    display_text=display_text,
                    rids=token_rids,
                    source_line=citation.source_line,
                )
            )
            if index < len(run.connectors):
                full_parts.append(run.connectors[index])
                query_parts.append(run.connectors[index])

        return _RenderedCitationRun(
            full_text="".join(full_parts),
            query_text_with_markers="".join(query_parts),
            cited_rids=run.cited_rids,
            occurrences=tuple(occurrences),
            diagnostics=tuple(diagnostics),
        )

    def _classify_semantic_citation(
        self,
        citation: CitationToken,
        *,
        is_parenthetically_wrapped: bool,
    ) -> CitationForm:
        """Classify semantic xref text without inferring from bibliography metadata."""
        display_text = citation.display_text.strip()
        core_text = _TRAILING_LABEL_PUNCTUATION_PATTERN.sub(
            "",
            display_text,
        ).strip()
        has_year = _YEAR_PATTERN.search(core_text) is not None
        has_letters = any(character.isalpha() for character in core_text)
        is_bare_year = _YEAR_PATTERN.fullmatch(core_text) is not None
        has_own_wrapper = (
            len(core_text) >= 2
            and (core_text[0], core_text[-1]) in {("(", ")"), ("[", "]")}
        )
        is_safe_parenthetical = (
            has_year
            and has_letters
            and not is_bare_year
            and (has_own_wrapper or is_parenthetically_wrapped)
        )
        if is_safe_parenthetical:
            return CitationForm.PARENTHETICAL_AUTHOR_YEAR
        if has_year:
            return CitationForm.NARRATIVE_AUTHOR_YEAR
        return CitationForm.MIXED_OR_MALFORMED

    @staticmethod
    def _semantic_trailing_punctuation(display_text: str) -> str:
        """Preserve sentence punctuation stored after a semantic citation wrapper."""
        stripped = display_text.rstrip()
        match = re.search(r"([.!?]+)$", stripped)
        return match.group(1) if match is not None else ""

    def _resolve_token_rids(self, token: CitationToken) -> tuple[str, ...]:
        """Resolve explicit IDs plus numeric lists or ranges in one xref display."""
        label_keys = self._label_keys_for_display(token.display_text, token.rids)
        if label_keys is None:
            return self._stable_unique(token.rids)

        if len(label_keys) == 1:
            return self._stable_unique(token.rids)

        mapped_rids = tuple(
            self._label_to_rid[label]
            for label in label_keys
            if label in self._label_to_rid
        )
        if len(mapped_rids) == len(label_keys):
            has_explicit_anchor = any(rid in mapped_rids for rid in token.rids)
            if has_explicit_anchor:
                return self._stable_unique((*mapped_rids, *token.rids))
            return self._stable_unique(token.rids)

        fallback_rids = self._resolve_labels_from_order(label_keys, token.rids)
        if fallback_rids is not None:
            return self._stable_unique((*fallback_rids, *token.rids))
        return self._stable_unique(token.rids)

    def _resolve_labels_from_order(
        self,
        label_keys: Sequence[str],
        explicit_rids: Sequence[str],
    ) -> tuple[str, ...] | None:
        """Resolve numeric labels by an explicit RID anchor when labels are absent."""
        if not label_keys or not explicit_rids:
            return None
        if any(label in self._ambiguous_labels for label in label_keys):
            return None

        anchor_position = self._reference_positions.get(explicit_rids[0])
        anchor_parts = self._label_number_and_suffix(label_keys[0])
        if anchor_position is None or anchor_parts is None or anchor_parts[1]:
            return None

        anchor_number = anchor_parts[0]
        positions: list[int] = []
        for label in label_keys:
            parts = self._label_number_and_suffix(label)
            if parts is None or parts[1]:
                return None
            position = anchor_position + parts[0] - anchor_number
            if not 0 <= position < len(self._reference_order):
                return None
            known_rid = self._label_to_rid.get(label)
            if known_rid is not None and known_rid != self._reference_order[position]:
                return None
            positions.append(position)
        return tuple(self._reference_order[position] for position in positions)

    def _expand_range(
        self,
        start_rid: str,
        end_rid: str,
        source_line: int | None,
        *,
        start_label_hint: str | None = None,
        end_label_hint: str | None = None,
    ) -> tuple[tuple[str, ...], CitationDiagnostic | None]:
        """Expand one range by unique labels, then by validated document order."""
        if start_rid == end_rid:
            return (start_rid,), None

        start_label = self._rid_to_label.get(start_rid)
        if start_label is None:
            start_label = start_label_hint
        end_label = self._rid_to_label.get(end_rid)
        if end_label is None:
            end_label = end_label_hint
        if start_label is not None and end_label is not None:
            label_range = self._expand_label_range(start_label, end_label)
            if label_range is not None:
                mapped = tuple(
                    self._label_to_rid[label]
                    for label in label_range
                    if label in self._label_to_rid
                )
                if len(mapped) == len(label_range):
                    return mapped, None

        start_position = self._reference_positions.get(start_rid)
        end_position = self._reference_positions.get(end_rid)
        is_forward_range = (
            start_position is not None
            and end_position is not None
            and start_position <= end_position
        )
        if is_forward_range and start_position is not None and end_position is not None:
            ordered_slice = self._reference_order[start_position : end_position + 1]
            if self._ordered_slice_matches_labels(
                ordered_slice,
                start_label,
                end_label,
            ):
                return ordered_slice, None

        diagnostic = CitationDiagnostic(
            code=CitationDiagnosticCode.INVALID_RANGE,
            rids=(start_rid, end_rid),
            source_line=source_line,
        )
        return (start_rid, end_rid), diagnostic

    def _ordered_slice_matches_labels(
        self,
        ordered_slice: Sequence[str],
        start_label: str | None,
        end_label: str | None,
    ) -> bool:
        """Reject a fallback slice whose numeric labels are out of range or order."""
        if start_label is None or end_label is None:
            return True
        start_parts = self._label_number_and_suffix(start_label)
        end_parts = self._label_number_and_suffix(end_label)
        if start_parts is None or end_parts is None:
            return True
        if start_parts[1] or end_parts[1] or start_parts[0] > end_parts[0]:
            return False

        observed_numbers = [
            parts[0]
            for rid in ordered_slice
            if (parts := self._label_number_and_suffix(self._rid_to_label.get(rid)))
            is not None
            and not parts[1]
        ]
        within_bounds = all(
            start_parts[0] <= number <= end_parts[0]
            for number in observed_numbers
        )
        return within_bounds and observed_numbers == sorted(observed_numbers)

    def _is_label_only(self, token: CitationToken) -> bool:
        """Return whether an xref display consists only of removable labels."""
        if not token.rids:
            return False
        if not token.display_text.strip():
            return True
        return self._label_keys_for_display(token.display_text, token.rids) is not None

    def _label_keys_for_display(
        self,
        display_text: str,
        explicit_rids: Sequence[str],
    ) -> tuple[str, ...] | None:
        """Parse label-like display text without mistaking bare years for labels."""
        label_keys = self._parse_label_expression(display_text)
        if label_keys is None:
            return None
        if all(label in self._label_to_rid for label in label_keys):
            return label_keys

        parsed_parts = [self._label_number_and_suffix(label) for label in label_keys]
        is_short_numeric_expression = all(
            parts is not None and parts[0] <= 999
            for parts in parsed_parts
        )
        if explicit_rids and is_short_numeric_expression:
            return label_keys
        return None

    @classmethod
    def _parse_label_expression(cls, display_text: str) -> tuple[str, ...] | None:
        """Parse a numeric citation label, list, or ascending range."""
        core = display_text.strip()
        core = _TRAILING_LABEL_PUNCTUATION_PATTERN.sub("", core).strip()
        while len(core) >= 2 and (core[0], core[-1]) in {("[", "]"), ("(", ")")}:
            core = core[1:-1].strip()
        if not core:
            return None

        parts = _LABEL_SEPARATOR_PATTERN.split(core)
        if not parts or len(parts) % 2 == 0:
            return None
        first_key = cls._normalize_label_atom(parts[0])
        if first_key is None:
            return None

        label_keys = [first_key]
        previous_key = first_key
        for index in range(1, len(parts), 2):
            separator = parts[index]
            next_key = cls._normalize_label_atom(parts[index + 1])
            if next_key is None:
                return None
            if _RANGE_SEPARATOR_PATTERN.fullmatch(separator):
                expanded = cls._expand_label_range(previous_key, next_key)
                if expanded is None:
                    return None
                label_keys.extend(expanded[1:])
            else:
                label_keys.append(next_key)
            previous_key = next_key
        return cls._stable_unique(label_keys)

    @classmethod
    def _expand_label_range(
        cls,
        start_label: str,
        end_label: str,
    ) -> tuple[str, ...] | None:
        """Expand a bounded ascending range of plain numeric reference labels."""
        start_parts = cls._label_number_and_suffix(start_label)
        end_parts = cls._label_number_and_suffix(end_label)
        if start_parts is None or end_parts is None:
            return None
        start_number, start_suffix = start_parts
        end_number, end_suffix = end_parts
        is_plain_ascending_range = (
            not start_suffix
            and not end_suffix
            and start_number <= end_number
            and end_number - start_number <= 500
        )
        if not is_plain_ascending_range:
            return None
        return tuple(str(number) for number in range(start_number, end_number + 1))

    @staticmethod
    def _normalize_label_atom(raw_label: str) -> str | None:
        """Normalize one reference label to an integer plus optional suffix."""
        label = raw_label.strip()
        if len(label) >= 2 and (label[0], label[-1]) in {("[", "]"), ("(", ")")}:
            label = label[1:-1].strip()
        match = _LABEL_ATOM_PATTERN.fullmatch(label)
        if match is None:
            return None
        number, suffix = match.groups()
        return f"{int(number)}{suffix.casefold()}"

    @classmethod
    def _normalize_reference_label(cls, raw_label: str) -> str | None:
        """Normalize a ref-list label whose terminal period is non-semantic."""
        label_without_period = re.sub(r"\s*\.\s*$", "", raw_label).strip()
        return cls._normalize_label_atom(label_without_period)

    @staticmethod
    def _label_trailing_punctuation(display_text: str) -> str:
        """Return sentence or clause punctuation stored inside a numeric xref."""
        match = _TRAILING_LABEL_PUNCTUATION_PATTERN.search(display_text)
        return match.group(1) if match is not None else ""

    @staticmethod
    def _label_number_and_suffix(label: str | None) -> tuple[int, str] | None:
        """Split a normalized label key into its numeric and suffix components."""
        if label is None:
            return None
        match = _LABEL_ATOM_PATTERN.fullmatch(label)
        if match is None:
            return None
        number, suffix = match.groups()
        return int(number), suffix.casefold()

    def _diagnostics_for_token(
        self,
        token: CitationToken,
    ) -> tuple[CitationDiagnostic, ...]:
        """Return missing- and unknown-ID diagnostics for one citation token."""
        if not token.rids:
            return (
                CitationDiagnostic(
                    code=CitationDiagnosticCode.MISSING_RID,
                    rids=(),
                    source_line=token.source_line,
                ),
            )
        if not self._known_rids:
            return ()
        unknown_rids = tuple(rid for rid in token.rids if rid not in self._known_rids)
        if not unknown_rids:
            return ()
        return (
            CitationDiagnostic(
                code=CitationDiagnosticCode.UNKNOWN_RID,
                rids=unknown_rids,
                source_line=token.source_line,
            ),
        )

    @staticmethod
    def _render_markers(rids: Sequence[str]) -> str:
        """Render ordered IDs as an unambiguous comma-separated marker cluster."""
        return ", ".join(f"[xref:{rid}]" for rid in rids)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Collapse layout whitespace while retaining semantic punctuation."""
        normalized = text.replace("▪", " ")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
        normalized = re.sub(r"([\(\[\{])\s+", r"\1", normalized)
        return re.sub(r"\s+([\)\]\}])", r"\1", normalized)

    @staticmethod
    def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
        """Return values once each while retaining first-occurrence order."""
        return tuple(dict.fromkeys(values))
