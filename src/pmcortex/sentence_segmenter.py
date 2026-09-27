"""Deterministically segment normalized JATS text into sentences."""

from dataclasses import dataclass
import re


_MARKER_TEXT = r"\[xref:[^\]]+\]"
_MARKER_CLUSTER_PATTERN = re.compile(
    (
        rf"{_MARKER_TEXT}"
        rf"(?:\s*(?:[,;/&–—−-]|\band\b|\bto\b)\s*{_MARKER_TEXT})*"
    ),
    flags=re.IGNORECASE,
)
_MARKER_AT_END_PATTERN = re.compile(
    rf"{_MARKER_TEXT}\s*",
    flags=re.IGNORECASE,
)
_INITIAL_TOKEN_PATTERN = re.compile(r"(?:[A-Z]\.)+")


@dataclass(frozen=True, slots=True)
class SentenceSegmentationConfig:
    """Immutable lexical rules used by :class:`SentenceSegmenter`.

    Attributes:
        abbreviations: Lowercase period-terminated forms that do not end a
            sentence on their own.
        marker_cluster_pattern: Pattern for one or more citation markers joined
            by supported list or range separators.
        marker_at_end_pattern: Pattern for a marker immediately before terminal
            punctuation.
        initial_token_pattern: Pattern for one or more dotted uppercase initials.
        terminal_characters: Characters recognized as sentence punctuation.
        closing_quote_characters: Quote characters retained at the end of the
            preceding sentence.
        opening_characters: Characters ignored when checking whether the next
            visible token begins with an uppercase letter or digit.
    """

    abbreviations: frozenset[str]
    marker_cluster_pattern: re.Pattern[str]
    marker_at_end_pattern: re.Pattern[str]
    initial_token_pattern: re.Pattern[str]
    terminal_characters: frozenset[str]
    closing_quote_characters: frozenset[str]
    opening_characters: frozenset[str]

    def __post_init__(self) -> None:
        """Validate immutable segmentation configuration values."""
        if not isinstance(self.abbreviations, frozenset):
            raise TypeError("abbreviations must be a frozenset")
        if not all(isinstance(value, str) for value in self.abbreviations):
            raise TypeError("abbreviations must contain only strings")
        for name, pattern in (
            ("marker_cluster_pattern", self.marker_cluster_pattern),
            ("marker_at_end_pattern", self.marker_at_end_pattern),
            ("initial_token_pattern", self.initial_token_pattern),
        ):
            if not isinstance(pattern, re.Pattern):
                raise TypeError(f"{name} must be a compiled regular expression")
        for name, characters in (
            ("terminal_characters", self.terminal_characters),
            ("closing_quote_characters", self.closing_quote_characters),
            ("opening_characters", self.opening_characters),
        ):
            if not isinstance(characters, frozenset):
                raise TypeError(f"{name} must be a frozenset")
            if not all(
                isinstance(character, str) and len(character) == 1
                for character in characters
            ):
                raise ValueError(f"{name} must contain one-character strings")


DEFAULT_SENTENCE_SEGMENTATION_CONFIG = SentenceSegmentationConfig(
    abbreviations=frozenset({
        "e.g.",
        "i.e.",
        "etc.",
        "dr.",
        "prof.",
        "ph.d.",
        "fig.",
        "u.s.",
        "vs.",
    }),
    marker_cluster_pattern=_MARKER_CLUSTER_PATTERN,
    marker_at_end_pattern=_MARKER_AT_END_PATTERN,
    initial_token_pattern=_INITIAL_TOKEN_PATTERN,
    terminal_characters=frozenset({".", "?", "!"}),
    closing_quote_characters=frozenset({'"', "'", "”", "’"}),
    opening_characters=frozenset({'"', "'", "“", "‘", "(", "["}),
)


class SentenceSegmenter:
    """Split normalized prose while preserving citation-marker association."""

    __slots__ = ("_config",)

    def __init__(
        self,
        config: SentenceSegmentationConfig | None = None,
    ) -> None:
        """Initialize a segmenter with immutable lexical rules.

        Args:
            config: Optional segmentation rules. The module default is used
                only when this value is ``None``.

        Raises:
            TypeError: If ``config`` has the wrong type.
        """
        if config is not None and not isinstance(
            config,
            SentenceSegmentationConfig,
        ):
            raise TypeError("config must be a SentenceSegmentationConfig or None")
        self._config = (
            DEFAULT_SENTENCE_SEGMENTATION_CONFIG
            if config is None
            else config
        )

    def split(self, text: str) -> tuple[str, ...]:
        """Split text at deterministic sentence boundaries.

        Periods, question marks, exclamation marks, and combinations such as
        ``?!`` may end a sentence. A boundary requires separating whitespace
        and a following uppercase letter, digit, opening quote/bracket, or a
        citation marker that belongs to the next sentence. Citation-marker
        clusters following punctuation remain attached to the preceding
        sentence unless that punctuation already follows a marker.

        Args:
            text: Normalized marker-bearing or visible source text.

        Returns:
            Sentences in source order. Empty or whitespace-only input returns
            an empty tuple.

        Raises:
            TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        sentences: list[str] = []
        sentence_start = 0
        parenthesis_depth = 0
        index = 0
        while index < len(text):
            character = text[index]
            if character in "([":
                parenthesis_depth += 1
            elif character in ")]":
                parenthesis_depth = max(0, parenthesis_depth - 1)

            if character not in self._config.terminal_characters:
                index += 1
                continue

            punctuation_end = self._terminal_cluster_end(text, index)
            can_end_at_depth = parenthesis_depth == 0
            follows_closing_bracket = index > 0 and text[index - 1] in ")]"
            can_recover_unclosed_parenthesis = (
                follows_closing_bracket and not can_end_at_depth
            )
            is_boundary_candidate = (
                can_end_at_depth or can_recover_unclosed_parenthesis
            )
            if not is_boundary_candidate:
                index = punctuation_end
                continue

            parenthesis_depth = 0
            punctuation = text[index:punctuation_end]
            is_plain_period = punctuation == "."
            is_decimal_point = (
                is_plain_period
                and index > 0
                and punctuation_end < len(text)
                and text[index - 1].isdigit()
                and text[punctuation_end].isdigit()
            )
            prefix = text[max(sentence_start, index - 10):punctuation_end].lower()
            is_abbreviation = (
                is_plain_period
                and any(
                    prefix.endswith(abbreviation)
                    for abbreviation in self._config.abbreviations
                )
            )
            is_initial = is_plain_period and self._is_initial_token(
                text,
                sentence_start,
                punctuation_end,
            )

            sentence_end = self._closing_quote_end(text, punctuation_end)
            next_index = self._first_nonspace_index(text, sentence_end)
            has_separator = next_index > sentence_end
            marker_starts_next_sentence = False
            marker_match = self._config.marker_cluster_pattern.match(
                text,
                next_index,
            )
            if marker_match is not None:
                marker_starts_next_sentence = self._marker_precedes_punctuation(
                    text,
                    sentence_start,
                    index,
                )
                if not marker_starts_next_sentence:
                    sentence_end = self._closing_quote_end(
                        text,
                        marker_match.end(),
                    )
                    next_index = self._first_nonspace_index(text, sentence_end)
                    has_separator = next_index > sentence_end

            is_parenthetical_et_al_continuation = (
                prefix.endswith("et al.")
                and next_index < len(text)
                and text[next_index] == "("
            )
            next_starts_sentence = (
                marker_starts_next_sentence
                or self._starts_with_sentence_token(text, next_index)
            )
            blocks_boundary = (
                is_decimal_point
                or is_abbreviation
                or is_initial
                or is_parenthetical_et_al_continuation
            )
            should_split = (
                has_separator
                and not blocks_boundary
                and next_starts_sentence
            )
            if should_split:
                sentence = text[sentence_start:sentence_end].strip()
                if sentence:
                    sentences.append(sentence)
                sentence_start = next_index
                index = next_index
                continue

            index = punctuation_end

        tail = text[sentence_start:].strip()
        if tail:
            sentences.append(tail)
        return tuple(sentences)

    def _terminal_cluster_end(self, text: str, start: int) -> int:
        """Return the exclusive end of consecutive terminal punctuation."""
        end = start + 1
        while end < len(text) and text[end] in self._config.terminal_characters:
            end += 1
        return end

    def _closing_quote_end(self, text: str, start: int) -> int:
        """Return the end after quote characters closing the current sentence."""
        end = start
        while (
            end < len(text)
            and text[end] in self._config.closing_quote_characters
        ):
            end += 1
        return end

    @staticmethod
    def _first_nonspace_index(text: str, start: int) -> int:
        """Return the next non-whitespace index or the text length."""
        index = start
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    def _marker_precedes_punctuation(
        self,
        text: str,
        sentence_start: int,
        punctuation_start: int,
    ) -> bool:
        """Return whether a citation marker immediately precedes punctuation."""
        marker_start = text.rfind(
            "[xref:",
            sentence_start,
            punctuation_start,
        )
        if marker_start < 0:
            return False
        marker_match = self._config.marker_at_end_pattern.fullmatch(
            text,
            marker_start,
            punctuation_start,
        )
        return marker_match is not None

    def _is_initial_token(
        self,
        text: str,
        sentence_start: int,
        punctuation_end: int,
    ) -> bool:
        """Return whether punctuation completes a dotted uppercase initial."""
        token_start = punctuation_end - 1
        while (
            token_start > sentence_start
            and not text[token_start - 1].isspace()
        ):
            token_start -= 1
        token = text[token_start:punctuation_end]
        return self._config.initial_token_pattern.fullmatch(token) is not None

    def _starts_with_sentence_token(self, text: str, start: int) -> bool:
        """Return whether visible content at ``start`` can begin a sentence."""
        index = start
        while index < len(text) and text[index] in self._config.opening_characters:
            index += 1
        if index >= len(text):
            return False
        return text[index].isupper() or text[index].isdigit()
