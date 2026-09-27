"""Boundary and performance tests for deterministic sentence segmentation."""

from dataclasses import FrozenInstanceError
import time
import unittest

from pmcortex.sentence_segmenter import (
    DEFAULT_SENTENCE_SEGMENTATION_CONFIG,
    SentenceSegmentationConfig,
    SentenceSegmenter,
)


class TestSentenceSegmenter(unittest.TestCase):
    """Exercise punctuation, lexical exceptions, brackets, and markers."""

    def setUp(self) -> None:
        """Create one stateless segmenter for each test."""
        self.segmenter = SentenceSegmenter()

    def test_public_api_validates_inputs_and_configuration(self) -> None:
        """Public construction and splitting reject technically invalid input."""
        with self.assertRaises(TypeError):
            SentenceSegmenter(config="default")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.segmenter.split(123)  # type: ignore[arg-type]
        with self.assertRaises(FrozenInstanceError):
            setattr(
                DEFAULT_SENTENCE_SEGMENTATION_CONFIG,
                "abbreviations",
                frozenset(),
            )

    def test_empty_whitespace_and_unterminated_text(self) -> None:
        """Empty input yields no sentence and unterminated prose stays intact."""
        self.assertEqual(self.segmenter.split(""), ())
        self.assertEqual(self.segmenter.split(" \n\t "), ())
        self.assertEqual(
            self.segmenter.split("One sentence without final punctuation"),
            ("One sentence without final punctuation",),
        )

    def test_period_question_exclamation_and_clusters_end_sentences(self) -> None:
        """All configured terminal forms split before a plausible next token."""
        text = "First. Second? Third! Fourth?! Fifth!? Sixth."

        self.assertEqual(
            self.segmenter.split(text),
            (
                "First.",
                "Second?",
                "Third!",
                "Fourth?!",
                "Fifth!?",
                "Sixth.",
            ),
        )

    def test_decimals_initials_and_abbreviations_do_not_split(self) -> None:
        """Periods in numeric and configured lexical forms remain internal."""
        text = (
            "Dr. J.R. Smith measured 3.14 in the U.S. cohort. "
            "The next result followed."
        )

        self.assertEqual(
            self.segmenter.split(text),
            (
                "Dr. J.R. Smith measured 3.14 in the U.S. cohort.",
                "The next result followed.",
            ),
        )

    def test_et_al_parenthesis_and_marker_rules_remain_stable(self) -> None:
        """Author continuations stay intact while a trailing marker may close."""
        self.assertEqual(
            self.segmenter.split("Smith et al. (2020) reported this result."),
            ("Smith et al. (2020) reported this result.",),
        )
        self.assertEqual(
            self.segmenter.split(
                "Smith et al. [xref:R1] reported this result."
            ),
            ("Smith et al. [xref:R1] reported this result.",),
        )
        self.assertEqual(
            self.segmenter.split(
                "Smith et al. [xref:R1] Subsequent work continued."
            ),
            ("Smith et al. [xref:R1]", "Subsequent work continued."),
        )

    def test_nested_and_unfinished_parentheses_follow_recovery_rules(self) -> None:
        """Nested content stays intact and a closing bracket permits recovery."""
        self.assertEqual(
            self.segmenter.split(
                "Finding (with a nested (detail)). Next finding."
            ),
            (
                "Finding (with a nested (detail)).",
                "Next finding.",
            ),
        )
        self.assertEqual(
            self.segmenter.split(
                "Finding (with an unfinished detail. still related."
            ),
            ("Finding (with an unfinished detail. still related.",),
        )
        self.assertEqual(
            self.segmenter.split(
                "Finding (with (partially closed). Next finding."
            ),
            (
                "Finding (with (partially closed).",
                "Next finding.",
            ),
        )

    def test_closing_and_opening_quotes_are_assigned_to_correct_sentence(self) -> None:
        """Closing quotes stay left while opening quotes are inspected on right."""
        text = '"Quoted statement." "Is this supported?" Yes it is!'

        self.assertEqual(
            self.segmenter.split(text),
            (
                '"Quoted statement."',
                '"Is this supported?"',
                "Yes it is!",
            ),
        )

    def test_markers_before_and_after_punctuation_keep_sentence_ownership(self) -> None:
        """Marker position determines whether the following cluster moves left."""
        self.assertEqual(
            self.segmenter.split(
                "First claim. [xref:R1], [xref:R2] Next sentence."
            ),
            (
                "First claim. [xref:R1], [xref:R2]",
                "Next sentence.",
            ),
        )
        self.assertEqual(
            self.segmenter.split(
                "First claim [xref:R1]. [xref:R2] Next sentence."
            ),
            (
                "First claim [xref:R1].",
                "[xref:R2] Next sentence.",
            ),
        )
        self.assertEqual(
            self.segmenter.split("Question? [xref:R1] Next sentence."),
            ("Question? [xref:R1]", "Next sentence."),
        )

    def test_marker_lists_and_reverse_ranges_remain_single_clusters(self) -> None:
        """Segmentation retains marker lists without judging range validity."""
        self.assertEqual(
            self.segmenter.split(
                "Claim. [xref:R1], [xref:R2] and [xref:R3] Next."
            ),
            (
                "Claim. [xref:R1], [xref:R2] and [xref:R3]",
                "Next.",
            ),
        )
        self.assertEqual(
            self.segmenter.split(
                "Claim. [xref:R5] – [xref:R3] Next sentence."
            ),
            (
                "Claim. [xref:R5] – [xref:R3]",
                "Next sentence.",
            ),
        )

    def test_custom_immutable_configuration_is_supported(self) -> None:
        """A caller can supply a fully typed closed segmentation rule set."""
        config = SentenceSegmentationConfig(
            abbreviations=frozenset({"custom."}),
            marker_cluster_pattern=(
                DEFAULT_SENTENCE_SEGMENTATION_CONFIG.marker_cluster_pattern
            ),
            marker_at_end_pattern=(
                DEFAULT_SENTENCE_SEGMENTATION_CONFIG.marker_at_end_pattern
            ),
            initial_token_pattern=(
                DEFAULT_SENTENCE_SEGMENTATION_CONFIG.initial_token_pattern
            ),
            terminal_characters=frozenset({"."}),
            closing_quote_characters=frozenset({'"'}),
            opening_characters=frozenset({'"'}),
        )

        result = SentenceSegmenter(config).split(
            "A custom. Result remains together. Final sentence."
        )

        self.assertEqual(
            result,
            (
                "A custom. Result remains together.",
                "Final sentence.",
            ),
        )

    def test_long_text_completes_with_linear_scale_headroom(self) -> None:
        """A moderate long-text regression catches accidental quadratic scans."""
        text = " ".join(
            f"Sentence {number}." for number in range(10_000)
        )

        started_at = time.perf_counter()
        sentences = self.segmenter.split(text)
        elapsed_seconds = time.perf_counter() - started_at

        self.assertEqual(len(sentences), 10_000)
        self.assertLess(elapsed_seconds, 2.0)


if __name__ == "__main__":
    unittest.main()
