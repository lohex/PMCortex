"""Regression tests for structure-aware JATS citation normalization."""

import unittest

from lxml import etree

from pmcortex.citation_normalizer import (
    CitationDiagnosticCode,
    CitationForm,
    CitationNormalizer,
    NormalizedCitationText,
)
from pmcortex.jatsparser import JATSDocument, _ArticleParseSession
from pmcortex.models import (
    CitationCleanupAction,
    PositionedSentence,
    Reference,
    SentencePosition,
)
from pmcortex.serialization import render_positioned_sentences
from pmcortex.sentence_segmenter import SentenceSegmenter


class TestCitationNormalizer(unittest.TestCase):
    """Exercise citation shapes observed in PMC JATS articles."""

    @staticmethod
    def _normalize(
        xml: str,
        normalizer: CitationNormalizer,
    ) -> NormalizedCitationText:
        """Normalize one XML snippet with the supplied article reference map."""
        element = etree.fromstring(xml.encode("utf-8"))
        return normalizer.normalize(normalizer.tokenize(element))

    @staticmethod
    def _numbered_normalizer(
        reference_order: tuple[str, ...],
    ) -> CitationNormalizer:
        """Build a normalizer whose visible labels are RID digits."""
        labels = {
            rid: "".join(character for character in rid if character.isdigit())
            for rid in reference_order
        }
        return CitationNormalizer(reference_order, labels)

    def test_range_uses_labels_instead_of_rid_shape_or_document_order(self) -> None:
        """Numeric labels must correct opaque RIDs and a misordered ref list."""
        normalizer = CitationNormalizer(
            ("CR4", "CR5", "CR6", "CR8", "CR7"),
            {f"CR{number}": str(number) for number in range(4, 9)},
        )

        result = self._normalize(
            """
            <p>Evidence [<xref ref-type="bibr" rid="CR4">4</xref>
            −<xref ref-type="bibr" rid="CR7">7</xref>] supports this.</p>
            """,
            normalizer,
        )

        self.assertEqual(
            result.text_with_markers,
            (
                "Evidence [xref:CR4], [xref:CR5], [xref:CR6], "
                "[xref:CR7] supports this."
            ),
        )
        self.assertEqual(result.query_text, "Evidence supports this.")
        self.assertEqual(result.cited_rids, ("CR4", "CR5", "CR6", "CR7"))
        self.assertEqual(result.diagnostics, ())

    def test_single_xref_numeric_range_and_list_resolve_reference_labels(self) -> None:
        """One xref may encode several targets even when its RID names only one."""
        normalizer = self._numbered_normalizer(
            tuple(f"R{number}" for number in range(1, 11))
        )

        range_result = self._normalize(
            '<p>Range <xref ref-type="bibr" rid="R6">6–10</xref>.</p>',
            normalizer,
        )
        list_result = self._normalize(
            '<p>List <xref ref-type="bibr" rid="R1">1, 6, 9</xref>.</p>',
            normalizer,
        )
        word_range_result = self._normalize(
            '<p>Range <xref ref-type="bibr" rid="R6">6 to 10</xref>.</p>',
            normalizer,
        )

        self.assertEqual(
            range_result.cited_rids,
            ("R6", "R7", "R8", "R9", "R10"),
        )
        self.assertEqual(
            range_result.text_with_markers,
            "Range [xref:R6], [xref:R7], [xref:R8], [xref:R9], [xref:R10].",
        )
        self.assertEqual(list_result.cited_rids, ("R1", "R6", "R9"))
        self.assertEqual(
            list_result.text_with_markers,
            "List [xref:R1], [xref:R6], [xref:R9].",
        )
        self.assertEqual(
            word_range_result.cited_rids,
            ("R6", "R7", "R8", "R9", "R10"),
        )

    def test_opaque_rid_is_preserved_without_parsing_its_spelling(self) -> None:
        """RID punctuation must not be mistaken for a numeric reference label."""
        first_rid = "pone.0281234.ref-001"
        second_rid = "pone.0281234.ref-002"
        normalizer = CitationNormalizer(
            (first_rid, second_rid),
            {first_rid: "41", second_rid: "42"},
        )

        result = self._normalize(
            (
                '<p>Evidence <xref ref-type="bibr" '
                f'rid="{second_rid}"><sup>42</sup></xref> supports this.</p>'
            ),
            normalizer,
        )

        self.assertEqual(result.cited_rids, (second_rid,))
        self.assertEqual(
            result.text_with_markers,
            f"Evidence [xref:{second_rid}] supports this.",
        )

    def test_marker_removal_preserves_grammatical_comma(self) -> None:
        """Only separators between markers may disappear from the query."""
        normalizer = self._numbered_normalizer(("R1", "R2"))
        result = self._normalize(
            """
            <p>Rates were established
            <xref ref-type="bibr" rid="R1">[1]</xref>,
            <xref ref-type="bibr" rid="R2">[2]</xref>, and mortality fell.</p>
            """,
            normalizer,
        )

        self.assertEqual(
            result.text_with_markers,
            "Rates were established [xref:R1], [xref:R2], and mortality fell.",
        )
        self.assertEqual(
            result.query_text,
            "Rates were established, and mortality fell.",
        )

    def test_marker_removal_preserves_unrelated_empty_delimiters(self) -> None:
        """Removing citations must not rewrite empty syntax elsewhere in prose."""
        query = CitationNormalizer.remove_markers(
            "Call foo() and retain [] before [xref:R1]."
        )

        self.assertEqual(query, "Call foo() and retain [] before.")

        parser = _ArticleParseSession()
        parser.sentences = [
            PositionedSentence(
                position=SentencePosition(
                    source_pmcid="PMC_TEST",
                    section_index=0,
                    paragraph_index=0,
                    sentence_index=0,
                ),
                text="Call foo() and retain [].",
            )
        ]
        self.assertEqual(
            render_positioned_sentences(tuple(parser.sentences)),
            "PMC_TEST/0/0/0\tCall foo() and retain [].",
        )

    def test_nested_semantic_xref_is_removed_and_classified_as_unsafe(self) -> None:
        """Nested author-year text must be absent from the retrieval query."""
        normalizer = self._numbered_normalizer(("R1", "R2"))
        result = self._normalize(
            """
            <p>Evidence<xref ref-type="bibr" rid="R1"><sup>1</sup></xref>
            follows <xref ref-type="bibr" rid="R2"><italic>Smith et al., 2020</italic></xref>.</p>
            """,
            normalizer,
        )

        self.assertEqual(
            result.text_with_markers,
            "Evidence [xref:R1] follows Smith et al., 2020 [xref:R2].",
        )
        self.assertEqual(
            result.query_text,
            "Evidence follows.",
        )
        self.assertEqual(
            result.citation_occurrences[-1].form,
            CitationForm.NARRATIVE_AUTHOR_YEAR,
        )
        self.assertEqual(result.unsafe_citation_rids, ("R2",))

    def test_multiple_rids_are_emitted_as_individual_markers(self) -> None:
        """Whitespace-separated JATS IDREFS must become independent targets."""
        normalizer = self._numbered_normalizer(("R1", "R2"))
        result = self._normalize(
            '<p>Evidence [<xref ref-type="bibr" rid="R1 R2">1, 2</xref>] supports this.</p>',
            normalizer,
        )

        self.assertEqual(
            result.text_with_markers,
            "Evidence [xref:R1], [xref:R2] supports this.",
        )
        self.assertEqual(result.query_text, "Evidence supports this.")
        self.assertEqual(result.cited_rids, ("R1", "R2"))

    def test_split_author_year_text_is_removed_and_marked_unsafe(self) -> None:
        """Balanced narrative citations may span an xref tail or several xrefs."""
        normalizer = self._numbered_normalizer(("R1", "R2"))
        split_result = self._normalize(
            '<p>Prior work <xref ref-type="bibr" rid="R1">Smith (</xref>2020) supports this.</p>',
            normalizer,
        )
        multi_result = self._normalize(
            """
            <p><xref ref-type="bibr" rid="R1">Nikolaivits et al. (2019</xref>,
            <xref ref-type="bibr" rid="R2">2020)</xref> screened strains.</p>
            """,
            normalizer,
        )

        self.assertEqual(
            split_result.text_with_markers,
            "Prior work Smith (2020) [xref:R1] supports this.",
        )
        self.assertEqual(
            split_result.query_text,
            "Prior work supports this.",
        )
        self.assertEqual(
            multi_result.text_with_markers,
            (
                "Nikolaivits et al. (2019 [xref:R1], 2020) "
                "[xref:R2] screened strains."
            ),
        )
        self.assertEqual(
            multi_result.query_text,
            "screened strains.",
        )
        self.assertEqual(split_result.unsafe_citation_rids, ("R1",))
        self.assertEqual(multi_result.unsafe_citation_rids, ("R1", "R2"))

    def test_split_round_parenthesis_and_xref_terminal_period_are_preserved(self) -> None:
        """Citation delimiters and sentence punctuation may end in the xref tail."""
        normalizer = self._numbered_normalizer(("R1", "R2"))
        split_result = self._normalize(
            '<p>Evidence <xref ref-type="bibr" rid="R1">(1</xref>) supports this.</p>',
            normalizer,
        )
        period_result = self._normalize(
            '<p>First claim <xref ref-type="bibr" rid="R1">1.</xref> Next sentence.</p>',
            normalizer,
        )
        separate_citations_result = self._normalize(
            (
                '<p>First claim <xref ref-type="bibr" rid="R1">1.</xref> '
                '<xref ref-type="bibr" rid="R2">2</xref> independently reports this.</p>'
            ),
            normalizer,
        )

        self.assertEqual(
            split_result.text_with_markers,
            "Evidence [xref:R1] supports this.",
        )
        self.assertEqual(
            period_result.text_with_markers,
            "First claim [xref:R1]. Next sentence.",
        )
        self.assertEqual(period_result.query_text, "First claim. Next sentence.")
        self.assertEqual(
            separate_citations_result.text_with_markers,
            (
                "First claim [xref:R1]. "
                "[xref:R2] independently reports this."
            ),
        )
        self.assertEqual(
            SentenceSegmenter().split(
                separate_citations_result.text_with_markers
            ),
            (
                "First claim [xref:R1].",
                "[xref:R2] independently reports this.",
            ),
        )

    def test_public_api_rejects_iterators_and_recognizes_label_only_ids(self) -> None:
        """Public normalization inputs and known-ID diagnostics must be consistent."""
        normalizer = CitationNormalizer((), {"R1": "1"})

        with self.assertRaises(TypeError):
            normalizer.normalize(iter(()))  # type: ignore[arg-type]

        known_result = self._normalize(
            '<p><xref ref-type="bibr" rid="R1">1</xref></p>',
            normalizer,
        )
        unknown_result = self._normalize(
            '<p><xref ref-type="bibr" rid="R2">2</xref></p>',
            normalizer,
        )

        self.assertEqual(known_result.diagnostics, ())
        self.assertEqual(
            unknown_result.diagnostics[0].code,
            CitationDiagnosticCode.UNKNOWN_RID,
        )

    def test_semantic_dash_does_not_expand_a_numeric_reference_range(self) -> None:
        """A dash between author-year xrefs is prose, not an implicit RID range."""
        normalizer = self._numbered_normalizer(("R1", "R2", "R3"))
        result = self._normalize(
            (
                '<p><xref ref-type="bibr" rid="R1">Smith, 2019</xref>'
                '–<xref ref-type="bibr" rid="R3">Jones, 2021</xref></p>'
            ),
            normalizer,
        )

        self.assertEqual(result.cited_rids, ("R1", "R3"))
        self.assertNotIn("[xref:R2]", result.text_with_markers)
        self.assertEqual(
            CitationNormalizer.extract_rids(result.text_with_markers),
            result.cited_rids,
        )

    def test_partial_labels_use_a_consistent_document_order_anchor(self) -> None:
        """Known endpoint labels may validate missing labels resolved by ref order."""
        reference_order = tuple(f"R{number}" for number in range(1, 10))
        normalizer = CitationNormalizer(
            reference_order,
            {"R1": "1", "R9": "9"},
        )

        result = self._normalize(
            '<p>List <xref ref-type="bibr" rid="R1">1, 6, 9</xref>.</p>',
            normalizer,
        )

        self.assertEqual(result.cited_rids, ("R1", "R6", "R9"))

    def test_conflicting_visible_label_does_not_fabricate_a_known_rid(self) -> None:
        """A single visible label cannot override its structural xref target."""
        normalizer = CitationNormalizer(("R1",), {"R1": "1"})
        result = self._normalize(
            '<p>Claim <xref ref-type="bibr" rid="R999">1</xref>.</p>',
            normalizer,
        )

        self.assertEqual(result.cited_rids, ("R999",))
        self.assertNotIn("[xref:R1]", result.text_with_markers)
        self.assertEqual(
            result.diagnostics[0].code,
            CitationDiagnosticCode.UNKNOWN_RID,
        )

    def test_descending_range_without_ref_labels_is_not_expanded(self) -> None:
        """Visible descending endpoints must override a misleading ref order."""
        normalizer = CitationNormalizer(
            ("R5", "R4", "R3"),
            {"R5": None, "R4": None, "R3": None},
        )
        result = self._normalize(
            (
                '<p><xref ref-type="bibr" rid="R5">5</xref>'
                '–<xref ref-type="bibr" rid="R3">3</xref></p>'
            ),
            normalizer,
        )

        self.assertEqual(result.cited_rids, ("R5", "R3"))
        self.assertIn(
            CitationDiagnosticCode.INVALID_RANGE,
            [diagnostic.code for diagnostic in result.diagnostics],
        )

    def test_block_children_are_separated_when_tokenizing_a_caption(self) -> None:
        """Adjacent caption paragraphs must not be concatenated."""
        normalizer = self._numbered_normalizer(("R1",))
        result = self._normalize(
            (
                '<caption><p>First sentence.</p><p>Second claim '
                '<xref ref-type="bibr" rid="R1">1.</xref></p></caption>'
            ),
            normalizer,
        )

        self.assertEqual(
            result.text_with_markers,
            "First sentence. Second claim [xref:R1].",
        )

    def test_parenthetical_author_year_is_removed_without_rejecting_query(self) -> None:
        """A structurally wrapped author-year xref is safe to remove in place."""
        normalizer = self._numbered_normalizer(("R1",))
        result = self._normalize(
            (
                '<p>The enzyme remained active '
                '(<xref ref-type="bibr" rid="R1">Trincone, 2011</xref>).</p>'
            ),
            normalizer,
        )

        self.assertEqual(
            result.text_with_markers,
            "The enzyme remained active (Trincone, 2011 [xref:R1]).",
        )
        self.assertEqual(
            result.query_text_with_markers,
            "The enzyme remained active ([xref:R1]).",
        )
        self.assertEqual(result.query_text, "The enzyme remained active.")
        self.assertEqual(
            result.citation_occurrences[0].form,
            CitationForm.PARENTHETICAL_AUTHOR_YEAR,
        )
        self.assertEqual(result.unsafe_citation_rids, ())

    def test_narrative_et_al_and_substantive_parentheses_are_preserved(self) -> None:
        """Citation normalization must not pre-empt later semantic filters."""
        normalizer = CitationNormalizer(
            ("CR22", "R1", "R2"),
            {"CR22": "22", "R1": "27", "R2": "44"},
        )
        narrative_result = self._normalize(
            """
            <p>Qi et al.<sup><xref ref-type="bibr" rid="CR22">22</xref></sup>,
            who found an effect.</p>
            """,
            normalizer,
        )
        mixed_result = self._normalize(
            """
            <p>This occurred elsewhere (e.g. skin inflammation
            <xref ref-type="bibr" rid="R1"><sup>27</sup></xref> or tuberculous meningitis
            <xref ref-type="bibr" rid="R2"><sup>44</sup></xref>).</p>
            """,
            normalizer,
        )

        self.assertEqual(
            narrative_result.query_text,
            "Qi et al., who found an effect.",
        )
        self.assertEqual(
            mixed_result.query_text,
            "This occurred elsewhere (e.g. skin inflammation or tuberculous meningitis).",
        )

    def test_missing_rid_preserves_raw_text_but_cleans_query_and_reports(self) -> None:
        """Malformed xrefs retain raw evidence but cannot leak into query text."""
        normalizer = CitationNormalizer((), {})
        result = self._normalize(
            """
            <p>Prior work <xref ref-type="bibr"><italic>Smith, 2020</italic></xref>
            supports this.</p>
            """,
            normalizer,
        )

        self.assertEqual(
            result.text_with_markers,
            "Prior work Smith, 2020 supports this.",
        )
        self.assertEqual(result.query_text, "Prior work supports this.")
        self.assertEqual(result.cited_rids, ())
        self.assertEqual(result.diagnostics[0].code, CitationDiagnosticCode.MISSING_RID)

    def test_invalid_descending_range_keeps_endpoints_and_reports_diagnostic(self) -> None:
        """An unsafe range must never fabricate or delete its endpoint citations."""
        normalizer = self._numbered_normalizer(
            tuple(f"R{number}" for number in range(1, 6))
        )
        result = self._normalize(
            '<p>Evidence [<xref ref-type="bibr" rid="R5">5</xref>–<xref ref-type="bibr" rid="R3">3</xref>].</p>',
            normalizer,
        )

        self.assertEqual(
            result.text_with_markers,
            "Evidence [xref:R5] – [xref:R3].",
        )
        self.assertEqual(result.query_text, "Evidence.")
        self.assertEqual(result.cited_rids, ("R5", "R3"))
        self.assertIn(
            CitationDiagnosticCode.INVALID_RANGE,
            [diagnostic.code for diagnostic in result.diagnostics],
        )


class TestCitationNormalizerIntegration(unittest.TestCase):
    """Verify that normalized markers become positioned, unique PMID hits."""

    def test_jatsparser_expands_range_and_aligns_query_with_fulltext(self) -> None:
        """Parser integration must retain every resolvable positive in a range."""
        parser = _ArticleParseSession()
        root = etree.fromstring(
            b"""
            <article xmlns="http://jats.nlm.nih.gov">
              <body><sec><p>Evidence [<xref ref-type="bibr" rid="R1">1</xref>
              -<xref ref-type="bibr" rid="R3">3</xref>] supports this.</p></sec></body>
              <back><ref-list>
                <ref id="R1"><label>1</label></ref>
                <ref id="R2"><label>2</label></ref>
                <ref id="R3"><label>3</label></ref>
              </ref-list></back>
            </article>
            """,
            parser=parser._xml_parser,
        )
        parser.document = JATSDocument.from_root(root)
        parser.pmcid = "PMC_TEST"
        references = [
            Reference(
                rid=f"R{number}",
                label=str(number),
                doi=None,
                pmid=str(100 + number),
                pmcid=None,
                year=None,
                title=None,
                journal=None,
                authors=(),
            )
            for number in range(1, 4)
        ]

        parser.extract_contexts(
            parser.extract_sections(),
            references,
            save_sentences=True,
        )

        self.assertEqual(len(parser.contexts), 1)
        self.assertEqual(parser.contexts[0].query, "Evidence supports this.")
        self.assertEqual(parser.contexts[0].hits, ("101", "102", "103"))
        self.assertEqual(parser.contexts[0].n_hits, 3)
        self.assertEqual(
            render_positioned_sentences(tuple(parser.sentences)),
            "PMC_TEST/0/0/0\tEvidence [1 -3] supports this.",
        )

    def test_parser_keeps_clean_parenthetical_query_and_rejects_narrative(self) -> None:
        """Only structurally safe semantic citations may become contexts."""
        parser = _ArticleParseSession()
        root = etree.fromstring(
            b"""
            <article xmlns="http://jats.nlm.nih.gov">
              <body><sec>
                <p>Safe claim (<xref ref-type="bibr" rid="R1">Smith, 2020</xref>).</p>
                <p><xref ref-type="bibr" rid="R2">Jones (2021)</xref> reported this.</p>
              </sec></body>
              <back><ref-list>
                <ref id="R1"><label>1</label></ref>
                <ref id="R2"><label>2</label></ref>
              </ref-list></back>
            </article>
            """,
            parser=parser._xml_parser,
        )
        parser.document = JATSDocument.from_root(root)
        parser.pmcid = "PMC_TEST"
        references = [
            Reference(
                rid=f"R{number}",
                label=str(number),
                doi=None,
                pmid=str(100 + number),
                pmcid=None,
                year=2019 + number,
                title=None,
                journal=None,
                authors=(),
            )
            for number in range(1, 3)
        ]

        parser.extract_contexts(
            parser.extract_sections(),
            references,
            save_sentences=True,
        )

        self.assertEqual(len(parser.contexts), 1)
        self.assertEqual(parser.contexts[0].query, "Safe claim.")
        self.assertEqual(parser.contexts[0].query_raw, "Safe claim (Smith, 2020).")
        self.assertEqual(parser.sentences[0].text, parser.contexts[0].query_raw)
        self.assertEqual(parser.sentences[1].text, "Jones (2021) reported this.")
        self.assertEqual(
            parser.contexts[0].citation_forms,
            (CitationForm.PARENTHETICAL_AUTHOR_YEAR.value,),
        )
        self.assertEqual(
            parser.contexts[0].citation_cleanup_action,
            CitationCleanupAction.REMOVED_PARENTHETICAL_CITATION,
        )
        self.assertEqual(parser.unsafe_citation_rejections, 1)

    def test_citation_after_period_with_no_space_is_rejected(self) -> None:
        """Different visible and query sentence counts reject the whole block."""
        parser = _ArticleParseSession()
        root = etree.fromstring(
            b"""
            <article xmlns="http://jats.nlm.nih.gov">
              <body><sec><p>First claim.<xref ref-type="bibr" rid="R1"><sup>1</sup></xref>
              Second sentence.</p></sec></body>
              <back><ref-list><ref id="R1"><label>1</label></ref></ref-list></back>
            </article>
            """,
            parser=parser._xml_parser,
        )
        parser.document = JATSDocument.from_root(root)
        parser.pmcid = "PMC_TEST"
        references = [
            Reference(
                rid="R1",
                label="1",
                doi=None,
                pmid="101",
                pmcid=None,
                year=None,
                title=None,
                journal=None,
                authors=(),
            )
        ]

        parser.extract_contexts(
            parser.extract_sections(),
            references,
            save_sentences=True,
        )

        self.assertEqual(parser.contexts, [])
        self.assertEqual(parser.alignment_rejections, 1)
        self.assertEqual(parser.parser_diagnostics[0].code.value, "sentence_alignment_failed")
        self.assertEqual(
            render_positioned_sentences(tuple(parser.sentences)).splitlines(),
            ["PMC_TEST/0/0/0\tFirst claim.1 Second sentence."],
        )

    def test_sentence_splitting_handles_et_al_and_retained_range_separators(self) -> None:
        """Post-period markers must not suppress a following sentence boundary."""
        self.assertEqual(
            SentenceSegmenter().split(
                "Smith et al. [xref:R1] Subsequent work continued."
            ),
            ("Smith et al. [xref:R1]", "Subsequent work continued."),
        )
        self.assertEqual(
            SentenceSegmenter().split(
                "Smith et al. (2020) reported this result."
            ),
            ("Smith et al. (2020) reported this result.",),
        )
        self.assertEqual(
            SentenceSegmenter().split(
                "Smith et al. [xref:R1] reported this result."
            ),
            ("Smith et al. [xref:R1] reported this result.",),
        )
        self.assertEqual(
            SentenceSegmenter().split(
                "Claim. [xref:R5] – [xref:R3] Next sentence."
            ),
            ("Claim. [xref:R5] – [xref:R3]", "Next sentence."),
        )


if __name__ == "__main__":
    unittest.main()
