import unittest

from lxml import etree

from pmcortex.jatsparser import JATSDocument, JATSParser, _ArticleParseSession
from pmcortex.models import JATSArticle, Reference
from pmcortex.serialization import (
    author_lists_to_payload,
    contexts_to_dataframe,
    normalize_author_lists,
    references_to_dataframe,
    render_positioned_sentences,
)
from pmcortex.sentence_segmenter import SentenceSegmenter

from tests._shared import EXAMPLE_PMCIDS, existing_example_files


class TestJATSParser(unittest.TestCase):
    @staticmethod
    def _parser_from_xml(xml: str) -> _ArticleParseSession:
        parser = _ArticleParseSession()
        root = etree.fromstring(xml.encode("utf-8"), parser=parser._xml_parser)
        parser.document = JATSDocument.from_root(root)
        return parser

    def test_example_file_availability(self) -> None:
        existing = existing_example_files()
        self.assertEqual(len(existing), 5)
        self.assertNotIn("PMC440378", existing)
        self.assertEqual(sorted(existing.keys()), sorted([p for p in EXAMPLE_PMCIDS if p != "PMC440378"]))

    def test_parse_for_available_examples(self) -> None:
        parser = JATSParser()
        for pmcid, path in existing_example_files().items():
            with self.subTest(pmcid=pmcid):
                result = parser.parse(path, expected_pmcid=pmcid)
                article = result.article
                self.assertEqual(article.pmcid, pmcid)
                self.assertTrue(article.title)
                self.assertGreater(len(article.references), 0)
                self.assertGreater(len(article.sections), 0)

    def test_contexts_to_dataframe_has_expected_columns(self) -> None:
        pmcid, path = next(iter(existing_example_files().items()))
        result = JATSParser().parse(path, expected_pmcid=pmcid)
        df = contexts_to_dataframe(result.contexts)

        expected = {
            "source_pmcid",
            "section_index",
            "paragraph_index",
            "sentence_index",
            "query",
            "hits",
            "query_length",
            "n_hits",
            "context",
        }
        self.assertTrue(expected.issubset(set(df.columns)))
        self.assertTrue(df["source_pmcid"].eq(pmcid).all())
        self.assertNotIn("position", df.columns)

    def test_query_and_fulltext_share_structural_sentence_position(self) -> None:
        parser = self._parser_from_xml(
            """
            <article xmlns="http://jats.nlm.nih.gov">
              <body>
                <sec>
                  <p>
                    First sentence. Cited sentence
                    <xref ref-type="bibr" rid="R1">1</xref>.
                  </p>
                </sec>
              </body>
            </article>
            """
        )
        parser.pmcid = "PMC_TEST"
        sections = parser.extract_sections()
        references = (
            Reference(
                rid="R1",
                label="1",
                doi=None,
                pmid="12345",
                pmcid=None,
                year=None,
                title=None,
                journal=None,
                authors=(),
            ),
        )

        parser.extract_contexts(tuple(sections), references, save_sentences=True)

        context = contexts_to_dataframe(tuple(parser.contexts)).iloc[0]
        self.assertEqual(context["source_pmcid"], "PMC_TEST")
        self.assertEqual(context["section_index"], 0)
        self.assertEqual(context["paragraph_index"], 0)
        self.assertEqual(context["sentence_index"], 1)
        self.assertEqual(
            parser.contexts[0].position,
            parser.sentences[1].position,
        )
        self.assertEqual(
            parser.contexts[0].position.sentence_id,
            "PMC_TEST/0/0/1",
        )
        self.assertEqual(
            render_positioned_sentences(tuple(parser.sentences)).splitlines(),
            [
                (
                    "0/0/0\tFirst sentence."
                ),
                (
                    "0/0/1\tCited sentence [xref:R1]."
                ),
            ],
        )

    def test_context_extraction_uses_question_and_exclamation_boundaries(self) -> None:
        """Parser sentence positions follow the Phase-4 segmenter rules."""
        parser = self._parser_from_xml(
            """
            <article xmlns="http://jats.nlm.nih.gov">
              <body><sec><p>
                Initial question? Cited claim
                <xref ref-type="bibr" rid="R1">1</xref>! Final sentence.
              </p></sec></body>
            </article>
            """
        )
        parser.pmcid = "PMC_TEST"
        references = (
            Reference(
                rid="R1",
                label="1",
                doi=None,
                pmid="12345",
                pmcid=None,
                year=None,
                title=None,
                journal=None,
                authors=(),
            ),
        )

        parser.extract_contexts(
            parser.extract_sections(),
            references,
            save_sentences=True,
        )

        self.assertEqual(len(parser.sentences), 3)
        self.assertEqual(len(parser.contexts), 1)
        self.assertEqual(parser.contexts[0].position.sentence_index, 1)
        self.assertEqual(parser.contexts[0].query, "Cited claim!")

    def test_sources_to_dataframe_has_reference_fields(self) -> None:
        pmcid, path = next(iter(existing_example_files().items()))
        result = JATSParser().parse(path, expected_pmcid=pmcid)
        df = references_to_dataframe(pmcid, result.article.references)

        expected = {"rid", "label", "doi", "pmid", "pmcid", "year", "title", "journal", "authors"}
        self.assertTrue(expected.issubset(set(df.columns)))

    def test_split_sentences_does_not_split_inside_dotted_abbreviations(self) -> None:
        text = "This was conducted in the U.S. cohort."
        sentences = SentenceSegmenter().split(text)

        self.assertEqual(
            sentences,
            ("This was conducted in the U.S. cohort.",),
        )

    def test_replace_refs_extracts_figure_caption_as_separate_paragraph(self) -> None:
        parser = self._parser_from_xml(
            """
            <article xmlns="http://jats.nlm.nih.gov">
              <body>
                <sec>
                  <p>
                    Before <xref ref-type="bibr" rid="R1">1</xref> after.
                    <fig id="F1">
                      <label>Fig. 1.</label>
                      <caption><p>Caption with <xref ref-type="bibr" rid="R2">2</xref>.</p></caption>
                      <graphic xlink:href="fig.jpg" xmlns:xlink="http://www.w3.org/1999/xlink"/>
                    </fig>
                    Tail <disp-formula>E = mc^2</disp-formula> end.
                  </p>
                </sec>
              </body>
            </article>
            """
        )
        paragraph = parser._find_element(".//j:p")

        self.assertEqual(
            parser._replace_refs(paragraph),
            [
                "Before [xref:R1] after.",
                "Fig. 1. Caption with [xref:R2].",
                "Tail end.",
            ],
        )

    def test_replace_refs_moves_tail_text_before_split_closing_parenthesis(self) -> None:
        parser = self._parser_from_xml(
            """
            <article xmlns="http://jats.nlm.nih.gov">
              <body>
                <p>Prior work <xref ref-type="bibr" rid="R1">Smith (</xref>2020) supports this result.</p>
              </body>
            </article>
            """
        )
        paragraph = parser._find_element(".//j:p")

        self.assertEqual(
            parser._replace_refs(paragraph),
            ["Prior work Smith (2020) [xref:R1] supports this result."],
        )

    def test_replace_refs_skips_configured_non_text_nodes(self) -> None:
        parser = self._parser_from_xml(
            """
            <article xmlns="http://jats.nlm.nih.gov" xmlns:mml="http://www.w3.org/1998/Math/MathML">
              <body>
                <sec>
                  <p>
                    Alpha
                    <alternatives><p>drop alternatives</p></alternatives>
                    <tex-math>drop tex</tex-math>
                    <mml:math><mml:mi>x</mml:mi></mml:math>
                    <supplementary-material>
                      <media xlink:href="supp.docx" xmlns:xlink="http://www.w3.org/1999/xlink"/>
                      <p>drop supplementary</p>
                    </supplementary-material>
                    <chem-struct-wrap><p>drop chemistry</p></chem-struct-wrap>
                    Omega.
                  </p>
                  <table-wrap id="T1">
                    <label>Table 1.</label>
                    <caption><p>Overview with <xref ref-type="bibr" rid="R3">3</xref>.</p></caption>
                    <alternatives>
                      <graphic xlink:href="table.jpg" xmlns:xlink="http://www.w3.org/1999/xlink"/>
                    </alternatives>
                  </table-wrap>
                </sec>
              </body>
            </article>
            """
        )
        paragraph = parser._find_element(".//j:p")
        table = parser._find_element(".//j:table-wrap")

        self.assertEqual(parser._replace_refs(paragraph), ["Alpha Omega."])
        self.assertEqual(parser._replace_refs(table), ["Table 1. Overview with [xref:R3]."])

    def test_extract_sections_does_not_duplicate_nested_sections_for_body_paragraphs(self) -> None:
        parser = self._parser_from_xml(
            """
            <article xmlns="http://jats.nlm.nih.gov">
              <body>
                <p>Intro text.</p>
                <sec id="S1">
                  <title>Results</title>
                  <p>Nested section text.</p>
                </sec>
                <p>Closing text.</p>
              </body>
            </article>
            """
        )

        sections = parser.extract_sections()
        flattened = [
            text
            for section in sections
            for paragraph in section.paragraphs
            for text in parser._replace_refs(paragraph)
        ]

        self.assertEqual(
            [section.full_title for section in sections],
            ["Untitled Section", "Results", "Untitled Section"],
        )
        self.assertEqual(
            flattened,
            ["Intro text.", "Nested section text.", "Closing text."],
        )

    def test_normalize_author_lists_returns_expected_structure(self) -> None:
        article = JATSArticle(
            pmcid="PMC1",
            pmid=None,
            title="Example",
            abstract=None,
            authors=("Smith John", "Miller Anne"),
            references=(
                Reference(
                    rid="R1",
                    label="1",
                    doi=None,
                    pmid=None,
                    pmcid=None,
                    year=None,
                    title="Ref 1",
                    journal="Journal",
                    authors=("Doe Jane", "Roe Richard"),
                ),
                Reference(
                    rid="R2",
                    label="2",
                    doi=None,
                    pmid=None,
                    pmcid=None,
                    year=None,
                    title="Ref 2",
                    journal="Journal",
                    authors=("Brown Alice",),
                ),
            ),
            sections=(),
        )

        data = author_lists_to_payload(normalize_author_lists(article))

        self.assertEqual(
            data,
            {
                "authors": ["Smith J", "Miller A"],
                "cited_authors": [["Doe J", "Roe R"], ["Brown A"]],
            },
        )


if __name__ == "__main__":
    unittest.main()
