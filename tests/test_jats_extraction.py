"""Unit tests for stateless JATS content extraction."""

from pathlib import Path
import unittest

from lxml import etree

from pmcortex.jats_extraction import (
    JATSDocument,
    extract_article_authors,
    extract_caption_text,
    extract_metadata,
    extract_references,
    extract_sections,
    flatten_text,
    select_article_pmcid,
)


class TestJATSMetadataAndAuthors(unittest.TestCase):
    """Verify article-level extraction without parser-session state."""

    @staticmethod
    def _document(xml: str) -> JATSDocument:
        """Parse a compact test document with normal lxml defaults."""
        root = etree.fromstring(xml.encode("utf-8"))
        return JATSDocument.from_root(root)

    def test_extracts_metadata_and_prefers_structured_authors(self) -> None:
        """Nested title text, IDs, and structured names retain source order."""
        document = self._document(
            """
            <article xmlns="http://jats.nlm.nih.gov">
              <front><article-meta>
                <article-id pub-id-type="pmid">123456</article-id>
                <article-id pub-id-type="pmcid">9000003</article-id>
                <title-group><article-title>A <italic>nested</italic> title</article-title></title-group>
                <abstract><p>First <bold>abstract</bold>.</p></abstract>
                <contrib-group>
                  <contrib contrib-type="author"><name><surname>Smith</surname><given-names>Jane Ann</given-names></name></contrib>
                  <contrib contrib-type="author"><name><surname>DOE</surname><given-names>AB</given-names></name></contrib>
                  <contrib contrib-type="editor"><name><surname>Ignored</surname></name></contrib>
                </contrib-group>
              </article-meta></front>
            </article>
            """
        )

        metadata = extract_metadata(document)
        authors = extract_article_authors(document)

        self.assertEqual(metadata.title, "A nested title")
        self.assertEqual(metadata.abstract, "First abstract .")
        self.assertEqual(metadata.pmcid, "PMC9000003")
        self.assertEqual(metadata.pmid, "123456")
        self.assertEqual(authors, ("Smith JA", "DOE AB"))

    def test_string_name_is_used_when_structured_names_are_absent(self) -> None:
        """Article contributors may provide only free-form string names."""
        document = self._document(
            """
            <article><front><article-meta><contrib-group>
              <contrib contrib-type="author"><string-name>Consortium Name</string-name></contrib>
            </contrib-group></article-meta></front></article>
            """
        )

        self.assertEqual(extract_article_authors(document), ("Consortium Name",))

    def test_pmcid_selection_has_document_expected_filename_priority(self) -> None:
        """The named selector makes the three fallback levels explicit."""
        article_path = Path("/dataset/PMC_FROM_FILE.nxml")

        self.assertEqual(
            select_article_pmcid("PMC_FROM_XML", "PMC_EXPECTED", article_path),
            "PMC_FROM_XML",
        )
        self.assertEqual(
            select_article_pmcid(None, "PMC_EXPECTED", article_path),
            "PMC_EXPECTED",
        )
        self.assertEqual(
            select_article_pmcid(None, None, article_path),
            "PMC_FROM_FILE",
        )


class TestJATSReferenceExtraction(unittest.TestCase):
    """Verify reference encodings and documented fallback behavior."""

    @staticmethod
    def _document(xml: str) -> JATSDocument:
        """Parse a compact reference fixture."""
        root = etree.fromstring(xml.encode("utf-8"))
        return JATSDocument.from_root(root)

    def test_element_and_mixed_citations_use_expected_fallbacks(self) -> None:
        """Chapter titles, string names, PMCID normalization, and DOI regex work."""
        document = self._document(
            """
            <article><back><ref-list>
              <ref id="R1"><label>1</label><element-citation>
                <name><surname>Smith</surname><given-names>Jane</given-names></name>
                <article-title>Element title</article-title><source>Journal</source>
                <year>2024</year><pub-id pub-id-type="pmid">101</pub-id>
                <pub-id pub-id-type="pmcid">7654321</pub-id>
                <pub-id pub-id-type="doi">10.1000/element</pub-id>
              </element-citation></ref>
              <ref id="R2"><label>2</label><mixed-citation>
                <string-name>Brown Alice, Green Bob</string-name>
                <chapter-title>Mixed chapter</chapter-title><date>undated</date>
                Available as doi:10.5555/mixed.example.
              </mixed-citation></ref>
              <ref id="R3"><label>3</label></ref>
            </ref-list></back></article>
            """
        )

        references = extract_references(document)

        self.assertEqual(len(references), 2)
        first, second = references
        self.assertEqual(first.authors, ("Smith J",))
        self.assertEqual(first.pmcid, "PMC7654321")
        self.assertEqual(first.year, 2024)
        self.assertEqual(first.doi, "10.1000/element")
        self.assertEqual(second.authors, ("Brown Alice", "Green Bob"))
        self.assertEqual(second.title, "Mixed chapter")
        self.assertIsNone(second.year)
        self.assertEqual(second.doi, "10.5555/mixed.example")


class TestJATSSectionExtraction(unittest.TestCase):
    """Verify deterministic body grouping and parent-before-child preorder."""

    @staticmethod
    def _document(xml: str) -> JATSDocument:
        """Parse a compact nested-section fixture."""
        root = etree.fromstring(xml.encode("utf-8"))
        return JATSDocument.from_root(root)

    def test_nested_sections_are_emitted_in_preorder(self) -> None:
        """A parent and its own text precede child and grandchild sections."""
        document = self._document(
            """
            <article xmlns="http://jats.nlm.nih.gov"><body>
              <p>Leading body paragraph.</p>
              <sec><title>Results</title>
                <p>Parent before child.</p>
                <sec><title>Validation</title>
                  <p>Child text.</p>
                  <sec><title>External</title><p>Grandchild text.</p></sec>
                </sec>
                <p>Parent after child node.</p>
              </sec>
              <p>Trailing body paragraph.</p>
            </body></article>
            """
        )

        sections = extract_sections(document)

        self.assertEqual(
            tuple(section.full_title for section in sections),
            (
                "Untitled Section",
                "Results",
                "Results > Validation",
                "Results > Validation > External",
                "Untitled Section",
            ),
        )
        self.assertEqual(
            tuple(
                tuple(flatten_text(paragraph) for paragraph in section.paragraphs)
                for section in sections
            ),
            (
                ("Leading body paragraph.",),
                ("Parent before child.", "Parent after child node."),
                ("Child text.",),
                ("Grandchild text.",),
                ("Trailing body paragraph.",),
            ),
        )

    def test_missing_body_returns_no_sections(self) -> None:
        """Body is optional and yields an empty immutable result when absent."""
        document = self._document("<article><front/></article>")

        self.assertEqual(extract_sections(document), ())

    def test_caption_text_combines_label_caption_and_citation_marker(self) -> None:
        """Caption flattening retains its label and marker-bearing citation."""
        document = self._document(
            """
            <article><body><fig>
              <label>Fig. 1.</label>
              <caption><p>Result from <xref ref-type="bibr" rid="R1">1</xref>.</p></caption>
            </fig></body><back><ref-list>
              <ref id="R1"><label>1</label></ref>
            </ref-list></back></article>
            """
        )
        figure = document.find(".//j:fig")
        if figure is None:
            self.fail("fixture must contain a figure")

        caption = extract_caption_text(document, figure)

        self.assertEqual(caption, "Fig. 1. Result from [xref:R1].")


if __name__ == "__main__":
    unittest.main()
