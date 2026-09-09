"""Tests for hardened JATS XML parsing and namespace-independent selection."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lxml import etree

from pmcortex.jatsparser import (
    JATSDocument,
    JATSParseError,
    JATSParser,
    _ArticleParseSession,
)
from pmcortex.models import ParserDiagnostic, ParserDiagnosticCode


NAMESPACE_FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "jats_namespaces"
)


class TestJATSDocumentNamespaces(unittest.TestCase):
    """Verify equivalent extraction from all supported namespace variants."""

    def test_namespace_variants_produce_identical_results(self) -> None:
        """Default, prefixed, and namespace-free JATS must parse identically."""
        paths = tuple(sorted(NAMESPACE_FIXTURE_DIR.glob("*.nxml")))
        self.assertEqual(len(paths), 3)

        results = tuple(
            JATSParser().parse(
                path,
                expected_pmcid="PMC9000001",
                include_sentences=True,
            )
            for path in paths
        )

        self.assertTrue(all(result == results[0] for result in results[1:]))
        self.assertEqual(results[0].article.authors, ("Example A",))
        self.assertEqual(results[0].article.references[0].authors, ("Cited B",))
        self.assertEqual(results[0].contexts[0].hits, ("12345678",))

    def test_parse_tree_returns_document_with_working_selectors(self) -> None:
        """The parse-tree contract exposes central find/findall/xpath helpers."""
        expected_namespaces = {
            "default.nxml": "http://jats.nlm.nih.gov",
            "prefixed.nxml": "http://jats.nlm.nih.gov",
            "namespace_free.nxml": None,
        }
        for filename, namespace_uri in expected_namespaces.items():
            with self.subTest(filename=filename):
                session = _ArticleParseSession()
                document = session.parse_tree(NAMESPACE_FIXTURE_DIR / filename)

                self.assertIsInstance(document, JATSDocument)
                self.assertEqual(document.namespace_uri, namespace_uri)
                self.assertIsNotNone(document.find(".//j:article-meta"))
                self.assertEqual(
                    len(document.findall(".//j:back//j:ref-list//j:ref")),
                    1,
                )
                self.assertEqual(
                    len(document.xpath(".//j:ref[@id='R1']")),
                    1,
                )


class TestJATSXMLFailuresAndRecovery(unittest.TestCase):
    """Distinguish recoverable XML, technical failures, and invalid JATS."""

    @staticmethod
    def _write_xml(directory: Path, filename: str, content: str) -> Path:
        """Write one isolated XML test input."""
        path = directory / filename
        path.write_text(content, encoding="utf-8")
        return path

    def test_recovered_xml_produces_a_parser_diagnostic(self) -> None:
        """A repaired tree remains usable and exposes the libxml2 error data."""
        malformed_xml = (
            "<article><front><article-meta>"
            '<article-id pub-id-type="pmcid">PMC9000002</article-id>'
            "<title-group><article-title>Recovered</article-title></title-group>"
            "</article-meta></front><body><sec><p>Broken paragraph"
            "</sec></body></article>"
        )
        with TemporaryDirectory() as temporary_directory:
            path = self._write_xml(
                Path(temporary_directory),
                "PMC9000002.nxml",
                malformed_xml,
            )

            result = JATSParser().parse(
                path,
                expected_pmcid="PMC9000002",
            )

        parser_diagnostics = tuple(
            diagnostic
            for diagnostic in result.diagnostics
            if isinstance(diagnostic, ParserDiagnostic)
        )
        self.assertGreaterEqual(len(parser_diagnostics), 1)
        self.assertTrue(
            all(
                diagnostic.code is ParserDiagnosticCode.XML_RECOVERY
                for diagnostic in parser_diagnostics
            )
        )
        self.assertTrue(all(diagnostic.message for diagnostic in parser_diagnostics))
        self.assertEqual(result.article.title, "Recovered")

    def test_empty_xml_keeps_the_standard_syntax_exception(self) -> None:
        """A document without a recoverable root remains a technical error."""
        with TemporaryDirectory() as temporary_directory:
            path = self._write_xml(
                Path(temporary_directory),
                "empty.nxml",
                "",
            )

            with self.assertRaises(etree.XMLSyntaxError):
                JATSParser().parse(path)

    def test_readable_non_jats_xml_raises_domain_error(self) -> None:
        """A readable XML root other than article is fachlich invalid."""
        with TemporaryDirectory() as temporary_directory:
            path = self._write_xml(
                Path(temporary_directory),
                "not-jats.nxml",
                "<catalog><record/></catalog>",
            )

            with self.assertRaisesRegex(JATSParseError, "JATS <article>"):
                JATSParser().parse(path)

    def test_missing_body_returns_empty_content_collections(self) -> None:
        """Article metadata remains useful when a document has no body."""
        xml = (
            "<article><front><article-meta>"
            '<article-id pub-id-type="pmcid">PMC9000003</article-id>'
            "<title-group><article-title>No body</article-title></title-group>"
            "</article-meta></front></article>"
        )
        with TemporaryDirectory() as temporary_directory:
            path = self._write_xml(
                Path(temporary_directory),
                "PMC9000003.nxml",
                xml,
            )
            result = JATSParser().parse(path, expected_pmcid="PMC9000003")

        self.assertEqual(result.article.title, "No body")
        self.assertEqual(result.article.sections, ())
        self.assertEqual(result.contexts, ())

    def test_missing_article_meta_uses_expected_pmcid(self) -> None:
        """Body extraction works without optional article metadata."""
        xml = "<article><body><sec><p>Body only.</p></sec></body></article>"
        with TemporaryDirectory() as temporary_directory:
            path = self._write_xml(
                Path(temporary_directory),
                "PMC9000004.nxml",
                xml,
            )
            result = JATSParser().parse(
                path,
                expected_pmcid="PMC9000004",
                include_sentences=True,
            )

        self.assertEqual(result.article.pmcid, "PMC9000004")
        self.assertIsNone(result.article.title)
        self.assertEqual(len(result.article.sections), 1)
        self.assertEqual(len(result.sentences), 1)


if __name__ == "__main__":
    unittest.main()
