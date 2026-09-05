"""Characterize the pre-refactoring JATS parser behavior and artifact schema."""

from dataclasses import FrozenInstanceError, asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
import unittest
from unittest.mock import patch

from lxml import etree

from pmcortex.citation_normalizer import (
    CitationForm,
    CitationOccurrence,
    NormalizedCitationText,
)
from pmcortex.jatsparser import JATSParser, _ArticleParseSession, _ElementSection
from pmcortex.models import ParsedJATSResult, Reference
from pmcortex.pipeline import CONTEXT_COLUMNS, METADATA_COLUMNS, SOURCE_COLUMNS
from pmcortex.serialization import (
    author_lists_to_payload,
    normalize_author_lists,
)
from tests._shared import existing_example_files


type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


EXPECTED_DIR = Path(__file__).resolve().parent / "fixtures" / "expected"


def _as_json_object(value: object) -> JsonObject:
    """Convert a JSON-compatible object into a precisely typed dictionary."""
    serialized = json.dumps(value, ensure_ascii=False)
    decoded = cast(JsonValue, json.loads(serialized))
    if not isinstance(decoded, dict):
        raise TypeError("expected a JSON object")
    return decoded


def _load_expected_snapshot(pmcid: str) -> JsonObject:
    """Load one checked-in parser snapshot by PMCID."""
    snapshot_path = EXPECTED_DIR / f"{pmcid}.json"
    decoded = cast(JsonValue, json.loads(snapshot_path.read_text(encoding="utf-8")))
    if not isinstance(decoded, dict):
        raise TypeError(f"expected a JSON object in {snapshot_path}")
    return decoded


def _snapshot_parser_output(
    result: ParsedJATSResult,
) -> JsonObject:
    """Build the stable pre-refactoring snapshot for one parsed article."""
    article = result.article
    payload = {
        "metadata": {
            "pmcid": article.pmcid,
            "pmid": article.pmid,
            "title": article.title,
            "abstract": article.abstract,
            "authors": article.authors,
        },
        "references": [asdict(reference) for reference in article.references],
        "sections": [
            {
                "title": section.title,
                "full_title": section.full_title,
                "paragraph_count": len(section.source_blocks),
            }
            for section in article.sections
        ],
        "contexts": [
            {
                "source_pmcid": context.position.source_pmcid,
                "section_index": context.position.section_index,
                "paragraph_index": context.position.paragraph_index,
                "sentence_index": context.position.sentence_index,
                "query_raw": context.query_raw,
                "query": context.query,
                "citation_forms": context.citation_forms,
                "citation_cleanup_action": context.citation_cleanup_action.value,
                "hits": context.hits,
                "query_length": context.query_length,
                "n_hits": context.n_hits,
                "context": context.context,
            }
            for context in result.contexts
        ],
        "sentences": [
            {
                "section_index": sentence.position.section_index,
                "paragraph_index": sentence.position.paragraph_index,
                "sentence_index": sentence.position.sentence_index,
                "text": sentence.text,
            }
            for sentence in result.sentences
        ],
        "normalized_authors_and_refs": author_lists_to_payload(
            normalize_author_lists(article)
        ),
        "citation_diagnostics": [
            {
                "code": diagnostic.code.value,
                "rids": list(diagnostic.rids),
                "source_line": diagnostic.source_line,
            }
            for diagnostic in result.diagnostics
        ],
        "unsafe_citation_rejection_count": (
            result.unsafe_citation_rejection_count
        ),
    }
    return _as_json_object(payload)


class TestJATSParserGoldenBaseline(unittest.TestCase):
    """Freeze representative parser outputs before the breaking refactor."""

    def test_all_example_articles_match_checked_in_snapshots(self) -> None:
        """Every available example must match its complete golden snapshot."""
        examples = existing_example_files()
        snapshot_pmcids = {
            snapshot_path.stem for snapshot_path in EXPECTED_DIR.glob("PMC*.json")
        }
        self.assertEqual(snapshot_pmcids, set(examples))

        for pmcid, article_path in examples.items():
            with self.subTest(pmcid=pmcid):
                result = JATSParser().parse(
                    article_path,
                    expected_pmcid=pmcid,
                    include_sentences=True,
                )

                actual = _snapshot_parser_output(result)
                expected = _load_expected_snapshot(pmcid)

                self.assertEqual(actual, expected)

    def test_reusing_parser_replaces_article_scoped_output(self) -> None:
        """A second parse must not retain outputs belonging to the first article."""
        examples = existing_example_files()
        parser = JATSParser()
        parser.parse(
            examples["PMC3438321"],
            expected_pmcid="PMC3438321",
            include_sentences=True,
        )

        second_result = parser.parse(
            examples["PMC2693326"],
            expected_pmcid="PMC2693326",
            include_sentences=True,
        )

        actual = _snapshot_parser_output(second_result)
        expected = _load_expected_snapshot("PMC2693326")
        self.assertEqual(actual, expected)


class TestJATSParserResultAPI(unittest.TestCase):
    """Verify the new stateless facade and immutable result contract."""

    def test_result_collections_and_models_are_immutable(self) -> None:
        """Public collections are tuples and frozen models reject mutation."""
        pmcid, path = next(iter(existing_example_files().items()))

        result = JATSParser().parse(
            path,
            expected_pmcid=pmcid,
            include_sentences=True,
        )

        self.assertIsInstance(result.article.authors, tuple)
        self.assertIsInstance(result.article.references, tuple)
        self.assertIsInstance(result.article.sections, tuple)
        self.assertIsInstance(result.contexts, tuple)
        self.assertIsInstance(result.sentences, tuple)
        self.assertIsInstance(result.diagnostics, tuple)
        with self.assertRaises(FrozenInstanceError):
            setattr(result.article, "title", "changed")

    def test_public_parse_validates_all_arguments(self) -> None:
        """Every public argument rejects wrong types and invalid values."""
        pmcid, path = next(iter(existing_example_files().items()))
        parser = JATSParser()

        with self.assertRaises(TypeError):
            parser.parse(123)  # type: ignore[arg-type]
        with self.assertRaises(FileNotFoundError):
            parser.parse(path.parent / "missing.nxml")
        with self.assertRaises(TypeError):
            parser.parse(path, expected_pmcid=123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            parser.parse(path, expected_pmcid=" ")
        with self.assertRaises(TypeError):
            parser.parse(path, include_sentences=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            parser.parse(path, expected_pmcid=f"{pmcid}_MISMATCH")

    def test_public_parser_does_not_retain_result_state(self) -> None:
        """The facade has no mutable instance dictionary or result attributes."""
        parser = JATSParser()
        pmcid, path = next(iter(existing_example_files().items()))

        parser.parse(path, expected_pmcid=pmcid)

        self.assertFalse(hasattr(parser, "__dict__"))
        for name in ("article", "contexts", "sentences", "citation_diagnostics"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(parser, name))


class TestJATSParserKnownBaselineLimitations(unittest.TestCase):
    """Record limitations that later refactoring phases intentionally change."""

    @staticmethod
    def _write_xml(directory: Path, filename: str, xml: str) -> Path:
        """Write one temporary XML document and return its path."""
        path = directory / filename
        path.write_text(xml, encoding="utf-8")
        return path

    def test_parse_tree_requires_a_default_namespace(self) -> None:
        """The current parser rejects prefixed and namespace-free JATS roots."""
        documents = {
            "prefixed.nxml": (
                '<j:article xmlns:j="http://jats.nlm.nih.gov">'
                "<j:body/></j:article>"
            ),
            "namespace-free.nxml": "<article><body/></article>",
        }
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for filename, xml in documents.items():
                with self.subTest(filename=filename):
                    parser = _ArticleParseSession()
                    path = self._write_xml(directory, filename, xml)
                    with self.assertRaises(KeyError):
                        parser.parse_tree(path)

    def test_question_and_exclamation_marks_are_not_sentence_boundaries(self) -> None:
        """Record that question and exclamation marks are not boundaries yet."""
        text = "Is this supported? Yes it is! Final statement."

        sentences = _ArticleParseSession._split_sentences(text)

        self.assertEqual(sentences, [text])

    def test_alignment_mismatch_sets_query_raw_to_none(self) -> None:
        """The current parser keeps a context but drops raw text after misalignment."""
        parser = _ArticleParseSession()
        parser.pmcid = "PMC_TEST"
        paragraph = etree.fromstring(b"<p/>")
        sections = [
            _ElementSection(
                title="Untitled Section",
                full_title="Untitled Section",
                paragraphs=(paragraph,),
            )
        ]
        references = [
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
            )
        ]
        normalized = NormalizedCitationText(
            text_with_markers="Claim. Next [xref:R1].",
            query_text_with_markers="Claim [xref:R1].",
            query_text="Claim.",
            cited_rids=("R1",),
            citation_occurrences=(
                CitationOccurrence(
                    form=CitationForm.LABEL_ONLY,
                    display_text="1",
                    rids=("R1",),
                    source_line=None,
                ),
            ),
            diagnostics=(),
        )

        with patch.object(parser, "_normalize_refs", return_value=[normalized]):
            parser.extract_contexts(sections, references)

        self.assertEqual(len(parser.contexts), 1)
        self.assertIsNone(parser.contexts[0].query_raw)

    def test_state_dependent_methods_are_absent_from_public_parser(self) -> None:
        """The breaking facade exposes no state-dependent result accessors."""
        method_names = (
            "extract_metadata",
            "extract_authors",
            "extract_references",
            "extract_sections",
            "sources_to_dataframe",
            "contexts_to_dataframe",
            "text_to_list",
            "normalized_authors_and_refs",
        )
        for method_name in method_names:
            with self.subTest(method=method_name):
                self.assertFalse(hasattr(JATSParser(), method_name))

    def test_extract_contexts_requires_an_assigned_pmcid(self) -> None:
        """Direct context extraction reports its one explicit state precondition."""
        parser = _ArticleParseSession()

        with self.assertRaisesRegex(RuntimeError, "Set parser.pmcid"):
            parser.extract_contexts([], [])


class TestCurrentArtifactSchema(unittest.TestCase):
    """Freeze the current pipeline CSV column order before schema changes."""

    def test_metadata_columns(self) -> None:
        """Metadata shards must retain their current columns and order."""
        self.assertEqual(
            METADATA_COLUMNS,
            ("pmcid", "pmid", "title", "abstract", "authors"),
        )

    def test_source_columns(self) -> None:
        """Source shards must retain their current columns and order."""
        self.assertEqual(
            SOURCE_COLUMNS,
            (
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
            ),
        )

    def test_context_columns(self) -> None:
        """Context shards must retain their current columns and order."""
        self.assertEqual(
            CONTEXT_COLUMNS,
            (
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
            ),
        )


if __name__ == "__main__":
    unittest.main()
