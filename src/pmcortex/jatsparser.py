"""Parse PMC JATS XML into retrieval queries and cited documents."""

from pathlib import Path
from typing import Iterable

from lxml import etree
from loguru import logger

from pmcortex.citation_normalizer import (
    CitationDiagnostic,
    CitationForm,
    CitationNormalizer,
    CitationOccurrence,
    NormalizedCitationText,
)
from pmcortex.jats_extraction import (
    JATSDocument,
    _ElementSection,
    build_citation_normalizer,
    extract_article_authors,
    extract_caption_text,
    extract_metadata,
    extract_references,
    extract_sections,
    flatten_text,
    local_name,
    normalize_caption,
    normalize_paragraph_blocks,
    normalize_text,
    sections_to_public,
    select_article_pmcid,
)
from pmcortex.models import (
    CitationCleanupAction,
    Context,
    ContextExtractionResult,
    JATSArticle,
    JATSSection,
    ParsedJATSResult,
    ParserDiagnostic,
    ParserDiagnosticCode,
    PositionedSentence,
    Reference,
    SentencePosition,
)
from pmcortex.sentence_segmenter import SentenceSegmenter

class JATSParseError(ValueError):
    """A technically readable XML document is not a JATS article."""


class _ArticleParseSession:
    """
    Hold temporary mutable state while parsing exactly one JATS article.

    The public :class:`JATSParser` creates a fresh session for every call and
    exposes only the resulting immutable dataclasses.
    """

    def __init__(self) -> None:
        """Initialize the hardened XML parser used for one article.

        Entity resolution, DTD loading, and network access are disabled.
        Comments are removed because they are not article content. Recovery is
        enabled for robust PMC processing and every recovery error is retained
        as a :class:`ParserDiagnostic`. ``huge_tree=True`` relaxes libxml2 size
        and depth limits for unusually large trusted PMC documents.
        """
        self._xml_parser = etree.XMLParser(
            recover=True,
            resolve_entities=False,
            load_dtd=False,
            no_network=True,
            huge_tree=True,
            remove_comments=True,
        )
        self.last_context: str | None = None
        self.parser_diagnostics: tuple[ParserDiagnostic, ...] = ()
        self._sentence_segmenter = SentenceSegmenter()

    def parse(
        self,
        article_path: Path,
        *,
        expected_pmcid: str | None,
        include_sentences: bool,
    ) -> ParsedJATSResult:
        """
        Parse one validated article path into immutable public result models.

        This internal method may use mutable session state. Callers receive no
        reference to the session after the result has been assembled.
        """
        document = self.parse_tree(article_path)
        metadata = extract_metadata(document)
        article_pmcid = select_article_pmcid(
            metadata.pmcid,
            expected_pmcid,
            article_path,
        )
        self.pmcid = article_pmcid
        logger.info(
            "Parsed article {} with title {!r}",
            article_pmcid,
            metadata.title,
        )

        authors = extract_article_authors(document)
        references = extract_references(document)
        element_sections = extract_sections(document)
        logger.info(
            "Extracted {} references and {} sections from {}",
            len(references),
            len(element_sections),
            article_path,
        )

        self.extract_contexts(
            element_sections,
            references,
            save_sentences=include_sentences,
        )
        logger.info(
            "Extracted {} contexts from {}.",
            len(self.contexts),
            article_path,
        )

        sections = sections_to_public(document, element_sections, references)
        article = JATSArticle(
            pmcid=article_pmcid,
            pmid=metadata.pmid,
            title=metadata.title,
            abstract=metadata.abstract,
            authors=authors,
            references=references,
            sections=sections,
        )
        extraction = ContextExtractionResult(
            contexts=tuple(self.contexts),
            sentences=tuple(self.sentences),
            diagnostics=(
                self.parser_diagnostics + tuple(self.citation_diagnostics)
            ),
            unsafe_citation_rejection_count=self.unsafe_citation_rejections,
        )
        return ParsedJATSResult(article=article, extraction=extraction)

    def extract_metadata(
        self,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """Delegate legacy session access to the stateless metadata extractor."""
        metadata = extract_metadata(self.document)
        return metadata.title, metadata.abstract, metadata.pmcid, metadata.pmid

    def extract_authors(self) -> list[str]:
        """Delegate legacy session access to article-author extraction."""
        return list(extract_article_authors(self.document))

    def extract_references(self) -> list[Reference]:
        """Delegate legacy session access to reference extraction."""
        return list(extract_references(self.document))

    def extract_sections(self) -> list[_ElementSection]:
        """Delegate legacy session access to preorder section extraction."""
        return list(extract_sections(self.document))

    def _to_public_sections(
        self,
        sections: tuple[_ElementSection, ...],
        references: tuple[Reference, ...],
    ) -> tuple[JATSSection, ...]:
        """Delegate conversion of XML-bearing sections to public values."""
        return sections_to_public(self.document, sections, references)
    
    def _replace_refs(
        self,
        paragraph: etree._Element,
        references: Iterable[Reference] = (),
    ) -> list[str]:
        """
        Return marker-bearing text blocks for one paragraph-like element.

        This compatibility wrapper delegates citation handling to
        :class:`CitationNormalizer`. Figures and tables remain separate blocks.
        """
        reference_list = list(references)
        normalizer = self._build_citation_normalizer(reference_list)
        normalized_blocks = self._normalize_refs(paragraph, normalizer)
        return [block.text_with_markers for block in normalized_blocks]

    def _normalize_refs(
        self,
        paragraph: etree._Element,
        normalizer: CitationNormalizer,
    ) -> list[NormalizedCitationText]:
        """Delegate paragraph normalization to the extraction component."""
        normalized_blocks = normalize_paragraph_blocks(
            self.document,
            paragraph,
            normalizer,
        )
        return list(normalized_blocks)

    @staticmethod
    def _clean_string(in_string: str) -> str:
        """
        Normalize layout whitespace without removing semantic citation text.

        Citation markers are removed separately by :class:`CitationNormalizer`,
        which preserves grammatical punctuation and author-year expressions.
        """
        normalized = in_string.replace("\n", " ").replace("▪", " ")
        return _ArticleParseSession._normalize_text(normalized)

    def _context_from_sentence(
        self,
        sentence: PositionedSentence,
        normalizer: CitationNormalizer,
        ref_id_to_pmid: dict[str, str],
        unsafe_citation_rids: frozenset[str],
        citation_occurrences: tuple[CitationOccurrence, ...],
        raw_sentence_text: str | None,
    ) -> Context | None:
        """
        Build one retrieval context from a marker-bearing positioned sentence.

        PMID hits are stable-deduplicated because the benchmark represents cited
        documents, not repeated citation occurrences. On success, the query is
        stored as ``self.last_context`` for the next emitted context.
        """
        cited_rids = normalizer.extract_rids(sentence.text)
        has_unsafe_citation = bool(
            set(cited_rids).intersection(unsafe_citation_rids)
        )
        if has_unsafe_citation:
            self.unsafe_citation_rejections += 1
            return None

        pmid_hits = tuple(dict.fromkeys(
            ref_id_to_pmid[rid]
            for rid in cited_rids
            if rid in ref_id_to_pmid
        ))
        if not pmid_hits:
            return None

        clean_query = normalizer.remove_markers(sentence.text)
        if not clean_query:
            return None

        cited_rid_set = set(cited_rids)
        sentence_occurrences = tuple(
            occurrence
            for occurrence in citation_occurrences
            if cited_rid_set.intersection(occurrence.rids)
        )
        citation_forms = tuple(dict.fromkeys(
            occurrence.form.value for occurrence in sentence_occurrences
        ))
        has_parenthetical_author_year = any(
            occurrence.form is CitationForm.PARENTHETICAL_AUTHOR_YEAR
            for occurrence in sentence_occurrences
        )
        cleanup_action = (
            CitationCleanupAction.REMOVED_PARENTHETICAL_CITATION
            if has_parenthetical_author_year
            else CitationCleanupAction.REMOVED_LABEL_CITATION
        )
        raw_query = None
        if raw_sentence_text is not None:
            raw_query = normalizer.remove_markers(raw_sentence_text)

        context = Context(
            position=sentence.position,
            query_raw=raw_query,
            query=clean_query,
            citation_forms=citation_forms,
            citation_cleanup_action=cleanup_action,
            query_length=len(clean_query.split()),
            hits=pmid_hits,
            n_hits=len(pmid_hits),
            context=self.last_context,
        )
        self.last_context = clean_query
        return context


    def extract_contexts(
        self,
        sections: tuple[_ElementSection, ...] | list[_ElementSection],
        references: tuple[Reference, ...] | list[Reference],
        save_sentences: bool = False,
    ) -> None:
        """
        Build reference contexts from parsed sections.

        Populates ``self.contexts`` with positioned :class:`Context` values. If
        ``save_sentences`` is true, marker-bearing full-text sentences are kept
        in ``self.sentences`` at exactly the same structural positions.

        Raises:
            RuntimeError: If no source PMCID has been assigned to the parser.
        """
        if not hasattr(self, "pmcid") or not isinstance(self.pmcid, str):
            raise RuntimeError("Set parser.pmcid before extracting contexts")

        self.contexts: list[Context] = []
        self.sentences: list[PositionedSentence] = []
        self.citation_diagnostics: list[CitationDiagnostic] = []
        self.unsafe_citation_rejections = 0

        normalizer = self._build_citation_normalizer(references)
        ref_id_to_pmid = {
            reference.rid: reference.pmid
            for reference in references
            if reference.rid is not None and reference.pmid is not None
        }
        for s, section in enumerate(sections):
            for p, ref_paragraph in enumerate(section.paragraphs):
                next_sentence_index = 0
                normalized_blocks = self._normalize_refs(ref_paragraph, normalizer)
                for normalized in normalized_blocks:
                    self.citation_diagnostics.extend(normalized.diagnostics)
                    sentence_texts = self._sentence_segmenter.split(
                        normalized.query_text_with_markers
                    )
                    raw_sentence_texts = self._sentence_segmenter.split(
                        normalized.text_with_markers
                    )
                    raw_sentences_align = (
                        len(raw_sentence_texts) == len(sentence_texts)
                    )
                    sentences = [
                        PositionedSentence(
                            position=SentencePosition(
                                source_pmcid=self.pmcid,
                                section_index=s,
                                paragraph_index=p,
                                sentence_index=next_sentence_index + index,
                            ),
                            text=sentence_text,
                        )
                        for index, sentence_text in enumerate(sentence_texts)
                    ]
                    next_sentence_index += len(sentences)
                    if save_sentences:
                        self.sentences.extend(sentences)
                    for block_sentence_index, sentence in enumerate(sentences):
                        raw_sentence_text = None
                        if raw_sentences_align:
                            raw_sentence_text = raw_sentence_texts[
                                block_sentence_index
                            ]
                        context = self._context_from_sentence(
                            sentence,
                            normalizer,
                            ref_id_to_pmid,
                            frozenset(normalized.unsafe_citation_rids),
                            normalized.citation_occurrences,
                            raw_sentence_text,
                        )
                        if context is not None:
                            self.contexts.append(context)
            self.last_context = None

        if self.citation_diagnostics:
            logger.warning(
                "Recovered from {} citation-normalization issues in {}",
                len(self.citation_diagnostics),
                self.pmcid,
            )
        if self.unsafe_citation_rejections:
            logger.warning(
                "Rejected {} queries with unsafe semantic citations in {}",
                self.unsafe_citation_rejections,
                self.pmcid,
            )

    # -----------------------
    # Helpers
    # -----------------------

    def _find_element(
        self,
        xpath: str,
        context: etree._Element | None = None,
    ) -> etree._Element | None:
        """
        Find one element through the document's namespace-aware selector.

        If no context is provided, the search runs from the document root.
        """
        return self.document.find(xpath, context)

    @staticmethod
    def _local_name(element: etree._Element) -> str:
        """Return the namespace-agnostic local name of an XML element."""
        return local_name(element)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Collapse whitespace in a text fragment."""
        return normalize_text(text)

    def _is_bibr_xref(self, element: etree._Element) -> bool:
        """Return True for bibliographic cross-reference nodes."""
        return self._local_name(element) == "xref" and element.get("ref-type") == "bibr"

    def _build_citation_normalizer(
        self,
        references: Iterable[Reference],
    ) -> CitationNormalizer:
        """Delegate construction of the article-scoped citation normalizer."""
        document = getattr(self, "document", None)
        return build_citation_normalizer(document, references)

    def _normalize_caption(
        self,
        element: etree._Element,
        normalizer: CitationNormalizer,
    ) -> NormalizedCitationText:
        """Delegate figure or table caption normalization."""
        return normalize_caption(self.document, element, normalizer)

    def _extract_caption_paragraph(self, element: etree._Element) -> str:
        """Extract one figure/table caption paragraph from label and caption."""
        return extract_caption_text(self.document, element)

    def _flatten_text(self, element: etree._Element | None) -> str | None:
        """
        Flatten an XML element into normalized plain text.

        Uses `itertext()` to include nested nodes and collapses whitespace.
        """
        return flatten_text(element)

    def parse_tree(self, path: str | Path) -> JATSDocument:
        """Parse XML and return a namespace-aware JATS document.

        XML syntax failures that cannot be recovered remain lxml
        ``XMLSyntaxError`` exceptions. A readable XML document whose root is
        not ``article`` raises :class:`JATSParseError`.
        """
        xml = Path(path).read_bytes()
        root = etree.fromstring(xml, parser=self._xml_parser)
        document = JATSDocument.from_root(root)
        if self._local_name(document.root) != "article":
            raise JATSParseError(
                "XML root must be a JATS <article> element, got "
                f"<{self._local_name(document.root)}>"
            )

        self.document = document
        self.parser_diagnostics = self._xml_recovery_diagnostics()
        if self.parser_diagnostics:
            logger.warning(
                "Recovered from {} XML parsing issues in {}",
                len(self.parser_diagnostics),
                path,
            )
        return document

    def _xml_recovery_diagnostics(self) -> tuple[ParserDiagnostic, ...]:
        """Convert the current libxml2 error log into immutable diagnostics."""
        return tuple(
            ParserDiagnostic(
                code=ParserDiagnosticCode.XML_RECOVERY,
                message=entry.message.strip(),
                line=entry.line,
                column=entry.column,
                level=entry.level_name,
                domain=entry.domain_name,
                error_type=entry.type_name,
            )
            for entry in self._xml_parser.error_log
        )


class JATSParser:
    """Stateless public facade for parsing one JATS document per call."""

    __slots__ = ()

    def parse(
        self,
        path: str | Path,
        *,
        expected_pmcid: str | None = None,
        include_sentences: bool = False,
    ) -> ParsedJATSResult:
        """Parse one JATS file into a complete immutable result.

        Args:
            path: Existing JATS XML file.
            expected_pmcid: Optional PMCID expected in the parsed document.
                It is also used when the document contains no PMCID.
            include_sentences: Retain positioned full-text sentences when true.

        Returns:
            Immutable article data, contexts, optional sentences, and
            diagnostics.

        Raises:
            TypeError: If a public argument has the wrong type.
            ValueError: If ``expected_pmcid`` is empty or disagrees with XML.
            FileNotFoundError: If ``path`` does not identify an existing file.
        """
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or pathlib.Path")
        if expected_pmcid is not None and not isinstance(expected_pmcid, str):
            raise TypeError("expected_pmcid must be a string or None")
        if expected_pmcid is not None and not expected_pmcid.strip():
            raise ValueError("expected_pmcid must not be empty")
        if not isinstance(include_sentences, bool):
            raise TypeError("include_sentences must be a bool")

        article_path = Path(path)
        if not article_path.is_file():
            raise FileNotFoundError(f"JATS file does not exist: {article_path}")

        result = _ArticleParseSession().parse(
            article_path,
            expected_pmcid=expected_pmcid,
            include_sentences=include_sentences,
        )
        parsed_pmcid = result.article.pmcid
        pmcid_mismatch = (
            expected_pmcid is not None and parsed_pmcid != expected_pmcid
        )
        if pmcid_mismatch:
            raise ValueError(
                f"JATS PMCID {parsed_pmcid!r} does not match "
                f"expected PMCID {expected_pmcid!r}"
            )
        return result
