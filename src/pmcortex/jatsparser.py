"""Parse PMC JATS XML into retrieval queries and cited documents."""

from dataclasses import dataclass
import re
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, cast

from lxml import etree
from loguru import logger

from pmcortex.citation_normalizer import (
    CitationDiagnostic,
    CitationForm,
    CitationNormalizer,
    CitationOccurrence,
    NormalizedCitationText,
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

ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "etc.",
    "dr.",
    "prof.",
    "ph.d.",
    "fig.",
    "u.s.",
    "vs.",
}

EXTRACTED_CAPTION_TAGS = frozenset({
    "fig",
    "table-wrap",
})

SKIPPED_CONTENT_TAGS = frozenset({
    "alternatives",
    "chem-struct-wrap",
    "disp-formula",
    "graphic",
    "math",  # includes mml:math
    "media",
    "supplementary-material",
    "tex-math",
})


class JATSParseError(ValueError):
    """A technically readable XML document is not a JATS article."""


@dataclass(frozen=True, slots=True)
class JATSDocument:
    """Parsed JATS root and namespace-aware central selector configuration."""

    root: etree._Element
    namespace_uri: str | None
    selector_namespaces: Mapping[str, str]

    @classmethod
    def from_root(cls, root: etree._Element) -> "JATSDocument":
        """Create a selector-ready document from an XML root element."""
        namespace_uri = cast(str | None, etree.QName(root).namespace)
        namespace_values = (
            {"j": namespace_uri} if namespace_uri is not None else {}
        )
        return cls(
            root=root,
            namespace_uri=namespace_uri,
            selector_namespaces=MappingProxyType(namespace_values),
        )

    def find(
        self,
        selector: str,
        context: etree._Element | None = None,
    ) -> etree._Element | None:
        """Find the first matching element with or without a JATS namespace."""
        search_root = self.root if context is None else context
        effective_selector = self._effective_selector(selector)
        return search_root.find(
            effective_selector,
            namespaces=dict(self.selector_namespaces),
        )

    def findall(
        self,
        selector: str,
        context: etree._Element | None = None,
    ) -> tuple[etree._Element, ...]:
        """Find all matching elements with or without a JATS namespace."""
        search_root = self.root if context is None else context
        effective_selector = self._effective_selector(selector)
        return tuple(
            search_root.findall(
                effective_selector,
                namespaces=dict(self.selector_namespaces),
            )
        )

    def findtext(
        self,
        selector: str,
        context: etree._Element | None = None,
    ) -> str | None:
        """Return direct text from the first namespace-compatible element."""
        search_root = self.root if context is None else context
        effective_selector = self._effective_selector(selector)
        return search_root.findtext(
            effective_selector,
            namespaces=dict(self.selector_namespaces),
        )

    def xpath(
        self,
        selector: str,
        context: etree._Element | None = None,
    ) -> tuple[etree._Element, ...]:
        """Evaluate an element-returning XPath with central namespace handling."""
        search_root = self.root if context is None else context
        effective_selector = self._effective_selector(selector)
        raw_results = cast(
            list[object],
            search_root.xpath(
                effective_selector,
                namespaces=dict(self.selector_namespaces),
            ),
        )
        if not all(isinstance(result, etree._Element) for result in raw_results):
            raise TypeError("JATSDocument.xpath supports only element results")
        return tuple(cast(etree._Element, result) for result in raw_results)

    def _effective_selector(self, selector: str) -> str:
        """Remove the internal prefix only for namespace-free documents."""
        if self.namespace_uri is None:
            return selector.replace("j:", "")
        return selector


@dataclass(frozen=True, slots=True)
class _ElementSection:
    """Internal section whose paragraph elements remain available for extraction."""

    title: str
    full_title: str
    paragraphs: tuple[etree._Element, ...]


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
        self.parse_tree(article_path)
        title, abstract, extracted_pmcid, pmid = self.extract_metadata()
        article_pmcid = extracted_pmcid
        if article_pmcid is None:
            article_pmcid = (
                expected_pmcid
                if expected_pmcid is not None
                else article_path.stem
            )
        self.pmcid = article_pmcid
        logger.info("Parsed article {} with title {!r}", article_pmcid, title)

        authors = tuple(self.extract_authors())
        references = tuple(self.extract_references())
        element_sections = tuple(self.extract_sections())
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

        sections = self._to_public_sections(element_sections, references)
        article = JATSArticle(
            pmcid=article_pmcid,
            pmid=pmid,
            title=title,
            abstract=abstract,
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
        """
        Extract core article-level metadata from the parsed JATS tree.

        Reads the article title, abstract, and PMCID from `<article-meta>`.
        If a PubMed identifier is present, it is also stored on `self.pmid`.
        PMCID values are normalized to start with the `PMC` prefix.

        Returns:
            A tuple containing `(title, abstract, pmcid, pmid)`.
        """
        title_obj = self._find_element(".//j:article-meta//j:title-group//j:article-title")
        title = self._flatten_text(title_obj)

        abstract_obj = self._find_element(".//j:article-meta//j:abstract")
        abstract = self._flatten_text(abstract_obj)

        article_ids = self.document.findall(
            ".//j:article-meta/j:article-id"
        )
        pmid_article_ids = [
            element
            for element in article_ids
            if element.get("pub-id-type") == "pmid"
        ]
        pmid = pmid_article_ids[0].text if pmid_article_ids else None
            
        pmcid_obj = self._find_element(".//j:article-meta//j:article-id[@pub-id-type='pmcid']")
        pmcid = self._flatten_text(pmcid_obj)
        if pmcid and not pmcid.upper().startswith("PMC"):
            pmcid = "PMC" + pmcid

        return title, abstract, pmcid, pmid

    # -----------------------
    # Author extraction
    # -----------------------
    def extract_authors(self) -> list[str]:
        """
        Extract author names from the article metadata.

        Reads `<name>` entries first and falls back to `<string-name>`.
        """
        authors: list[str] = []
        names = self.document.findall(
            ".//j:article-meta//j:contrib-group//"
            "j:contrib[@contrib-type='author']//j:name"
        )
        for name in names:
            author = self._extract_author_name(name)
            if author:
                authors.append(author)

        if not authors:
            string_names = self.document.findall(
                ".//j:article-meta//j:contrib-group//"
                "j:contrib[@contrib-type='author']//j:string-name"
            )
            for string_name in string_names:
                author = self._extract_author_name(string_name)
                if author:
                    authors.append(author)

        return authors

    # -----------------------
    # References extraction
    # -----------------------

    def extract_references(self) -> list[Reference]:
        """
        Extract all <ref> items from <ref-list>.
        Handles common patterns:
          <ref id="R1"><element-citation>...</element-citation></ref>
          <ref id="R1"><mixed-citation>...</mixed-citation></ref>
        """
        references: list[Reference] = []
        for ref in self.document.findall(".//j:back//j:ref-list//j:ref"):
            rid = ref.get("id")

            label_obj = self._find_element(".//j:label", ref)
            label = self._flatten_text(label_obj)

            cit = self.document.find(".//j:element-citation", ref)
            if cit is None:
                cit = self.document.find(".//j:mixed-citation", ref)
            if cit is None:
                continue

            doi = self._extract_pub_id(cit, "doi")
            pmid = self._extract_pub_id(cit, "pmid")
            pmcid = self._extract_pub_id(cit, "pmcid")
            if pmcid and not pmcid.upper().startswith("PMC"):
                pmcid = "PMC" + pmcid

            year_obj = self._find_element(".//j:year", cit)
            if year_obj is None:
                year_obj = self._find_element(".//j:date", cit)
            year = self._flatten_text(year_obj)

            title_obj = self._find_element(".//j:article-title", cit)
            if title_obj is None:
                title_obj = self._find_element(".//j:chapter-title", cit)
            title = self._flatten_text(title_obj)

            journal_obj = self._find_element(".//j:source", cit)
            journal = self._flatten_text(journal_obj)

            authors = self._extract_authors(cit)

            references.append(
                Reference(
                    rid=rid,
                    label=label,
                    doi=doi,
                    pmid=pmid,
                    pmcid=pmcid,
                    year=int(year) if year and year.isdigit() else None,
                    title=title,
                    journal=journal,
                    authors=tuple(authors),
                )
            )
         
        return references
    
        
    def _extract_authors(self, cit: etree._Element) -> list[str]:
        """
        Extract author names from a citation node.

        Reads `<name>` entries first and falls back to `<string-name>`.
        """
        authors: list[str] = []
        for name in self.document.findall(".//j:name", cit):
            author = self._extract_author_name(name)
            if author:
                authors.append(author)

        if not authors:
            for string_name in self.document.findall(".//j:string-name", cit):
                author = self._extract_author_name(string_name)
                if author:
                    authors.extend(
                        part.strip() for part in author.split(",") if part.strip()
                    )

        return authors

    def _extract_author_name(self, element: etree._Element) -> str:
        """Extract and normalize one structured or string author name."""
        surname_text = self.document.findtext("j:surname", element)
        given_text = self.document.findtext("j:given-names", element)
        surname = "" if surname_text is None else surname_text.strip()
        given = "" if given_text is None else given_text.strip()

        if given and given.upper() != given:
            given = "".join(
                part[0].upper()
                for part in re.findall(r"[A-Za-z]+", given)
                if part
            )

        structured_name = " ".join(part for part in (surname, given) if part)
        if structured_name:
            return structured_name

        return self._normalize_text("".join(element.itertext()))

    def _extract_pub_id(self, node: etree._Element, pub_id_type: str) -> str | None:
        """
        Extract one publication identifier by type from a citation subtree.

        For DOI, a regex fallback is applied when no `<pub-id>` match exists.
        """
        if node is None:
            return None
        
        # <pub-id pub-id-type="doi">10.1234/...</pub-id>
        for element in self.document.findall(".//j:pub-id", node):
            if (element.get("pub-id-type") or "").lower() == pub_id_type.lower():
                text = (element.text or "").strip()
                if text:
                    return text
                
        # sometimes DOI appears in ext-link
        if pub_id_type.lower() == "doi":
            text = self._flatten_text(node) or ""
            match = re.search(r"\b10\.\d{4,9}/\S+\b", text)
            if match:
                return match.group(0).rstrip(").,;")
            
        return None

    # -----------------------
    # Sections extraction
    # -----------------------

    def extract_sections(self) -> list[_ElementSection]:
        """
        Alternative to extract_sections that keeps paragraphs separate.
        """
        sections: list[_ElementSection] = []
        body = self._find_element(".//j:body")
        if body is None:
            return sections

        body_paragraphs: list[etree._Element] = []

        def flush_body_paragraphs() -> None:
            """Append and clear paragraphs found directly under the body."""
            if not body_paragraphs:
                return
            sections.append(
                _ElementSection(
                    title="Untitled Section",
                    full_title="Untitled Section",
                    paragraphs=tuple(body_paragraphs),
                )
            )
            body_paragraphs.clear()

        for child in body:
            child_name = self._local_name(child)
            if child_name == "sec":
                flush_body_paragraphs()
                subsections = self._extract_sections_paragraphs(child)
                sections.extend(subsections)
            elif child_name == "p" or child_name in EXTRACTED_CAPTION_TAGS:
                body_paragraphs.append(child)

        flush_body_paragraphs()

        return sections

    def _extract_sections_paragraphs(
        self,
        parent: etree._Element,
        parent_titles: list[str] | None = None,
    ) -> list[_ElementSection]:
        """
        Recursively collect section paragraphs and hierarchical titles.

        Returns typed internal sections with paragraphs and hierarchical titles.
        """
        title_obj = self._find_element("./j:title", parent)
        title = self._flatten_text(title_obj) or "Untitled Section"
        paragraphs: list[etree._Element] = []
        sections: list[_ElementSection] = []
        if parent_titles is None:
            parent_titles = []

        for child in parent:
            child_name = self._local_name(child)
            if child_name == "p":
                paragraphs.append(child)
            elif child_name in EXTRACTED_CAPTION_TAGS:
                paragraphs.append(child)
            elif child_name == "sec":
                # Recursively extract subsections
                subsections = self._extract_sections_paragraphs(
                    child,
                    parent_titles + [title],
                )
                sections.extend(subsections)

        sections.append(
            _ElementSection(
                paragraphs=tuple(paragraphs),
                title=title,
                full_title=" > ".join(parent_titles + [title]),
            )
        )

        return sections

    def _to_public_sections(
        self,
        sections: tuple[_ElementSection, ...],
        references: tuple[Reference, ...],
    ) -> tuple[JATSSection, ...]:
        """Convert XML-bearing internal sections into immutable public values."""
        normalizer = self._build_citation_normalizer(references)
        public_sections: list[JATSSection] = []
        for section in sections:
            source_blocks = tuple(
                " ".join(
                    normalized.text_with_markers
                    for normalized in self._normalize_refs(paragraph, normalizer)
                )
                for paragraph in section.paragraphs
            )
            public_sections.append(
                JATSSection(
                    title=section.title,
                    full_title=section.full_title,
                    source_blocks=source_blocks,
                )
            )
        return tuple(public_sections)
    
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
        """Normalize one paragraph while preserving caption block ordering."""
        element_name = self._local_name(paragraph)
        if element_name in SKIPPED_CONTENT_TAGS:
            logger.debug(f"Skipping JATS node {element_name}")
            return []

        if element_name in EXTRACTED_CAPTION_TAGS:
            caption = self._normalize_caption(paragraph, normalizer)
            return [caption] if caption.text_with_markers else []

        normalized_blocks: list[NormalizedCitationText] = []
        current_fragments: list[str | etree._Element] = []

        def flush_current() -> None:
            """Normalize and clear the currently accumulated mixed content."""
            if not current_fragments:
                return
            normalized = normalizer.normalize_fragments(current_fragments)
            current_fragments.clear()
            if normalized.text_with_markers:
                normalized_blocks.append(normalized)

        if paragraph.text:
            current_fragments.append(paragraph.text)

        for node in paragraph:
            node_name = self._local_name(node)
            if node_name in EXTRACTED_CAPTION_TAGS:
                flush_current()
                caption = self._normalize_caption(node, normalizer)
                if caption.text_with_markers:
                    normalized_blocks.append(caption)
            elif node_name in SKIPPED_CONTENT_TAGS:
                logger.debug(f"Skipping JATS node {node_name}")
            else:
                current_fragments.append(node)

            if node.tail:
                current_fragments.append(node.tail)

        flush_current()
        return normalized_blocks

    @staticmethod
    def _clean_string(in_string: str) -> str:
        """
        Normalize layout whitespace without removing semantic citation text.

        Citation markers are removed separately by :class:`CitationNormalizer`,
        which preserves grammatical punctuation and author-year expressions.
        """
        normalized = in_string.replace("\n", " ").replace("▪", " ")
        return _ArticleParseSession._normalize_text(normalized)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split at sentence-final periods while retaining trailing citations."""
        sentences = []
        start = 0
        paren_depth = 0
        i = 0

        while i < len(text):
            ch = text[i]

            if ch in "([":
                paren_depth += 1
            elif ch in ")]":
                paren_depth = max(0, paren_depth - 1)

            all_closed = ch == "." and paren_depth == 0
            # allow splits at full stops after a closing bracket, e.g. "This is a sentence (with a comment (and a nested but unresolved). This is another sentence."
            just_closed = ch == "." and i > 0 and text[i-1] in "])"
            if just_closed and not all_closed:
                logger.info(f"Splitting at a full stop preceded by a closing bracket at position {i} in text: '{text[max(0, i-30):i+30]}'")
            if all_closed or just_closed:
                paren_depth = 0
                prefix = text[max(0, i - 10):i + 1].lower()

                is_abbrev = any(prefix.endswith(abbr) for abbr in ABBREVIATIONS)
                is_initial = text[i-1:i].isupper() and text[i-2:i-1].isspace() if i >= 2 else False

                sentence_end = i + 1
                j = sentence_end
                while j < len(text) and text[j].isspace():
                    j += 1
                marker_starts_next_sentence = False
                marker_match = re.match(
                    (
                        r"\[xref:[^\]]+\]"
                        r"(?:\s*(?:[,;/&–—−-]|\band\b|\bto\b)\s*"
                        r"\[xref:[^\]]+\])*"
                    ),
                    text[j:],
                    flags=re.IGNORECASE,
                )
                if marker_match is not None:
                    marker_end = j + marker_match.end()
                    period_follows_marker = re.search(
                        r"\[xref:[^\]]+\]\s*$",
                        text[:i],
                    ) is not None
                    if period_follows_marker:
                        marker_starts_next_sentence = True
                    else:
                        sentence_end = marker_end
                        j = sentence_end
                        while j < len(text) and text[j].isspace():
                            j += 1

                has_trailing_space = j > sentence_end

                is_parenthetical_et_al_continuation = (
                    prefix.endswith("et al.")
                    and j < len(text)
                    and text[j] == "("
                )

                sentence_start = j
                while sentence_start < len(text) and text[sentence_start] in "\"'“‘([":
                    sentence_start += 1

                next_starts_sentence = (
                    marker_starts_next_sentence
                    or (
                        sentence_start < len(text)
                        and (
                            text[sentence_start].isupper()
                            or text[sentence_start].isdigit()
                        )
                    )
                )

                if (
                    has_trailing_space
                    and not (
                        is_abbrev
                        or is_parenthetical_et_al_continuation
                    )
                    and not is_initial
                    and next_starts_sentence
                ):
                    sentences.append(text[start:sentence_end].strip())
                    start = j
                    i = j
                    continue

            i += 1

        tail = text[start:].strip()
        if tail:
            sentences.append(tail)

        return sentences

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
                    sentence_texts = self._split_sentences(
                        normalized.query_text_with_markers
                    )
                    raw_sentence_texts = self._split_sentences(
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
        return etree.QName(element).localname

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Collapse whitespace in a text fragment."""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([\(\[\{])\s+", r"\1", text)
        return re.sub(r"\s+([\)\]\}])", r"\1", text)

    def _is_bibr_xref(self, element: etree._Element) -> bool:
        """Return True for bibliographic cross-reference nodes."""
        return self._local_name(element) == "xref" and element.get("ref-type") == "bibr"

    def _build_citation_normalizer(
        self,
        references: Iterable[Reference],
    ) -> CitationNormalizer:
        """
        Build an article-scoped normalizer from JATS ref order and visible labels.

        Raw ``<ref>`` nodes are used first because ``extract_references`` may skip
        uncommon citation encodings that can still occur inside an xref range.
        """
        reference_order: list[str] = []
        reference_labels: dict[str, str | None] = {}

        if hasattr(self, "document"):
            ref_elements = self.document.findall(
                ".//j:back//j:ref-list//j:ref"
            )
            for ref_element in ref_elements:
                rid = ref_element.get("id")
                if not rid:
                    continue
                reference_order.append(rid)
                label_element = self.document.find("./j:label", ref_element)
                label: str | None = None
                if label_element is not None:
                    label = self._normalize_text(
                        "".join(label_element.itertext())
                    )
                reference_labels[rid] = label

        for reference in references:
            if reference.rid is None:
                continue
            reference_order.append(reference.rid)
            if reference.rid not in reference_labels:
                reference_labels[reference.rid] = reference.label

        return CitationNormalizer(
            reference_order,
            reference_labels,
            skipped_tags=SKIPPED_CONTENT_TAGS | EXTRACTED_CAPTION_TAGS,
        )

    def _normalize_caption(
        self,
        element: etree._Element,
        normalizer: CitationNormalizer,
    ) -> NormalizedCitationText:
        """Normalize a figure or table label and caption as one text block."""
        fragments: list[str | etree._Element] = []
        label_element = self._find_element("./j:label", element)
        caption_element = self._find_element("./j:caption", element)
        if label_element is not None:
            fragments.append(label_element)
        if label_element is not None and caption_element is not None:
            fragments.append(" ")
        if caption_element is not None:
            fragments.append(caption_element)
        return normalizer.normalize_fragments(fragments)

    def _extract_caption_paragraph(self, element: etree._Element) -> str:
        """Extract one figure/table caption paragraph from label and caption."""
        normalizer = self._build_citation_normalizer(())
        return self._normalize_caption(element, normalizer).text_with_markers

    def _flatten_text(self, element: etree._Element | None) -> str | None:
        """
        Flatten an XML element into normalized plain text.

        Uses `itertext()` to include nested nodes and collapses whitespace.
        """
        if element is None:
            return None
        
        # itertext collects nested tags
        text = " ".join(s.strip() for s in element.itertext() if s and s.strip())
        text = re.sub(r"\s+", " ", text).strip()
        return text

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
