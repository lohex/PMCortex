"""Parse PMC JATS XML into retrieval queries and cited documents."""

from dataclasses import asdict
import re
from pathlib import Path
from typing import Iterable

from lxml import etree
import pandas as pd
from loguru import logger

from pmcortex.citation_normalizer import (
    CitationDiagnostic,
    CitationForm,
    CitationNormalizer,
    CitationOccurrence,
    NormalizedCitationText,
)
from pmcortex.models import Context, JATSArticle, PositionedSentence, Reference

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

class JATSParser:
    """
    Parses PMC JATS (NISO JATS) .nxml files and extracts structured content.

    Keep this class pure:
    - no network
    - no file crawling logic
    - only parse + extract
    """

    def __init__(self) -> None:
        """Initialize the XML parser configuration used for all parses."""
        self._xml_parser = etree.XMLParser(
            recover=True,
            resolve_entities=False,
            huge_tree=True,
            remove_comments=True,
        )
        self.last_context = None

    def parse_article(
        self,
        path: str | Path,
        *,
        pmcid: str | None = None,
        save_sentences: bool = False,
    ) -> JATSArticle:
        """
        Parse one article file and return a structured `JATSArticle`.

        Besides the returned dataclass, this method also populates parser state:
        `self.root`, `self.namespaces` and `self.contexts`.
        """
        article_path = Path(path)
        if not article_path.is_file():
            raise FileNotFoundError(f"JATS article not found: {article_path}")

        self.parse_tree(article_path)
        title, abstract, extracted_pmcid, pmid = self.extract_metadata()
        article_pmcid = extracted_pmcid
        if article_pmcid is None:
            article_pmcid = pmcid if pmcid is not None else article_path.stem
        self.pmcid = article_pmcid
        logger.info(f"Parsed article {article_pmcid} with title '{title}'")

        authors = self.extract_authors()        
        references = self.extract_references()
        sections = self.extract_sections()
        logger.info(f"Extracted {len(references)} references and {len(sections)} sections from {path}")

        self.extract_contexts(
            sections,
            references,
            save_sentences=save_sentences,
        )
        logger.info(f"Extracted {len(self.contexts)} contexts from {path}.")

        self.article = JATSArticle(
            pmcid=article_pmcid,
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            references=references,
            sections=sections,
        )

        return self.article
    
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

        pmid_obj = self.root.findall(".//j:article-meta/j:article-id", namespaces=self.namespaces)
        pmid_article_ids = [o for o in pmid_obj if o.get('pub-id-type') == 'pmid']
        pmid = pmid_article_ids[0].text if pmid_article_ids else None
            
        pmcid_obj = self._find_element(".//j:article-meta//j:article-id[@pub-id-type='pmcid']")
        pmcid = self._flatten_text(pmcid_obj)
        if pmcid and not pmcid.upper().startswith("PMC"):
            pmcid = "PMC" + pmcid

        return title, abstract, pmcid, pmid

    def metadata_to_dataframe(self) -> pd.DataFrame:
        """Return the current article metadata as a one-row DataFrame."""
        metadata = asdict(self.article)
        metadata.pop("references")
        metadata.pop("sections")
        return pd.DataFrame([metadata])

    # -----------------------
    # Author extraction
    # -----------------------
    def extract_authors(self) -> list[str]:
        """
        Extract author names from the article metadata.

        Reads `<name>` entries first and falls back to `<string-name>`.
        """
        authors: list[str] = []
        for name in self.root.findall(".//j:article-meta//j:contrib-group//j:contrib[@contrib-type='author']//j:name", namespaces=self.namespaces):
            author = self._extract_author_name(name)
            if author:
                authors.append(author)

        if not authors:
            for sn in self.root.findall(".//j:article-meta//j:contrib-group//j:contrib[@contrib-type='author']//j:string-name", namespaces=self.namespaces):
                author = self._extract_author_name(sn)
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
        for ref in self.root.findall(".//j:back//j:ref-list//j:ref", namespaces=self.namespaces):
            rid = ref.get("id")

            label_obj = self._find_element(".//j:label", ref)
            label = self._flatten_text(label_obj)

            cit = ref.find(".//j:element-citation", namespaces=self.namespaces)
            if cit is None:
                cit = ref.find(".//j:mixed-citation", namespaces=self.namespaces)
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
                    authors=authors,
                )
            )
         
        return references
    
        
    def sources_to_dataframe(self) -> pd.DataFrame:
        """Convert extracted references to a pandas DataFrame."""
        return pd.DataFrame([asdict(reference) for reference in self.article.references])


    def _extract_authors(self, cit: etree._Element) -> list[str]:
        """
        Extract author names from a citation node.

        Reads `<name>` entries first and falls back to `<string-name>`.
        """
        authors: list[str] = []
        for name in cit.findall(".//j:name", namespaces=self.namespaces):
            author = self._extract_author_name(name)
            if author:
                authors.append(author)

        if not authors:
            for sn in cit.findall(".//j:string-name", namespaces=self.namespaces):
                author = self._extract_author_name(sn)
                if author:
                    authors.extend(
                        part.strip() for part in author.split(",") if part.strip()
                    )

        return authors

    def _extract_author_name(self, element: etree._Element) -> str:
        """Extract and normalize one structured or string author name."""
        surname_text = element.findtext("j:surname", namespaces=self.namespaces)
        given_text = element.findtext("j:given-names", namespaces=self.namespaces)
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
        for element in node.findall(".//j:pub-id", namespaces=self.namespaces):
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

    @staticmethod
    def _short_author_name(name: str) -> str:
        """Convert `Surname Given` into `Surname G`."""
        parts = [part for part in name.split() if part]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return f"{parts[0]} {parts[1][0]}"

    @staticmethod
    def _coerce_author_names(authors: str | Iterable[str]) -> list[str]:
        """Normalize reference authors into a list of non-empty names."""
        if isinstance(authors, str):
            return [name.strip() for name in authors.split(",") if name.strip()]
        return [str(name).strip() for name in authors if str(name).strip()]


    def normalized_authors_and_refs(self) -> dict:
        """
        Save the notebook-style `authors` and `cited_authors` lists to YAML.
        """
        payload = {
            "authors": [self._short_author_name(name) for name in self.article.authors],
            "cited_authors": [
                [self._short_author_name(name) for name in self._coerce_author_names(reference.authors)]
                for reference in self.article.references
            ],
        }
        return payload

    # -----------------------
    # Sections extraction
    # -----------------------

    def extract_sections(self) -> list[dict]:
        """
        Alternative to extract_sections that keeps paragraphs separate.
        """
        sections: list[dict] = []
        body = self._find_element(".//j:body")
        if body is None:
            return sections

        body_paragraphs: list[etree._Element] = []

        def flush_body_paragraphs() -> None:
            """Append and clear paragraphs found directly under the body."""
            if not body_paragraphs:
                return
            sections.append({
                "paragraphs": list(body_paragraphs),
                "title": "Untitled Section",
                "full_title": "Untitled Section",
            })
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

    def _extract_sections_paragraphs(self,
                                     parent: etree._Element,
                                     parent_titles: list[str] = None,
                                     ) -> list[dict]:
        """
        Recursively collect section paragraphs and hierarchical titles.

        Returns one or more section dictionaries with `paragraphs`, `title`,
        and `full_title`.
        """
        title_obj = self._find_element("./j:title", parent)
        title = self._flatten_text(title_obj) or "Untitled Section"
        paragraphs = []
        sections = []
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
                subsections = self._extract_sections_paragraphs(child, parent_titles + [title])
                sections.extend(subsections)

        sections.append({
            'paragraphs': paragraphs,
            'title': title,
            'full_title': " > ".join(parent_titles + [title])
        })

        return sections
    
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
        return JATSParser._normalize_text(normalized)

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

        pmid_hits = list(dict.fromkeys(
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
        citation_forms = list(dict.fromkeys(
            occurrence.form.value for occurrence in sentence_occurrences
        ))
        has_parenthetical_author_year = any(
            occurrence.form is CitationForm.PARENTHETICAL_AUTHOR_YEAR
            for occurrence in sentence_occurrences
        )
        cleanup_action = (
            "removed_parenthetical_citation"
            if has_parenthetical_author_year
            else "removed_label_citation"
        )
        raw_query = None
        if raw_sentence_text is not None:
            raw_query = normalizer.remove_markers(raw_sentence_text)

        context = Context(
            source_pmcid=self.pmcid,
            section_index=sentence.section_index,
            paragraph_index=sentence.paragraph_index,
            sentence_index=sentence.sentence_index,
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
        sections: list[dict[str, object]],
        references: list[Reference],
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
            paragraphs = section["paragraphs"]
            if not isinstance(paragraphs, list):
                continue
            for p, ref_paragraph in enumerate(paragraphs):
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
                            section_index=s,
                            paragraph_index=p,
                            sentence_index=next_sentence_index + index,
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

    def contexts_to_dataframe(self) -> pd.DataFrame:
        """
        Convert extracted contexts into a tabular representation.

        Returns a DataFrame with query text, PMID hits, and basic query stats.
        """
        return pd.DataFrame([asdict(context) for context in self.contexts])

    def text_to_list(self) -> str:
        """Return positioned full-text sentences, one tab-separated line each."""
        lines = [
            (
                f"{sentence.section_index}/"
                f"{sentence.paragraph_index}/"
                f"{sentence.sentence_index}\t"
                f"{sentence.text}"
            )
            for sentence in self.sentences
        ]
        return '\n'.join(lines)


    # -----------------------
    # Helpers
    # -----------------------

    def _find_element(self, xpath: str, context: etree._Element = None) -> etree._Element | None:
        """
        Find a single element using the parser's JATS namespace mapping.

        If no context is provided, the search runs from `self.root`.
        """
        if context is None:
            context = self.root

        element = context.find(xpath, namespaces=self.namespaces)
        return element

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

        if hasattr(self, "root"):
            ref_elements = self.root.xpath(
                ".//*[local-name()='back']//*[local-name()='ref-list']"
                "//*[local-name()='ref'][@id]"
            )
            for ref_element in ref_elements:
                rid = ref_element.get("id")
                if not rid:
                    continue
                reference_order.append(rid)
                label_elements = ref_element.xpath("./*[local-name()='label']")
                label = None
                if label_elements:
                    label = self._normalize_text(
                        "".join(label_elements[0].itertext())
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

    def _flatten_text(self, element: etree._Element) -> str | None:
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

    def parse_tree(self, path: str | Path) -> etree._Element:
        """
        Parse a JATS XML file and initialize parser state.

        Sets `self.root` to the parsed XML root element and `self.namespaces`
        to the default JATS namespace mapping used in XPath queries.
        """
        xml = Path(path).read_bytes()
        self.root = etree.fromstring(xml, parser=self._xml_parser)
        self.namespaces = {"j": self.root.nsmap[None]}
