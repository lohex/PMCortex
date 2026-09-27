"""Extract typed article content from namespace-aware JATS documents.

The functions in this module are stateless.  Each extraction receives the
document it operates on explicitly, which keeps XML selection independent of
the parser session used to assemble contexts.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import cast

from loguru import logger
from lxml import etree

from pmcortex.citation_normalizer import CitationNormalizer, NormalizedCitationText
from pmcortex.models import JATSSection, Reference


EXTRACTED_CAPTION_TAGS = frozenset({"fig", "table-wrap"})

SKIPPED_CONTENT_TAGS = frozenset({
    "alternatives",
    "chem-struct-wrap",
    "disp-formula",
    "graphic",
    "math",  # Includes namespace-qualified MathML ``math`` elements.
    "media",
    "supplementary-material",
    "tex-math",
})


@dataclass(frozen=True, slots=True)
class JATSDocument:
    """Parsed JATS root and namespace-aware central selector configuration.

    Attributes:
        root: Root ``article`` element of the parsed XML tree.
        namespace_uri: Namespace URI of the root, or ``None`` for unqualified
            JATS XML.
        selector_namespaces: Immutable namespace map used by selectors.
    """

    root: etree._Element
    namespace_uri: str | None
    selector_namespaces: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate the document wrapper and freeze a copy of its namespace map."""
        if not isinstance(self.root, etree._Element):
            raise TypeError("root must be an lxml element")
        if self.namespace_uri is not None and not isinstance(self.namespace_uri, str):
            raise TypeError("namespace_uri must be a string or None")
        if not isinstance(self.selector_namespaces, Mapping):
            raise TypeError("selector_namespaces must be a mapping")
        if not all(
            isinstance(prefix, str) and isinstance(uri, str)
            for prefix, uri in self.selector_namespaces.items()
        ):
            raise TypeError("selector_namespaces must map strings to strings")
        immutable_namespaces = MappingProxyType(dict(self.selector_namespaces))
        object.__setattr__(self, "selector_namespaces", immutable_namespaces)

    @classmethod
    def from_root(cls, root: etree._Element) -> "JATSDocument":
        """Create a selector-ready document from an XML root element.

        Args:
            root: Parsed XML root element.

        Returns:
            A namespace-aware immutable document wrapper.

        Raises:
            TypeError: If ``root`` is not an lxml element.
        """
        if not isinstance(root, etree._Element):
            raise TypeError("root must be an lxml element")

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
        self._validate_selector_context(selector, context)
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
        self._validate_selector_context(selector, context)
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
        self._validate_selector_context(selector, context)
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
        self._validate_selector_context(selector, context)
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

    @staticmethod
    def _validate_selector_context(
        selector: str,
        context: etree._Element | None,
    ) -> None:
        """Validate selector arguments shared by all document query methods."""
        if not isinstance(selector, str):
            raise TypeError("selector must be a string")
        if not selector:
            raise ValueError("selector must not be empty")
        if context is not None and not isinstance(context, etree._Element):
            raise TypeError("context must be an lxml element or None")


@dataclass(frozen=True, slots=True)
class ExtractedArticleMetadata:
    """Core metadata read from one JATS ``article-meta`` element."""

    title: str | None
    abstract: str | None
    pmcid: str | None
    pmid: str | None

    def __post_init__(self) -> None:
        """Validate optional metadata strings returned by JATS extraction."""
        for name, value in (
            ("title", self.title),
            ("abstract", self.abstract),
            ("pmcid", self.pmcid),
            ("pmid", self.pmid),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")


@dataclass(frozen=True, slots=True)
class _ElementSection:
    """Internal section whose paragraph elements remain available downstream."""

    title: str
    full_title: str
    paragraphs: tuple[etree._Element, ...]


def normalize_text(text: str) -> str:
    """Collapse layout whitespace and normalize spacing around punctuation.

    Args:
        text: Plain-text fragment to normalize.

    Returns:
        A single-line normalized string.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([\(\[\{])\s+", r"\1", normalized)
    return re.sub(r"\s+([\)\]\}])", r"\1", normalized)


def local_name(element: etree._Element) -> str:
    """Return the namespace-independent local name of an XML element.

    Args:
        element: XML element whose tag name is required.

    Returns:
        Namespace-free element name.

    Raises:
        TypeError: If ``element`` is not an lxml element.
    """
    if not isinstance(element, etree._Element):
        raise TypeError("element must be an lxml element")
    return etree.QName(element).localname


def flatten_text(element: etree._Element | None) -> str | None:
    """Flatten one optional XML subtree into normalized visible text.

    Args:
        element: XML element, or ``None`` when the selected field is absent.

    Returns:
        Normalized descendant text or ``None`` for an absent element.

    Raises:
        TypeError: If a non-element value other than ``None`` is supplied.
    """
    if element is None:
        return None
    if not isinstance(element, etree._Element):
        raise TypeError("element must be an lxml element or None")
    text = " ".join(part.strip() for part in element.itertext() if part.strip())
    return re.sub(r"\s+", " ", text).strip()


def extract_metadata(document: JATSDocument) -> ExtractedArticleMetadata:
    """Extract title, abstract, PMCID, and PMID from one JATS document.

    Args:
        document: Namespace-aware JATS document.

    Returns:
        Immutable article metadata. Missing JATS fields are represented by
        ``None``. Numeric PMC identifiers are normalized with a ``PMC`` prefix.

    Raises:
        TypeError: If ``document`` is not a :class:`JATSDocument`.
    """
    _validate_document(document)
    title = flatten_text(
        document.find(".//j:article-meta//j:title-group//j:article-title")
    )
    abstract = flatten_text(document.find(".//j:article-meta//j:abstract"))
    article_ids = document.findall(".//j:article-meta/j:article-id")
    pmid_elements = tuple(
        element
        for element in article_ids
        if element.get("pub-id-type") == "pmid"
    )
    pmid = pmid_elements[0].text if pmid_elements else None

    pmcid = flatten_text(
        document.find(".//j:article-meta//j:article-id[@pub-id-type='pmcid']")
    )
    if pmcid is not None and not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    return ExtractedArticleMetadata(title, abstract, pmcid, pmid)


def select_article_pmcid(
    extracted_pmcid: str | None,
    expected_pmcid: str | None,
    article_path: Path,
) -> str:
    """Select the source PMCID using XML, expected value, then filename.

    Args:
        extracted_pmcid: PMCID found in JATS metadata, if present.
        expected_pmcid: PMCID supplied by the caller, if present.
        article_path: Source path whose stem is the final fallback.

    Returns:
        The effective article identifier.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If a supplied identifier is empty.
    """
    for name, value in (
        ("extracted_pmcid", extracted_pmcid),
        ("expected_pmcid", expected_pmcid),
    ):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} must be a string or None")
        if value is not None and not value.strip():
            raise ValueError(f"{name} must not be empty")
    if not isinstance(article_path, Path):
        raise TypeError("article_path must be a pathlib.Path")

    if extracted_pmcid is not None:
        return extracted_pmcid
    if expected_pmcid is not None:
        return expected_pmcid
    return article_path.stem


def extract_article_authors(document: JATSDocument) -> tuple[str, ...]:
    """Extract article authors, preferring structured JATS names.

    Args:
        document: Namespace-aware JATS document.

    Returns:
        Authors in source order, normalized as ``surname initials``.

    Raises:
        TypeError: If ``document`` has the wrong type.
    """
    _validate_document(document)
    names = document.findall(
        ".//j:article-meta//j:contrib-group//"
        "j:contrib[@contrib-type='author']//j:name"
    )
    authors = tuple(
        author
        for name in names
        if (author := _extract_author_name(document, name))
    )
    if authors:
        return authors

    string_names = document.findall(
        ".//j:article-meta//j:contrib-group//"
        "j:contrib[@contrib-type='author']//j:string-name"
    )
    return tuple(
        author
        for name in string_names
        if (author := _extract_author_name(document, name))
    )


def extract_references(document: JATSDocument) -> tuple[Reference, ...]:
    """Extract bibliography entries from JATS element or mixed citations.

    Args:
        document: Namespace-aware JATS document.

    Returns:
        Parsed references in source order. References without a supported
        citation body are skipped.

    Raises:
        TypeError: If ``document`` has the wrong type.
    """
    _validate_document(document)
    references: list[Reference] = []
    for ref in document.findall(".//j:back//j:ref-list//j:ref"):
        citation = document.find(".//j:element-citation", ref)
        if citation is None:
            citation = document.find(".//j:mixed-citation", ref)
        if citation is None:
            continue

        pmcid = _extract_pub_id(document, citation, "pmcid")
        if pmcid is not None and not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        year_element = document.find(".//j:year", citation)
        if year_element is None:
            year_element = document.find(".//j:date", citation)
        year_text = flatten_text(year_element)

        title_element = document.find(".//j:article-title", citation)
        if title_element is None:
            title_element = document.find(".//j:chapter-title", citation)

        references.append(
            Reference(
                rid=ref.get("id"),
                label=flatten_text(document.find(".//j:label", ref)),
                doi=_extract_pub_id(document, citation, "doi"),
                pmid=_extract_pub_id(document, citation, "pmid"),
                pmcid=pmcid,
                year=(
                    int(year_text)
                    if year_text is not None and year_text.isdigit()
                    else None
                ),
                title=flatten_text(title_element),
                journal=flatten_text(document.find(".//j:source", citation)),
                authors=_extract_reference_authors(document, citation),
            )
        )
    return tuple(references)


def extract_sections(document: JATSDocument) -> tuple[_ElementSection, ...]:
    """Extract body sections in parent-before-child preorder.

    Direct body paragraphs before or after named sections are grouped into
    separate ``Untitled Section`` values. A named parent section is emitted
    before its nested sections, with direct paragraphs only.

    Args:
        document: Namespace-aware JATS document.

    Returns:
        XML-bearing internal sections in deterministic reading order.

    Raises:
        TypeError: If ``document`` has the wrong type.
    """
    _validate_document(document)
    body = document.find(".//j:body")
    if body is None:
        return ()

    sections: list[_ElementSection] = []
    body_paragraphs: list[etree._Element] = []

    def flush_body_paragraphs_inplace() -> None:
        """Append accumulated direct-body paragraphs and clear the buffer."""
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
        child_name = local_name(child)
        if child_name == "sec":
            flush_body_paragraphs_inplace()
            sections.extend(_extract_nested_sections(document, child, ()))
        elif child_name == "p" or child_name in EXTRACTED_CAPTION_TAGS:
            body_paragraphs.append(child)

    flush_body_paragraphs_inplace()
    return tuple(sections)


def build_citation_normalizer(
    document: JATSDocument | None,
    references: Iterable[Reference],
) -> CitationNormalizer:
    """Build an article-scoped normalizer from reference order and labels.

    Args:
        document: Namespace-aware source JATS document. ``None`` is accepted
            for isolated context tests and uses parsed references only.
        references: Parsed references used as a fallback for skipped raw refs.

    Returns:
        Citation normalizer configured for the article.

    Raises:
        TypeError: If ``document`` or a reference has the wrong type.
    """
    if document is not None:
        _validate_document(document)
    reference_values = tuple(references)
    if not all(isinstance(reference, Reference) for reference in reference_values):
        raise TypeError("references must contain only Reference values")

    reference_order: list[str] = []
    reference_labels: dict[str, str | None] = {}
    if document is not None:
        for ref_element in document.findall(".//j:back//j:ref-list//j:ref"):
            rid = ref_element.get("id")
            if not rid:
                continue
            reference_order.append(rid)
            label_element = document.find("./j:label", ref_element)
            label = None
            if label_element is not None:
                label = normalize_text("".join(label_element.itertext()))
            reference_labels[rid] = label

    for reference in reference_values:
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


def normalize_paragraph_blocks(
    document: JATSDocument,
    paragraph: etree._Element,
    normalizer: CitationNormalizer,
) -> tuple[NormalizedCitationText, ...]:
    """Normalize a paragraph-like node while retaining caption block order.

    Args:
        document: Namespace-aware source JATS document.
        paragraph: Paragraph, figure, or table wrapper to process.
        normalizer: Article-scoped citation normalizer.

    Returns:
        Normalized, non-empty text blocks in document order.

    Raises:
        TypeError: If an argument has the wrong type.
    """
    _validate_document(document)
    if not isinstance(paragraph, etree._Element):
        raise TypeError("paragraph must be an lxml element")
    if not isinstance(normalizer, CitationNormalizer):
        raise TypeError("normalizer must be a CitationNormalizer")

    element_name = local_name(paragraph)
    if element_name in SKIPPED_CONTENT_TAGS:
        logger.debug("Skipping JATS node {}", element_name)
        return ()
    if element_name in EXTRACTED_CAPTION_TAGS:
        caption = normalize_caption(document, paragraph, normalizer)
        return (caption,) if caption.text_with_markers else ()

    normalized_blocks: list[NormalizedCitationText] = []
    current_fragments: list[str | etree._Element] = []

    def flush_current_inplace() -> None:
        """Normalize accumulated mixed content and clear the local buffer."""
        if not current_fragments:
            return
        normalized = normalizer.normalize_fragments(current_fragments)
        current_fragments.clear()
        if normalized.text_with_markers:
            normalized_blocks.append(normalized)

    if paragraph.text:
        current_fragments.append(paragraph.text)
    for node in paragraph:
        node_name = local_name(node)
        if node_name in EXTRACTED_CAPTION_TAGS:
            flush_current_inplace()
            caption = normalize_caption(document, node, normalizer)
            if caption.text_with_markers:
                normalized_blocks.append(caption)
        elif node_name in SKIPPED_CONTENT_TAGS:
            logger.debug("Skipping JATS node {}", node_name)
        else:
            current_fragments.append(node)

        if node.tail:
            current_fragments.append(node.tail)

    flush_current_inplace()
    return tuple(normalized_blocks)


def sections_to_public(
    document: JATSDocument,
    sections: Iterable[_ElementSection],
    references: Iterable[Reference],
) -> tuple[JATSSection, ...]:
    """Convert XML-bearing sections into immutable public section values.

    Args:
        document: Namespace-aware source JATS document.
        sections: Internal sections with paragraph-like XML elements.
        references: Article references needed for citation markers.

    Returns:
        Public sections containing marker-bearing source blocks.

    Raises:
        TypeError: If a collection contains an unexpected value.
    """
    _validate_document(document)
    section_values = tuple(sections)
    if not all(isinstance(section, _ElementSection) for section in section_values):
        raise TypeError("sections must contain only _ElementSection values")
    reference_values = tuple(references)
    normalizer = build_citation_normalizer(document, reference_values)
    return tuple(
        JATSSection(
            title=section.title,
            full_title=section.full_title,
            source_blocks=tuple(
                " ".join(
                    normalized.text_with_markers
                    for normalized in normalize_paragraph_blocks(
                        document,
                        paragraph,
                        normalizer,
                    )
                )
                for paragraph in section.paragraphs
            ),
        )
        for section in section_values
    )


def extract_caption_text(
    document: JATSDocument,
    element: etree._Element,
) -> str:
    """Extract marker-bearing label and caption text from a figure or table.

    Args:
        document: Namespace-aware source JATS document.
        element: Figure or table wrapper.

    Returns:
        Flattened marker-bearing caption text.
    """
    _validate_document(document)
    if not isinstance(element, etree._Element):
        raise TypeError("element must be an lxml element")
    normalizer = build_citation_normalizer(document, ())
    return normalize_caption(document, element, normalizer).text_with_markers


def _validate_document(document: JATSDocument) -> None:
    """Validate the shared document argument at public extraction boundaries."""
    if not isinstance(document, JATSDocument):
        raise TypeError("document must be a JATSDocument")


def _extract_author_name(
    document: JATSDocument,
    element: etree._Element,
) -> str:
    """Extract and normalize one structured or free-form author name."""
    surname_text = document.findtext("j:surname", element)
    given_text = document.findtext("j:given-names", element)
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
    return normalize_text("".join(element.itertext()))


def _extract_reference_authors(
    document: JATSDocument,
    citation: etree._Element,
) -> tuple[str, ...]:
    """Extract structured reference authors with a string-name fallback."""
    authors = tuple(
        author
        for name in document.findall(".//j:name", citation)
        if (author := _extract_author_name(document, name))
    )
    if authors:
        return authors

    string_authors: list[str] = []
    for string_name in document.findall(".//j:string-name", citation):
        author = _extract_author_name(document, string_name)
        string_authors.extend(
            part.strip() for part in author.split(",") if part.strip()
        )
    return tuple(string_authors)


def _extract_pub_id(
    document: JATSDocument,
    node: etree._Element,
    pub_id_type: str,
) -> str | None:
    """Extract a publication ID, including the legacy DOI text fallback."""
    for element in document.findall(".//j:pub-id", node):
        element_type = (element.get("pub-id-type") or "").lower()
        if element_type == pub_id_type.lower():
            text = (element.text or "").strip()
            if text:
                return text

    if pub_id_type.lower() != "doi":
        return None
    text = flatten_text(node) or ""
    match = re.search(r"\b10\.\d{4,9}/\S+\b", text)
    return match.group(0).rstrip(").,;") if match is not None else None


def _extract_nested_sections(
    document: JATSDocument,
    parent: etree._Element,
    parent_titles: tuple[str, ...],
) -> tuple[_ElementSection, ...]:
    """Recursively emit one section before its child sections."""
    title = flatten_text(document.find("./j:title", parent)) or "Untitled Section"
    titles = (*parent_titles, title)
    paragraphs = tuple(
        child
        for child in parent
        if local_name(child) == "p"
        or local_name(child) in EXTRACTED_CAPTION_TAGS
    )
    section = _ElementSection(
        title=title,
        full_title=" > ".join(titles),
        paragraphs=paragraphs,
    )
    child_sections = tuple(
        child
        for child in parent
        if local_name(child) == "sec"
    )
    descendants = tuple(
        nested
        for child in child_sections
        for nested in _extract_nested_sections(document, child, titles)
    )
    return (section, *descendants)


def normalize_caption(
    document: JATSDocument,
    element: etree._Element,
    normalizer: CitationNormalizer,
) -> NormalizedCitationText:
    """Normalize a figure or table label and caption as one text block.

    Args:
        document: Namespace-aware source JATS document.
        element: Figure or table wrapper.
        normalizer: Article-scoped citation normalizer.

    Returns:
        One normalized caption block, which may contain empty text.

    Raises:
        TypeError: If an argument has the wrong type.
    """
    _validate_document(document)
    if not isinstance(element, etree._Element):
        raise TypeError("element must be an lxml element")
    if not isinstance(normalizer, CitationNormalizer):
        raise TypeError("normalizer must be a CitationNormalizer")
    fragments: list[str | etree._Element] = []
    label_element = document.find("./j:label", element)
    caption_element = document.find("./j:caption", element)
    if label_element is not None:
        fragments.append(label_element)
    if label_element is not None and caption_element is not None:
        fragments.append(" ")
    if caption_element is not None:
        fragments.append(caption_element)
    return normalizer.normalize_fragments(fragments)
