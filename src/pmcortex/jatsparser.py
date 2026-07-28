import sys
import re
from pathlib import Path
from typing import Iterable
from lxml import etree
import pandas as pd
import yaml
from loguru import logger
from pmcortex.models import JATSArticle, Reference, Context

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
            self, path: str | Path,
            *,
            save_sentences: bool = False,
        ) -> JATSArticle:
        """
        Parse one article file and return a structured `JATSArticle`.

        Besides the returned dataclass, this method also populates parser state:
        `self.root`, `self.namespaces` and `self.contexts`.
        """
        self.parse_tree(path)
        title, abstract, pmcid, pmid = self.extract_metadata()
        if pmcid is None:
            pmcid = path.replace('.nxml', '')
        self.pmcid = pmcid
        logger.info(f"Parsed article {pmcid} with title '{title}'")

        authors = self.extract_authors()        
        references = self.extract_references()
        sections = self.extract_sections()
        logger.info(f"Extracted {len(references)} references and {len(sections)} sections from {path}")

        self.extract_contexts(
            sections,
            references,
            pmcid,
            save_sentences=save_sentences
        )
        logger.info(f"Extracted {len(self.contexts)} contexts from {path}.")

        self.article = JATSArticle(
            pmcid=pmcid,
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            references=references,
            sections=sections,
        )

        return self.article
    
    def extract_metadata(self):
        """
        Extract core article-level metadata from the parsed JATS tree.

        Reads the article title, abstract, and PMCID from `<article-meta>`.
        If a PubMed identifier is present, it is also stored on `self.pmid`.
        PMCID values are normalized to start with the `PMC` prefix.

        Returns:
            tuple[str | None, str | None, str | None]: A tuple containing
            `(title, abstract, pmcid)`.
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

    def metadata_to_dataframe(self):
        df = pd.DataFrame([{
            name: getattr(self.article, name)
            for name in dir(self.article)
            if not name.startswith('__') and name not in ['references', 'sections']
        }])
        return df

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
            surname = (name.findtext("j:surname", namespaces=self.namespaces) or "").strip()
            given = (name.findtext("j:given-names", namespaces=self.namespaces) or "").strip()
            if given and given.upper() != given:  # if given is not already initials:
                given = "".join(
                    part[0].upper()
                    for part in re.findall(r"[A-Za-z]+", given)
                    if part
                )

            full = " ".join(x for x in [surname, given] if x).strip()
            if full:
                authors.append(full)

        # Sometimes authors are in <string-name>
        if not authors:
            for sn in self.root.findall(".//j:article-meta//j:contrib-group//j:contrib[@contrib-type='author']//j:string-name", namespaces=self.namespaces):
                text = (sn.text or "").strip()
                if text:
                    authors.append(text)

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
        
        df = pd.DataFrame([
            {name: getattr(a, name) for name in dir(a) if not name.startswith('__')}
            for a in self.article.references
        ])
        return df


    def _extract_authors(self, cit: etree._Element) -> list[str]:
        """
        Extract author names from a citation node.

        Reads `<name>` entries first and falls back to `<string-name>`.
        """
        authors: list[str] = []
        for name in cit.findall(".//j:name", namespaces=self.namespaces):
            surname = (name.findtext("j:surname", namespaces=self.namespaces) or "").strip()
            given = (name.findtext("j:given-names", namespaces=self.namespaces) or "").strip()
            
            if given and given.upper() != given:  # if given is not already initials
                given = "".join(
                    part[0].upper()
                    for part in re.findall(r"[A-Za-z]+", given)
                    if part
                )

            full = " ".join(x for x in [surname, given] if x).strip()
            if full:
                authors.append(full)

        # Sometimes authors are in <string-name>
        if not authors:
            for sn in cit.findall(".//j:string-name", namespaces=self.namespaces):
                surname = (sn.findtext("j:surname", namespaces=self.namespaces) or "").strip()
                given = (sn.findtext("j:given-names", namespaces=self.namespaces) or "").strip()
                if given and given.upper() != given:  # if given is not already initials
                    given = "".join(
                        part[0].upper()
                        for part in re.findall(r"[A-Za-z]+", given)
                        if part
                    )

                if not surname and not given:
                    text_names = ''.join(list(sn.itertext()))
                    text_names = re.sub(r'\s+', ' ', text_names).strip()
                    if text_names:
                        authors.append(text_names)

        authors = ', '.join(authors)
        return authors

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
    
    def _replace_refs(self, paragraph: etree._Element) -> list[str]:
        """
        Flatten one paragraph-like element into ordered text paragraphs.

        Bibliographic references are converted to placeholders like
        `[xref:R12]`. Figures and tables are emitted as separate paragraphs
        containing their label and caption.
        """
        element_name = self._local_name(paragraph)
        if element_name in SKIPPED_CONTENT_TAGS:
            logger.debug(f"Skipping JATS node {element_name}")
            return []

        if element_name in EXTRACTED_CAPTION_TAGS:
            caption_paragraph = self._extract_caption_paragraph(paragraph)
            return [caption_paragraph] if caption_paragraph else []

        paragraphs: list[str] = []
        current_parts: list[str] = []

        def flush_current() -> None:
            text = self._normalize_text("".join(current_parts))
            current_parts.clear()
            if text:
                paragraphs.append(text)

        if paragraph.text:
            current_parts.append(paragraph.text)

        for node in paragraph:
            node_name = self._local_name(node)
            tail_text = node.tail

            if self._is_bibr_xref(node):
                ref_id = node.get("rid")
                if ref_id:
                    inner_html = node.text or ""
                    if re.fullmatch(r'(?:\[[0-9]{1,3}[A-Za-z]?\]|[0-9]{1,3}[A-Za-z]?)', inner_html.strip()):
                        inner_html = ''
                    has_unclosed_parenthesis = "(" in inner_html and ")" not in inner_html
                    closing_parenthesis_is_in_tail = (
                        tail_text is not None and ")" in tail_text
                    )
                    if has_unclosed_parenthesis and closing_parenthesis_is_in_tail:
                        before, after = tail_text.split(")", maxsplit=1)
                        inner_html += before + ")"
                        tail_text = after
                    current_parts.append(f" {inner_html} [xref:{ref_id}] ")
            elif node_name in EXTRACTED_CAPTION_TAGS:
                flush_current()
                caption_paragraph = self._extract_caption_paragraph(node)
                if caption_paragraph:
                    paragraphs.append(caption_paragraph)
            elif node_name in SKIPPED_CONTENT_TAGS:
                logger.debug(f"Skipping JATS node {node_name}")
            else:
                current_parts.append(self._serialize_text_with_refs(node))

            if tail_text:
                current_parts.append(tail_text)

        flush_current()
        return paragraphs

    @staticmethod
    def _replace_ref_ranges(in_string):
        """
        Expand xref ranges into explicit xref markers.

        Example: `[xref:R1] – [xref:R3]` becomes
        `[xref:R1], [xref:R2], [xref:R3]`.
        """
        pattern = re.compile(r'\[xref:(\w+)(\d+)\]\s*(?:[–-]|to)\s*\[xref:(\w+)(\d+)\]')
        while pattern.findall(in_string):
            ref = pattern.search(in_string)
            start_pos, end_pos = ref.span()
            match_str = in_string[start_pos: end_pos]
            base, min_ref, _, max_ref = ref.groups()
            min_ref, max_ref = int(min_ref), int(max_ref)
            ref_range = [f'[xref:{base}{i}]' for i in range(min_ref, max_ref+1)]
            in_string = in_string.replace(match_str, ', '.join(ref_range))

        return in_string

    #@staticmethod
    def _clean_string(self, in_string: str):
        """
        Split text into sentence-like chunks and normalize whitespace/punctuation.

        Returns a list of cleaned strings ending with a period.
        """
        out_string = in_string.replace('\n', ' ').replace('▪', '')
        out_string = re.sub(r' +', ' ', out_string)
        out_string = re.sub(r' et al\.(,?\s*\[xref:[^\]]+\])', '\\1', out_string)
        out_string = re.sub(r'\s*\(\s*[^\[\(\]\)]+\s*(\[xref:[^\]]+\])', '(\\1', out_string)
        # Remove whitespace and unify seperators between multiple refs.
        out_string = re.sub(r'\]\s*[;,]?\s*\[xref', '], [xref', out_string) 
        # Remove opening brackes from ref-series e.g. see [[1, 2, 3]] -> 1, 2, 3
        out_string = re.sub(r'[\[\(]\s*((\[xref:[^\]]+\](, )?)+)\s*[\]\)]', ' \\1 ', out_string)
        # move refs before fullstop of the same sentence.
        out_string = re.sub(r'\.\s*((\[xref:[^\]]+\](, )?)+)\s*(?=[A-Z]|$)', ' \\1. ', out_string)
        out_string = out_string.replace(' .', '.')
        out_string = re.sub(r' +', ' ', out_string)
        
        return out_string

    #@staticmethod
    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text by sentences at full stop avoiding splits at abreviations (e.g.) and within brackets.
        """
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
            just_closed = ch == "." and text[i-1] in "])"
            if just_closed and not all_closed:
                logger.info(f"Splitting at a full stop preceded by a closing bracket at position {i} in text: '{text[max(0, i-30):i+30]}'")
            if all_closed or just_closed:
                paren_depth = 0
                prefix = text[max(0, i - 10):i + 1].lower()

                is_abbrev = any(prefix.endswith(abbr) for abbr in ABBREVIATIONS)
                is_initial = text[i-1:i].isupper() and text[i-2:i-1].isspace() if i >= 2 else False

                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                has_trailing_space = j > i + 1

                next_is_upper = j < len(text) and text[j].isupper()

                if has_trailing_space and (not is_abbrev) and (not is_initial) and next_is_upper:
                    sentences.append(text[start:i + 1].strip())
                    start = j
                    i = j
                    continue

            i += 1

        tail = text[start:].strip()
        if tail:
            sentences.append(tail)

        return sentences

    @staticmethod
    def _split_string(in_string: str):
        """
        Split text by sentences at full stop avoiding splits at abreviations (e.g.). 
        """
        out_string_splitted = []
        # alternatively use simple pattern: (?<=\.)\s+(?=[A-Z])
        for out in re.split(r'(?<!\bfig)(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bvs)(?<=\.)\s+(?=[A-Z])', in_string):
            out = out.strip()
            if not out.endswith('.'):
                out += '.'
            out_string_splitted.append(out)

        return out_string_splitted

    #@staticmethod
    def _remove_and_keep_refs(
        self,
        in_string_list: list[str],
        ref_id_to_pmid: dict[str, str],
        section: int,
        paragraph: int
        ):
        """
        Extract contexts that contain xref markers.

        Returns tuples of `(clean_query, refs)` where `clean_query` has xref
        placeholders removed and `refs` keeps the matched markers.
        """
        contexts = []
        for i, query in enumerate(in_string_list):
            refs = re.findall(r'\[xref:.*?\]', query)
            ref_list = [ref_id_to_pmid[ref] for ref in refs if ref in ref_id_to_pmid]
            if ref_list:
                clean_query = re.sub(r' ?(\[xref:.*?\](, )?)+', '', query)
                #prev_context = in_string_list[i-1] if i > 0 else ""
                #prev_context = self.sentences[-1]
                quey_words = len(clean_query.split(' '))
                context = Context(
                    query=clean_query,
                    query_length=quey_words,
                    hits=ref_list,
                    n_hits=len(ref_list),
                    context=self.last_context,
                    position=(section, paragraph)
                )
                self.last_context = clean_query
                contexts.append(context)

        return contexts


    def extract_contexts(
            self,
            sections: list,
            references: list,
            pmcid :str,
            save_sentences: bool = False
        ):
        """
        Build reference contexts from parsed sections.

        Populates `self.contexts` with `(query, refs)` tuples. If
        `save_sentences` is `True`, also populates `self.sentences`.
        """
        
        self.contexts = []
        if save_sentences:
            self.sentences = []

        pmid_replace = {f"[xref:{r.rid}]": r.pmid for r in references if r.pmid}
        for s, section in enumerate(sections):
            for p,ref_paragraph in enumerate(section['paragraphs']):
                for ref_text in self._replace_refs(ref_paragraph):
                    ref_text = self._replace_ref_ranges(ref_text)
                    ref_text = self._clean_string(ref_text)
                    ref_text = self._split_sentences(ref_text) #_split_sentences(ref_text)
                    #if save_sentences:
                    self.sentences.append(ref_text)
                    ref_contexts = self._remove_and_keep_refs(ref_text, pmid_replace, s, p)
                    self.contexts.extend(ref_contexts)
            self.last_context = None   
        
    def contexts_to_dataframe(self):
        """
        Convert extracted contexts into a tabular representation.

        Returns a DataFrame with query text, PMID hits, and basic query stats.
        """
        df = pd.DataFrame([
            {name: getattr(a, name) for name in dir(a) if not name.startswith('__')}
            for a in self.contexts
        ])
        return df

    def text_to_list(self):
        """
        Returns list of all extracted lines.
        """
        # to remove refs also use re.sub(r' ?(\[xref:.*?\](, )?)+', '', line)
        lines = [
            re.sub(r'\(\s*\)', '', line)
            for paragraph in self.sentences
            for line in paragraph
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

    def _serialize_text_with_refs(self, element: etree._Element) -> str:
        """
        Recursively flatten inline content while preserving bibliographic refs.

        Nodes listed in `SKIPPED_CONTENT_TAGS` and extracted caption blocks are
        skipped, but their tails are preserved by the caller.
        """
        if self._is_bibr_xref(element):
            ref_id = element.get("rid")
            return f" [xref:{ref_id}] " if ref_id else ""

        element_name = self._local_name(element)
        if element_name in SKIPPED_CONTENT_TAGS or element_name in EXTRACTED_CAPTION_TAGS:
            return ""

        parts: list[str] = []
        if element.text:
            parts.append(element.text)

        for child in element:
            parts.append(self._serialize_text_with_refs(child))
            if child.tail:
                parts.append(child.tail)

        return "".join(parts)

    def _extract_caption_paragraph(self, element: etree._Element) -> str:
        """Extract one figure/table caption paragraph from label and caption."""
        label_element = self._find_element("./j:label", element)
        caption_element = self._find_element("./j:caption", element)
        label = self._normalize_text(self._serialize_text_with_refs(label_element)) if label_element is not None else ""
        caption = self._normalize_text(self._serialize_text_with_refs(caption_element)) if caption_element is not None else ""
        return self._normalize_text(" ".join(part for part in [label, caption] if part))

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
