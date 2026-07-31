"""Fetch and extract PubMed abstracts, MeSH terms, and descriptor categories."""

from collections.abc import Callable, Iterable
from pathlib import Path
import re

from lxml import etree
import pandas as pd

from pmcortex.http_context_manager import HTTPContextManager, RequestRateLimiter


class PubMedMeSHClient(HTTPContextManager):
    """Fetch PubMed article metadata and extract assigned MeSH headings."""

    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(
        self,
        *,
        user_agent: str = "PubMedMeSHClient/1.0 (contact: you@example.org)",
        timeout_s: float = 30.0,
        max_requests_per_second: float = 2.9,
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        """Create a rate-limited PubMed E-utilities client."""
        self._rate_limiter = RequestRateLimiter(
            max_requests_per_second,
            time_fn=time_fn,
            sleep_fn=sleep_fn,
        )
        super().__init__(
            user_agent=user_agent,
            timeout_s=timeout_s,
            accept="application/xml,text/xml;q=0.9,*/*;q=0.1",
        )

    def fetch_mesh_terms(
        self,
        pmids: Iterable[str | int],
        *,
        include_qualifiers: bool = True,
    ) -> pd.DataFrame:
        """Fetch PubMed XML for the given PMIDs and return MeSH annotations."""
        xml = self.fetch_pubmed_xml(pmids)
        return extract_mesh_terms_from_pubmed_xml(
            xml,
            include_qualifiers=include_qualifiers,
        )

    def fetch_abstracts(self, pmids: Iterable[str | int]) -> pd.DataFrame:
        """Fetch PubMed XML for the given PMIDs and return article abstracts."""
        xml = self.fetch_pubmed_xml(pmids)
        return extract_abstracts_from_pubmed_xml(xml)

    def fetch_pubmed_xml(self, pmids: Iterable[str | int]) -> bytes:
        """Fetch PubMed article XML for a PMID iterable via E-utilities."""
        pmid_list = [str(pmid).strip() for pmid in pmids if str(pmid).strip()]
        if not pmid_list:
            raise ValueError("pmids must contain at least one PMID.")

        self._rate_limiter.wait()
        response = self.client.post(
            self.EFETCH_URL,
            data={
                "db": "pubmed",
                "id": ",".join(pmid_list),
                "retmode": "xml",
            },
        )
        response.raise_for_status()
        return response.content


def extract_mesh_terms_from_pubmed_xml(
    xml: bytes | str,
    *,
    include_qualifiers: bool = True,
) -> pd.DataFrame:
    """
    Parse PubMed XML and extract article-level MeSH descriptor assignments.

    Notes:
    - MeSH hierarchy itself is not encoded fully in PubMed article XML.
    - This parser extracts assigned descriptors and qualifiers only.
    - Tree numbers require the separate MeSH descriptor records.
    """
    parser = etree.XMLParser(recover=True, resolve_entities=False, huge_tree=True)
    root = etree.fromstring(xml, parser=parser)

    rows: list[dict[str, object]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//MedlineCitation/PMID")
        mesh_list = article.find(".//MedlineCitation/MeshHeadingList")
        if mesh_list is None:
            continue

        for mesh_heading in mesh_list.findall("./MeshHeading"):
            descriptor = mesh_heading.find("./DescriptorName")
            if descriptor is None:
                continue

            descriptor_name = " ".join(descriptor.itertext()).strip()
            descriptor_ui = descriptor.get("UI")
            descriptor_major = (descriptor.get("MajorTopicYN") or "N") == "Y"

            qualifiers = mesh_heading.findall("./QualifierName")
            if qualifiers and include_qualifiers:
                for qualifier in qualifiers:
                    qualifier_name = " ".join(qualifier.itertext()).strip()
                    qualifier_ui = qualifier.get("UI")
                    qualifier_major = (qualifier.get("MajorTopicYN") or "N") == "Y"
                    rows.append(
                        {
                            "pmid": pmid,
                            "descriptor_ui": descriptor_ui,
                            "descriptor_name": descriptor_name,
                            "descriptor_major_topic": descriptor_major,
                            "qualifier_ui": qualifier_ui,
                            "qualifier_name": qualifier_name,
                            "qualifier_major_topic": qualifier_major,
                        }
                    )
            else:
                rows.append(
                    {
                        "pmid": pmid,
                        "descriptor_ui": descriptor_ui,
                        "descriptor_name": descriptor_name,
                        "descriptor_major_topic": descriptor_major,
                        "qualifier_ui": None,
                        "qualifier_name": None,
                        "qualifier_major_topic": None,
                    }
                )

    return pd.DataFrame(rows, dtype=object)


def extract_abstracts_from_pubmed_xml(xml: bytes | str) -> pd.DataFrame:
    """Parse PubMed XML and extract article-level abstracts."""
    parser = etree.XMLParser(recover=True, resolve_entities=False, huge_tree=True)
    root = etree.fromstring(xml, parser=parser)

    rows: list[dict[str, object]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//MedlineCitation/PMID")
        abstract_nodes = article.findall(".//Article/Abstract/AbstractText")

        parts: list[str] = []
        for node in abstract_nodes:
            text = " ".join(part.strip() for part in node.itertext() if part and part.strip())
            if not text:
                continue

            label = (node.get("Label") or "").strip()
            if label:
                text = f"{label}: {text}"
            parts.append(text)

        rows.append(
            {
                "pmid": pmid,
                "abstract": "\n".join(parts) if parts else None,
            }
        )

    return pd.DataFrame(rows, dtype=object)

def extract_mesh_coarsest_descriptor_categories(xml_path: str | Path, level: int = 2) -> list[dict[str, str]]:
    """
    Extract MeSH descriptors at the coarsest concrete level, e.g. A01, B01, C01.

    Returns a list of dicts with:
    - ui
    - name
    - tree_number
    """
    if level == 1:
        pattern = re.compile(r"^[A-Z]\d{2}$")
    elif level == 2:
        pattern = re.compile(r"^[A-Z]\d{2}\.\d{3}$")
    else:
        raise ValueError(f"Unsupported level: {level}")

    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"MeSH descriptor file not found: {path}")

    root = etree.parse(path).getroot()
    results: list[dict[str, str]] = []
    seen: set[tuple[str | None, str]] = set()

    for desc in root.findall(".//DescriptorRecord"):
        ui = desc.findtext("DescriptorUI")
        name = desc.findtext("./DescriptorName/String")

        for tn in desc.findall(".//TreeNumberList/TreeNumber"):
            tree_number = "" if tn.text is None else tn.text.strip()
            if pattern.fullmatch(tree_number):
                key = (ui, tree_number)
                if key not in seen:
                    seen.add(key)
                    results.append(
                        {
                            "ui": ui,
                            "name": name,
                            "tree_number": tree_number,
                        }
                    )

    results.sort(key=lambda x: x["tree_number"])
    return results
