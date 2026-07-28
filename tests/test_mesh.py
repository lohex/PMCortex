import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from pmcortex.mesh import (
    PubMedMeSHClient,
    extract_abstracts_from_pubmed_xml,
    extract_mesh_coarsest_descriptor_categories,
    extract_mesh_terms_from_pubmed_xml,
)


PUBMED_XML = b"""
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName UI="D009369" MajorTopicYN="Y">Neoplasms</DescriptorName>
          <QualifierName UI="Q000235" MajorTopicYN="N">genetics</QualifierName>
          <QualifierName UI="Q000378" MajorTopicYN="Y">metabolism</QualifierName>
        </MeshHeading>
        <MeshHeading>
          <DescriptorName UI="D001943" MajorTopicYN="N">Breast Neoplasms</DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <Article>
      <Abstract>
        <AbstractText Label="BACKGROUND">First section.</AbstractText>
        <AbstractText>Second section.</AbstractText>
      </Abstract>
    </Article>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>67890</PMID>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName UI="D002318" MajorTopicYN="N">Cells</DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <Article>
      <Abstract>
        <AbstractText>Standalone abstract.</AbstractText>
      </Abstract>
    </Article>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>24680</PMID>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class TestMeSHExtraction(unittest.TestCase):
    def test_extract_descriptor_categories_requires_local_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "desc.xml"

            with self.assertRaises(FileNotFoundError):
                extract_mesh_coarsest_descriptor_categories(missing_path)

            self.assertFalse(missing_path.exists())

    def test_extract_descriptor_categories_filters_requested_level(self) -> None:
        xml = """
        <DescriptorRecordSet>
          <DescriptorRecord>
            <DescriptorUI>D000001</DescriptorUI>
            <DescriptorName><String>Example</String></DescriptorName>
            <TreeNumberList>
              <TreeNumber>A01</TreeNumber>
              <TreeNumber>A01.111</TreeNumber>
            </TreeNumberList>
          </DescriptorRecord>
        </DescriptorRecordSet>
        """
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "desc.xml"
            path.write_text(xml)

            categories = extract_mesh_coarsest_descriptor_categories(path, level=2)

        self.assertEqual(
            categories,
            [{"ui": "D000001", "name": "Example", "tree_number": "A01.111"}],
        )

    def test_extract_mesh_terms_with_qualifiers(self) -> None:
        df = extract_mesh_terms_from_pubmed_xml(PUBMED_XML)

        self.assertEqual(len(df), 4)
        self.assertEqual(
            list(df.columns),
            [
                "pmid",
                "descriptor_ui",
                "descriptor_name",
                "descriptor_major_topic",
                "qualifier_ui",
                "qualifier_name",
                "qualifier_major_topic",
            ],
        )

        first = df.iloc[0].to_dict()
        self.assertEqual(first["pmid"], "12345")
        self.assertEqual(first["descriptor_name"], "Neoplasms")
        self.assertEqual(first["qualifier_name"], "genetics")
        self.assertTrue(first["descriptor_major_topic"])
        self.assertFalse(first["qualifier_major_topic"])

        second = df.iloc[1].to_dict()
        self.assertEqual(second["qualifier_name"], "metabolism")
        self.assertTrue(second["qualifier_major_topic"])

        third = df.iloc[2].to_dict()
        self.assertEqual(third["descriptor_name"], "Breast Neoplasms")
        self.assertIsNone(third["qualifier_name"])

    def test_extract_mesh_terms_without_qualifiers(self) -> None:
        df = extract_mesh_terms_from_pubmed_xml(PUBMED_XML, include_qualifiers=False)

        self.assertEqual(len(df), 3)
        self.assertTrue(df["qualifier_name"].isna().all())
        self.assertEqual(df.iloc[0]["descriptor_name"], "Neoplasms")

    def test_extract_abstracts_from_pubmed_xml(self) -> None:
        df = extract_abstracts_from_pubmed_xml(PUBMED_XML)

        self.assertEqual(
            list(df.columns),
            ["pmid", "abstract"],
        )
        self.assertEqual(len(df), 3)
        self.assertEqual(
            df.iloc[0]["abstract"],
            "BACKGROUND: First section.\nSecond section.",
        )
        self.assertEqual(df.iloc[1]["abstract"], "Standalone abstract.")
        self.assertIsNone(df.iloc[2]["abstract"])

    def test_fetch_pubmed_xml_uses_post_with_joined_pmids(self) -> None:
        client = PubMedMeSHClient()
        response = Mock()
        response.content = PUBMED_XML
        response.raise_for_status = Mock()
        client.client = Mock()
        client.client.post.return_value = response

        xml = client.fetch_pubmed_xml(["12345", "67890"])

        self.assertEqual(xml, PUBMED_XML)
        client.client.post.assert_called_once_with(
            client.EFETCH_URL,
            data={
                "db": "pubmed",
                "id": "12345,67890",
                "retmode": "xml",
            },
        )
        response.raise_for_status.assert_called_once_with()
        client.close()

    def test_fetch_pubmed_xml_rate_limits_requests_below_three_per_second(self) -> None:
        now = 0.0
        sleep_calls: list[float] = []

        def fake_time() -> float:
            return now

        def fake_sleep(seconds: float) -> None:
            nonlocal now
            sleep_calls.append(seconds)
            now += seconds

        client = PubMedMeSHClient(
            max_requests_per_second=2.9,
            time_fn=fake_time,
            sleep_fn=fake_sleep,
        )
        response = Mock()
        response.content = PUBMED_XML
        response.raise_for_status = Mock()
        client.client = Mock()
        client.client.post.return_value = response

        client.fetch_pubmed_xml(["12345"])
        client.fetch_pubmed_xml(["67890"])

        self.assertEqual(client.client.post.call_count, 2)
        self.assertEqual(len(sleep_calls), 1)
        self.assertGreaterEqual(sleep_calls[0], 1 / 2.9)
        client.close()


if __name__ == "__main__":
    unittest.main()
