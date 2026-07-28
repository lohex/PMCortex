import unittest

from lxml import etree

from pmcortex.pmcsearch import PMCSearch


class TestPMCSearch(unittest.TestCase):
    def test_get_results_requires_query(self) -> None:
        search = PMCSearch()
        try:
            with self.assertRaises(ValueError):
                search.get_results()
        finally:
            search.close()

    def test_extract_pmcids_from_parsed_html(self) -> None:
        html = """
        <html><body>
          <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3438321" data-ga-label="PMC3438321">Title A</a>
          <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3438321" data-ga-label="PMC3438321">Duplicate</a>
          <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC2693326" data-ga-label="PMC2693326">Title B</a>
        </body></html>
        """
        search = PMCSearch()
        try:
            search.doc = etree.fromstring(html.encode("utf-8"), parser=etree.HTMLParser())
            hits = search.extract_pmcids()
            # Current implementation keeps duplicates.
            self.assertEqual(len(hits), 3)
            self.assertEqual([h["pmcid"] for h in hits], ["PMC3438321", "PMC3438321", "PMC2693326"])
        finally:
            search.close()

    def test_get_all_returns_dataframe(self) -> None:
        search = PMCSearch()
        try:
            search.hit_list = [{"pmcid": "PMC1", "title": "T1"}]
            df = search.get_all()
            self.assertEqual(list(df.columns), ["pmcid", "title"])
            self.assertEqual(len(df), 1)
        finally:
            search.close()


if __name__ == "__main__":
    unittest.main()
