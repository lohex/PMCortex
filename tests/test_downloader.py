import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from pmcortex.downloader import PMCJATSDownloader
from pmcortex.models import DownloadResult


class TestPMCJATSDownloader(unittest.TestCase):
    def test_normalize_pmcid(self) -> None:
        self.assertEqual(PMCJATSDownloader.normalize_pmcid("3438321"), "PMC3438321")
        self.assertEqual(PMCJATSDownloader.normalize_pmcid("pmc3438321"), "PMC3438321")

    def test_normalize_pmcid_invalid(self) -> None:
        with self.assertRaises(ValueError):
            PMCJATSDownloader.normalize_pmcid("PMC34A")

    def test_oai_identifier(self) -> None:
        self.assertEqual(
            PMCJATSDownloader.oai_identifier("PMC3438321"),
            "oai:pubmedcentral.nih.gov:3438321",
        )

    def test_extract_jats_success(self) -> None:
        oai_xml = b"""
        <OAI-PMH>
          <GetRecord>
            <record>
              <metadata>
                <article><front><article-meta/></front></article>
              </metadata>
            </record>
          </GetRecord>
        </OAI-PMH>
        """
        out = PMCJATSDownloader.extract_jats(oai_xml)
        self.assertIn(b"<article>", out)

    def test_extract_jats_oai_error(self) -> None:
        oai_xml = b"<OAI-PMH><error code='idDoesNotExist'>missing</error></OAI-PMH>"
        with self.assertRaises(ValueError):
            PMCJATSDownloader.extract_jats(oai_xml)

    def test_download_one_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            cached_path = out_dir / "PMC3438321.nxml"
            cached_path.write_text("<article/>", encoding="utf-8")

            downloader = PMCJATSDownloader(out_dir=out_dir)
            try:
                result = downloader.download_one("3438321")
                self.assertEqual(result.status, "ok")
                self.assertEqual(result.message, "cached")
                self.assertEqual(result.path, cached_path)
            finally:
                downloader.close()

    def test_download_one_not_found_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            downloader = PMCJATSDownloader(out_dir=tmp)
            try:
                with patch.object(
                    PMCJATSDownloader,
                    "fetch_oai_xml",
                    side_effect=ValueError("OAI error code=idDoesNotExist msg=missing"),
                ):
                    result = downloader.download_one("PMC440378")

                self.assertEqual(result.status, "not_found")
                self.assertIsNone(result.path)
            finally:
                downloader.close()

    def test_iter_downloads_yields_each_result_immediately(self) -> None:
        """The streaming API preserves order and stops at the requested limit."""
        with tempfile.TemporaryDirectory() as tmp:
            downloader = PMCJATSDownloader(out_dir=tmp)
            first = DownloadResult("PMC1", Path(tmp) / "PMC1.nxml", "ok")
            second = DownloadResult("PMC2", None, "not_found", "missing")
            try:
                with patch.object(
                    downloader,
                    "download_one",
                    side_effect=[first, second],
                ) as download_one:
                    results = list(
                        downloader.iter_downloads(
                            ["PMC1", "PMC2", "PMC3"],
                            limit=2,
                        )
                    )

                self.assertEqual(results, [first, second])
                self.assertEqual(
                    download_one.call_args_list,
                    [call("PMC1"), call("PMC2")],
                )
            finally:
                downloader.close()

    def test_iter_downloads_rejects_string_input(self) -> None:
        """A single string must not be interpreted as an iterable of IDs."""
        with tempfile.TemporaryDirectory() as tmp:
            downloader = PMCJATSDownloader(out_dir=tmp)
            try:
                with self.assertRaises(TypeError):
                    list(downloader.iter_downloads("PMC3438321"))
            finally:
                downloader.close()


if __name__ == "__main__":
    unittest.main()
