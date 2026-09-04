"""Integration tests for recoverable download-and-parse ingestion."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from pmcortex.downloader import PMCJATSDownloader
from pmcortex.models import DownloadResult
from pmcortex.pipeline import PMCIngestionPipeline
from tests._shared import DATA_DIR


class TestPMCIngestionPipeline(unittest.TestCase):
    """Exercise success, resumption, cleanup, and failure auditing."""

    def test_success_is_materialized_and_jats_is_deleted(self) -> None:
        """A complete status makes source deletion and later reuse safe."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_root = Path(temporary_directory)
            jats_dir = dataset_root / "xml_jats"
            jats_dir.mkdir()
            jats_path = jats_dir / "PMC3438321.nxml"
            shutil.copy2(DATA_DIR / jats_path.name, jats_path)

            downloader = PMCJATSDownloader(jats_dir)
            try:
                pipeline = PMCIngestionPipeline(
                    dataset_root,
                    downloader,
                    parser_workers=1,
                    delete_jats_after_success=True,
                    parser_schema_version="test-v1",
                )
                records = pipeline.parse_available()

                self.assertEqual(len(records), 1)
                self.assertEqual(records[0].status, "complete")
                self.assertFalse(jats_path.exists())
                self.assertTrue(
                    (dataset_root / "status" / "PMC3438321.json").is_file()
                )
                self.assertTrue(
                    (dataset_root / "fulltexts" / "PMC3438321.txt").is_file()
                )
                expected_artifacts = [
                    dataset_root / "parsed" / "PMC3438321" / filename
                    for filename in ("metadata.csv", "sources.csv", "contexts.csv")
                ]
                expected_artifacts.append(
                    dataset_root / "authors" / "PMC3438321.yaml"
                )
                self.assertTrue(all(path.is_file() for path in expected_artifacts))

                tables = pipeline.materialize_tables()
                contexts = pd.read_csv(tables.contexts)
                self.assertIn("source_pmcid", contexts.columns)
                self.assertTrue(contexts["source_pmcid"].eq("PMC3438321").all())

                with patch.object(
                    downloader,
                    "iter_downloads",
                    side_effect=AssertionError("completed article was downloaded"),
                ), patch("pmcortex.pipeline.tqdm") as progress_factory:
                    reused = pipeline.run(["PMC3438321"])
                self.assertEqual(reused[0].status, "complete")
                progress_factory.assert_called_once_with(
                    total=1,
                    initial=1,
                    desc="PMC ingestion",
                    unit="article",
                    dynamic_ncols=True,
                )
                progress = progress_factory.return_value.__enter__.return_value
                progress.update.assert_not_called()
            finally:
                downloader.close()

    def test_persistence_failure_is_recorded_and_keeps_jats(self) -> None:
        """Cleanup must not remove a source whose output could not be stored."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_root = Path(temporary_directory)
            jats_dir = dataset_root / "xml_jats"
            jats_dir.mkdir()
            jats_path = jats_dir / "PMC3438321.nxml"
            shutil.copy2(DATA_DIR / jats_path.name, jats_path)

            downloader = PMCJATSDownloader(jats_dir)
            try:
                pipeline = PMCIngestionPipeline(
                    dataset_root,
                    downloader,
                    parser_workers=1,
                    delete_jats_after_success=True,
                )
                with patch.object(
                    pipeline,
                    "_persist_parsed",
                    side_effect=OSError("disk full"),
                ):
                    records = pipeline.parse_available()

                self.assertEqual(records[0].status, "persist_error")
                self.assertIn("disk full", records[0].reason)
                self.assertTrue(jats_path.is_file())
                status_path = dataset_root / "status" / "PMC3438321.json"
                status_data = json.loads(status_path.read_text(encoding="utf-8"))
                self.assertEqual(status_data["status"], "persist_error")
            finally:
                downloader.close()

    def test_parse_failure_is_recorded_and_keeps_jats(self) -> None:
        """A mismatched JATS file remains available for diagnosis and retry."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_root = Path(temporary_directory)
            jats_dir = dataset_root / "xml_jats"
            jats_dir.mkdir()
            jats_path = jats_dir / "PMC123.nxml"
            shutil.copy2(DATA_DIR / "PMC3438321.nxml", jats_path)

            downloader = PMCJATSDownloader(jats_dir)
            try:
                pipeline = PMCIngestionPipeline(
                    dataset_root,
                    downloader,
                    parser_workers=1,
                    delete_jats_after_success=True,
                )
                records = pipeline.parse_available()

                self.assertEqual(records[0].status, "parse_error")
                self.assertIn("does not match", records[0].reason)
                self.assertTrue(jats_path.is_file())
                status_path = dataset_root / "status" / "PMC123.json"
                status_data = json.loads(status_path.read_text(encoding="utf-8"))
                self.assertEqual(status_data["status"], "parse_error")
                self.assertTrue(status_data["reason"])
            finally:
                downloader.close()

    def test_download_failure_reason_is_persisted(self) -> None:
        """Unavailable remote documents are visible in the status directory."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_root = Path(temporary_directory)
            jats_dir = dataset_root / "xml_jats"
            downloader = PMCJATSDownloader(jats_dir)
            try:
                pipeline = PMCIngestionPipeline(
                    dataset_root,
                    downloader,
                    parser_workers=1,
                )
                missing = DownloadResult(
                    "PMC440378",
                    None,
                    "not_found",
                    "OAI idDoesNotExist",
                )
                with patch.object(
                    downloader,
                    "iter_downloads",
                    return_value=iter([missing]),
                ):
                    records = pipeline.run(["PMC440378"])

                self.assertEqual(records[0].status, "not_found")
                self.assertEqual(records[0].reason, "OAI idDoesNotExist")
                persisted = pipeline.list_status_records()
                self.assertEqual(persisted, records)
            finally:
                downloader.close()


if __name__ == "__main__":
    unittest.main()
