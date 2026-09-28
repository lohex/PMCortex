"""Coordinate PMC JATS downloads, parsing, persistence, and cleanup."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Literal, cast

from loguru import logger
import pandas as pd
from tqdm.auto import tqdm

from pmcortex.downloader import PMCJATSDownloader
from pmcortex.jatsparser import JATSParser
from pmcortex.models import ParsedJATSResult
from pmcortex.serialization import (
    CONTEXT_COLUMNS,
    METADATA_COLUMNS,
    SOURCE_COLUMNS,
    contexts_to_dataframe,
    metadata_to_dataframe,
    normalize_author_lists,
    references_to_dataframe,
    render_author_yaml,
    render_diagnostics_json,
    render_positioned_sentences,
)


ProcessingStatus = Literal[
    "complete",
    "not_found",
    "no_fulltext",
    "download_error",
    "parse_error",
    "persist_error",
]
PROCESSING_STATUSES: tuple[ProcessingStatus, ...] = (
    "complete",
    "not_found",
    "no_fulltext",
    "download_error",
    "parse_error",
    "persist_error",
)


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    """Filesystem layout for one independently recoverable dataset build."""

    root: Path

    def __post_init__(self) -> None:
        """Normalize the dataset root without creating directories."""
        if not isinstance(self.root, Path):
            raise TypeError("root must be a pathlib.Path")
        object.__setattr__(self, "root", self.root.expanduser())

    @property
    def jats_dir(self) -> Path:
        """Return the directory containing downloaded JATS documents."""
        return self.root / "xml_jats"

    @property
    def parsed_dir(self) -> Path:
        """Return the directory containing per-article table shards."""
        return self.root / "parsed"

    @property
    def fulltexts_dir(self) -> Path:
        """Return the directory containing positioned full texts."""
        return self.root / "fulltexts"

    @property
    def authors_dir(self) -> Path:
        """Return the directory containing normalized author YAML files."""
        return self.root / "authors"

    @property
    def status_dir(self) -> Path:
        """Return the directory containing per-article processing records."""
        return self.root / "status"

    def prepare(self) -> None:
        """Create all directories needed by the ingestion pipeline."""
        for directory in (
            self.jats_dir,
            self.parsed_dir,
            self.fulltexts_dir,
            self.authors_dir,
            self.status_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class ProcessingRecord:
    """Persistent outcome for one PMC article and one parser schema version."""

    pmcid: str
    status: ProcessingStatus
    reason: str
    jats_path: str | None
    artifact_dir: str | None
    parser_schema_version: str
    source_size: int | None
    source_mtime_ns: int | None
    diagnostic_count: int
    unsafe_citation_rejection_count: int
    alignment_rejection_count: int
    updated_at: str


@dataclass(frozen=True, slots=True)
class MaterializedDataset:
    """Paths of the three dataset-wide tables built from article shards."""

    metadata: Path
    sources: Path
    contexts: Path


def _parse_jats(path: Path, expected_pmcid: str) -> ParsedJATSResult:
    """Parse one JATS document in an isolated worker process."""
    # Only the main process writes operational logs; concurrent file sinks can
    # otherwise interleave output when several parser processes finish at once.
    logger.disable("pmcortex")
    return JATSParser().parse(
        path,
        expected_pmcid=expected_pmcid,
        include_sentences=True,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace a text file atomically with a process-unique neighboring file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class PMCIngestionPipeline:
    """Stream downloaded JATS documents into parallel parser workers.

    Downloads remain serial so the downloader's NCBI rate limit is effective.
    Parsing runs in separate processes. The main process is the only writer and
    publishes per-article artifacts atomically. A successful status record is
    written after every artifact; optional JATS deletion happens only after
    that record has been stored.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        downloader: PMCJATSDownloader,
        *,
        parser_workers: int = 4,
        delete_jats_after_success: bool = False,
        parser_schema_version: str = "4",
        verbose: bool = False,
    ) -> None:
        """Configure ingestion without taking ownership of the downloader.

        Args:
            dataset_root: Root containing ``xml_jats``, outputs, and statuses.
            downloader: Configured downloader whose output is ``xml_jats``.
            parser_workers: Number of parser worker processes.
            delete_jats_after_success: Delete JATS only after durable outputs.
            parser_schema_version: Version used to invalidate old parser output.
            verbose: Show detailed operational messages in addition to progress.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If worker count, schema version, or output path is invalid.
        """
        if not isinstance(dataset_root, (str, Path)):
            raise TypeError("dataset_root must be a string or pathlib.Path")
        if not isinstance(downloader, PMCJATSDownloader):
            raise TypeError("downloader must be a PMCJATSDownloader")
        if not isinstance(parser_workers, int):
            raise TypeError("parser_workers must be an integer")
        if parser_workers <= 0:
            raise ValueError("parser_workers must be positive")
        if not isinstance(delete_jats_after_success, bool):
            raise TypeError("delete_jats_after_success must be a bool")
        if not isinstance(parser_schema_version, str):
            raise TypeError("parser_schema_version must be a string")
        if not parser_schema_version.strip():
            raise ValueError("parser_schema_version must not be empty")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a bool")

        self.layout = DatasetLayout(Path(dataset_root))
        self.layout.prepare()
        configured_output = downloader.out_dir.resolve()
        expected_output = self.layout.jats_dir.resolve()
        if configured_output != expected_output:
            raise ValueError(
                "downloader.out_dir must be the dataset's xml_jats directory: "
                f"{self.layout.jats_dir}"
            )

        self.downloader = downloader
        self.parser_workers = parser_workers
        self.delete_jats_after_success = delete_jats_after_success
        self.parser_schema_version = parser_schema_version
        self.verbose = verbose
        self._max_pending = parser_workers * 2

    def run(self, pmcids: Iterable[str]) -> list[ProcessingRecord]:
        """Parse local JATS files and stream missing requested IDs from PMC.

        Existing files are discovered first. Requested IDs whose current
        complete artifacts remain present are not downloaded again.

        Args:
            pmcids: PMC identifiers to ensure are processed.

        Returns:
            One current outcome per discovered or requested PMC identifier.

        Raises:
            TypeError: If ``pmcids`` is a string or contains non-string values.
            ValueError: If a contained PMC identifier is invalid.
        """
        normalized_ids = self._normalize_pmcids(pmcids)
        local_paths = sorted(self.layout.jats_dir.glob("PMC*.nxml"))
        return self._execute(local_paths, normalized_ids)

    def parse_available(self) -> list[ProcessingRecord]:
        """Parse all currently available JATS files without network requests."""
        local_paths = sorted(self.layout.jats_dir.glob("PMC*.nxml"))
        return self._execute(local_paths, ())

    def list_status_records(self) -> list[ProcessingRecord]:
        """Load all persisted article outcomes in PMCID order.

        Returns:
            Complete and failed outcomes, including the reason that prevented
            download, parsing, or persistence.
        """
        records = [
            record
            for path in sorted(self.layout.status_dir.glob("PMC*.json"))
            if (record := self._read_record(path.stem)) is not None
        ]
        return sorted(records, key=lambda record: record.pmcid)

    def materialize_tables(self) -> MaterializedDataset:
        """Atomically combine current complete article shards into CSV tables.

        Only records matching the configured parser schema and having every
        required artifact are included. This makes interrupted runs safe to
        resume before rebuilding the global tables.
        """
        current_records = [
            record
            for record in self.list_status_records()
            if self._is_current_complete(record.pmcid, None)
        ]
        self._log_info(
            "Materializing dataset tables from {} complete article shards",
            len(current_records),
        )
        metadata_frames = self._read_shards(current_records, "metadata.csv")
        source_frames = self._read_shards(current_records, "sources.csv")
        context_frames = self._read_shards(current_records, "contexts.csv")

        metadata_path = self.layout.root / "extracted_metadata.csv"
        sources_path = self.layout.root / "extracted_sources.csv"
        contexts_path = self.layout.root / "extracted_contexts.csv"
        self._write_combined_table(metadata_path, metadata_frames, METADATA_COLUMNS)
        self._write_combined_table(sources_path, source_frames, SOURCE_COLUMNS)
        self._write_combined_table(contexts_path, context_frames, CONTEXT_COLUMNS)
        self._log_info(
            "Materialized dataset tables: {} metadata rows, {} sources, "
            "{} contexts in {}",
            sum(len(frame) for frame in metadata_frames),
            sum(len(frame) for frame in source_frames),
            sum(len(frame) for frame in context_frames),
            self.layout.root,
        )
        return MaterializedDataset(metadata_path, sources_path, contexts_path)

    @staticmethod
    def _normalize_pmcids(pmcids: Iterable[str]) -> tuple[str, ...]:
        """Validate, normalize, and stable-deduplicate public PMC ID input."""
        if isinstance(pmcids, (str, bytes)):
            raise TypeError("pmcids must be an iterable of PMC ID strings")

        normalized: list[str] = []
        seen: set[str] = set()
        for pmcid in pmcids:
            if not isinstance(pmcid, str):
                raise TypeError("every PMCID must be a string")
            value = PMCJATSDownloader.normalize_pmcid(pmcid)
            if value not in seen:
                normalized.append(value)
                seen.add(value)
        return tuple(normalized)

    def _execute(
        self,
        local_paths: Sequence[Path],
        requested_pmcids: Sequence[str],
    ) -> list[ProcessingRecord]:
        """Run bounded parser work while serial downloads continue."""
        records: dict[str, ProcessingRecord] = {}
        local_pmcids = {path.stem for path in local_paths}
        local_paths_by_pmcid = {path.stem: path for path in local_paths}
        pending: dict[Future[ParsedJATSResult], tuple[str, Path]] = {}
        target_pmcids = local_pmcids | set(requested_pmcids)
        already_processed = {
            pmcid
            for pmcid in target_pmcids
            if self._is_current_complete(
                pmcid,
                local_paths_by_pmcid.get(pmcid),
            )
        }
        progress_pmcids = set(already_processed)

        self._log_info(
            "Starting PMC ingestion for {} articles: {} local JATS files, "
            "{} requested IDs, {} parser workers",
            len(target_pmcids),
            len(local_paths),
            len(requested_pmcids),
            self.parser_workers,
        )

        with tqdm(
            total=len(target_pmcids),
            initial=len(already_processed),
            desc="PMC ingestion",
            unit="article",
            dynamic_ncols=True,
        ) as progress:
            def advance_progress() -> None:
                """Count newly recorded terminal article outcomes once."""
                newly_recorded = set(records).difference(progress_pmcids)
                if not newly_recorded:
                    return
                progress.update(len(newly_recorded))
                progress_pmcids.update(newly_recorded)

            process_context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=self.parser_workers,
                mp_context=process_context,
            ) as executor:
                for path in local_paths:
                    self._schedule_or_reuse(
                        path.stem,
                        path,
                        executor,
                        pending,
                        records,
                    )
                    self._collect_completed(
                        pending,
                        records,
                        block_when_full=True,
                    )
                    advance_progress()

                to_download = [
                    pmcid
                    for pmcid in requested_pmcids
                    if pmcid not in local_pmcids
                    and not self._reuse_completed_without_source(pmcid, records)
                ]
                advance_progress()
                if to_download:
                    self._log_info(
                        "Downloading {} articles that have no reusable local output",
                        len(to_download),
                    )
                    for result in self.downloader.iter_downloads(to_download):
                        if result.status == "ok" and result.path is not None:
                            self._schedule_or_reuse(
                                result.pmcid,
                                result.path,
                                executor,
                                pending,
                                records,
                            )
                            self._collect_completed(
                                pending,
                                records,
                                block_when_full=True,
                            )
                        else:
                            record = self._download_failure_record(
                                result.pmcid,
                                result.status,
                                result.message,
                            )
                            self._write_record(record)
                            records[result.pmcid] = record
                            logger.warning(
                                "Download for {} finished with status {}: {}",
                                result.pmcid,
                                record.status,
                                record.reason,
                            )
                        advance_progress()
                else:
                    self._log_info("No article downloads are needed")

                if pending:
                    self._log_info(
                        "Waiting for {} remaining parser jobs",
                        len(pending),
                    )
                while pending:
                    self._collect_completed(pending, records, force_wait=True)
                    advance_progress()

        ordered_records = [records[pmcid] for pmcid in sorted(records)]
        status_counts = Counter(record.status for record in ordered_records)
        status_summary = ", ".join(
            f"{status}={status_counts[status]}"
            for status in PROCESSING_STATUSES
            if status_counts[status]
        )
        self._log_info(
            "Finished PMC ingestion: {} articles ({})",
            len(ordered_records),
            status_summary if status_summary else "no articles",
        )
        return ordered_records

    def _schedule_or_reuse(
        self,
        pmcid: str,
        path: Path,
        executor: ProcessPoolExecutor,
        pending: dict[Future[ParsedJATSResult], tuple[str, Path]],
        records: dict[str, ProcessingRecord],
    ) -> None:
        """Reuse current artifacts or submit one parse to the worker pool."""
        if self._is_current_complete(pmcid, path):
            record = self._read_record(pmcid)
            if record is None:
                raise RuntimeError(f"Missing complete processing record for {pmcid}")
            records[pmcid] = record
            self._log_info("Reusing complete artifacts for {}", pmcid)
            self._delete_jats_if_requested(path)
            return

        future = executor.submit(_parse_jats, path, pmcid)
        pending[future] = (pmcid, path)
        self._log_info(
            "Queued {} for parsing ({} parser jobs pending)",
            pmcid,
            len(pending),
        )

    def _reuse_completed_without_source(
        self,
        pmcid: str,
        records: dict[str, ProcessingRecord],
    ) -> bool:
        """Reuse complete artifacts after their optional JATS cleanup."""
        if not self._is_current_complete(pmcid, None):
            return False
        record = self._read_record(pmcid)
        if record is None:
            return False
        records[pmcid] = record
        self._log_info(
            "Reusing complete artifacts for {} without JATS source",
            pmcid,
        )
        return True

    def _collect_completed(
        self,
        pending: dict[Future[ParsedJATSResult], tuple[str, Path]],
        records: dict[str, ProcessingRecord],
        *,
        block_when_full: bool = False,
        force_wait: bool = False,
    ) -> None:
        """Persist selected completed futures in the main process."""
        should_block = force_wait or (
            block_when_full and len(pending) >= self._max_pending
        )
        if not pending or not should_block:
            return

        completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in completed:
            pmcid, path = pending.pop(future)
            try:
                parsed = future.result()
            except Exception as error:
                record = self._failure_record(
                    pmcid,
                    "parse_error",
                    repr(error),
                    path,
                )
                self._write_record(record)
                records[pmcid] = record
                logger.error("Could not parse {}: {!r}", path, error)
                continue

            try:
                record = self._persist_parsed(parsed, path)
            except Exception as error:
                record = self._failure_record(
                    pmcid,
                    "persist_error",
                    repr(error),
                    path,
                    diagnostic_count=len(parsed.diagnostics),
                    unsafe_citation_rejection_count=(
                        parsed.unsafe_citation_rejection_count
                    ),
                    alignment_rejection_count=parsed.alignment_rejection_count,
                )
                self._write_record(record)
                records[pmcid] = record
                logger.error("Could not persist parser output for {}: {!r}", pmcid, error)
                continue

            records[pmcid] = record
            self._log_info(
                "Completed {}: {} references, {} contexts, {} sentences, "
                "{} diagnostics, {} unsafe citation rejections, "
                "{} alignment rejections ({} parser jobs pending)",
                pmcid,
                len(parsed.article.references),
                len(parsed.contexts),
                len(parsed.sentences),
                len(parsed.diagnostics),
                parsed.unsafe_citation_rejection_count,
                parsed.alignment_rejection_count,
                len(pending),
            )
            self._delete_jats_if_requested(path)

    def _persist_parsed(
        self,
        parsed: ParsedJATSResult,
        path: Path,
    ) -> ProcessingRecord:
        """Publish every per-article artifact and finally its success record."""
        article = parsed.article
        if article.pmcid is None:
            raise ValueError("parsed article must have a PMCID before persistence")

        artifact_dir = self.layout.parsed_dir / article.pmcid
        metadata = metadata_to_dataframe(article)
        sources = references_to_dataframe(
            article.pmcid,
            article.references,
        )
        contexts = contexts_to_dataframe(parsed.contexts)
        fulltext = render_positioned_sentences(parsed.sentences)
        authors_yaml = render_author_yaml(normalize_author_lists(article))

        self._write_dataframe(artifact_dir / "metadata.csv", metadata)
        self._write_dataframe(artifact_dir / "sources.csv", sources)
        self._write_dataframe(artifact_dir / "contexts.csv", contexts)
        _atomic_write_text(
            artifact_dir / "diagnostics.json",
            render_diagnostics_json(parsed.diagnostics),
        )
        _atomic_write_text(
            self.layout.fulltexts_dir / f"{article.pmcid}.txt",
            fulltext,
        )
        _atomic_write_text(
            self.layout.authors_dir / f"{article.pmcid}.yaml",
            authors_yaml,
        )

        source_stat = path.stat()
        record = ProcessingRecord(
            pmcid=article.pmcid,
            status="complete",
            reason="parsed and persisted",
            jats_path=str(path),
            artifact_dir=str(artifact_dir),
            parser_schema_version=self.parser_schema_version,
            source_size=source_stat.st_size,
            source_mtime_ns=source_stat.st_mtime_ns,
            diagnostic_count=len(parsed.diagnostics),
            unsafe_citation_rejection_count=(
                parsed.unsafe_citation_rejection_count
            ),
            alignment_rejection_count=parsed.alignment_rejection_count,
            updated_at=self._timestamp(),
        )
        self._write_record(record)
        return record

    def _download_failure_record(
        self,
        pmcid: str,
        download_status: str,
        message: str | None,
    ) -> ProcessingRecord:
        """Map a downloader result to a persistent pipeline outcome."""
        status_by_download = {
            "not_found": "not_found",
            "no_fulltext": "no_fulltext",
            "error": "download_error",
        }
        status = cast(ProcessingStatus, status_by_download[download_status])
        reason = message if message is not None else download_status
        return ProcessingRecord(
            pmcid=pmcid,
            status=status,
            reason=reason,
            jats_path=None,
            artifact_dir=None,
            parser_schema_version=self.parser_schema_version,
            source_size=None,
            source_mtime_ns=None,
            diagnostic_count=0,
            unsafe_citation_rejection_count=0,
            alignment_rejection_count=0,
            updated_at=self._timestamp(),
        )

    def _failure_record(
        self,
        pmcid: str,
        status: Literal["parse_error", "persist_error"],
        reason: str,
        path: Path,
        *,
        diagnostic_count: int = 0,
        unsafe_citation_rejection_count: int = 0,
        alignment_rejection_count: int = 0,
    ) -> ProcessingRecord:
        """Build a failed parser or persistence record and retain its JATS."""
        source_stat = path.stat()
        return ProcessingRecord(
            pmcid=pmcid,
            status=status,
            reason=reason,
            jats_path=str(path),
            artifact_dir=None,
            parser_schema_version=self.parser_schema_version,
            source_size=source_stat.st_size,
            source_mtime_ns=source_stat.st_mtime_ns,
            diagnostic_count=diagnostic_count,
            unsafe_citation_rejection_count=unsafe_citation_rejection_count,
            alignment_rejection_count=alignment_rejection_count,
            updated_at=self._timestamp(),
        )

    def _write_record(self, record: ProcessingRecord) -> None:
        """Atomically store one article outcome as human-readable JSON."""
        path = self.layout.status_dir / f"{record.pmcid}.json"
        content = json.dumps(asdict(record), indent=2, sort_keys=True) + "\n"
        _atomic_write_text(path, content)

    def _read_record(self, pmcid: str) -> ProcessingRecord | None:
        """Read one internally generated status, ignoring damaged records."""
        path = self.layout.status_dir / f"{pmcid}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("status JSON must contain an object")
            return ProcessingRecord(
                pmcid=str(payload["pmcid"]),
                status=cast(ProcessingStatus, payload["status"]),
                reason=str(payload["reason"]),
                jats_path=cast(str | None, payload["jats_path"]),
                artifact_dir=cast(str | None, payload["artifact_dir"]),
                parser_schema_version=str(payload["parser_schema_version"]),
                source_size=cast(int | None, payload["source_size"]),
                source_mtime_ns=cast(int | None, payload["source_mtime_ns"]),
                diagnostic_count=int(payload["diagnostic_count"]),
                unsafe_citation_rejection_count=int(
                    payload.get("unsafe_citation_rejection_count", 0)
                ),
                alignment_rejection_count=int(
                    payload.get("alignment_rejection_count", 0)
                ),
                updated_at=str(payload["updated_at"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            logger.warning("Ignoring invalid processing record {}: {}", path, error)
            return None

    def _is_current_complete(self, pmcid: str, source_path: Path | None) -> bool:
        """Check schema version, artifact completeness, and source fingerprint."""
        record = self._read_record(pmcid)
        if record is None:
            return False
        correct_version = record.parser_schema_version == self.parser_schema_version
        if record.status != "complete" or not correct_version:
            return False
        if not all(path.is_file() for path in self._artifact_paths(pmcid)):
            return False
        if source_path is None:
            return True

        source_stat = source_path.stat()
        return (
            record.source_size == source_stat.st_size
            and record.source_mtime_ns == source_stat.st_mtime_ns
        )

    def _artifact_paths(self, pmcid: str) -> tuple[Path, ...]:
        """Return every file required before an article is considered complete."""
        artifact_dir = self.layout.parsed_dir / pmcid
        return (
            artifact_dir / "metadata.csv",
            artifact_dir / "sources.csv",
            artifact_dir / "contexts.csv",
            artifact_dir / "diagnostics.json",
            self.layout.fulltexts_dir / f"{pmcid}.txt",
            self.layout.authors_dir / f"{pmcid}.yaml",
        )

    def _delete_jats_if_requested(self, path: Path) -> None:
        """Delete a JATS source only after verified complete output exists."""
        if not self.delete_jats_after_success:
            return
        try:
            path.unlink(missing_ok=True)
            self._log_info("Deleted processed JATS source {}", path)
        except OSError as error:
            logger.warning("Could not delete processed JATS source {}: {}", path, error)

    def _log_info(self, message: str, *args: object) -> None:
        """Write an operational detail only when verbose output is enabled."""
        if self.verbose:
            logger.info(message, *args)

    @staticmethod
    def _write_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
        """Serialize one DataFrame and publish the CSV atomically."""
        _atomic_write_text(path, dataframe.to_csv(index=False))

    def _read_shards(
        self,
        records: Sequence[ProcessingRecord],
        filename: str,
    ) -> list[pd.DataFrame]:
        """Read one named shard for each current complete record."""
        return [
            pd.read_csv(self.layout.parsed_dir / record.pmcid / filename)
            for record in records
        ]

    @staticmethod
    def _write_combined_table(
        path: Path,
        frames: Sequence[pd.DataFrame],
        columns: Sequence[str],
    ) -> None:
        """Combine article shards with stable columns and publish atomically."""
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.reindex(columns=columns)
        else:
            combined = pd.DataFrame(columns=columns)
        _atomic_write_text(path, combined.to_csv(index=False))

    @staticmethod
    def _timestamp() -> str:
        """Return an unambiguous UTC timestamp for a processing record."""
        return datetime.now(timezone.utc).isoformat()
