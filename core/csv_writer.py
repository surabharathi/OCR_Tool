"""
core/csv_writer.py — Thread-safe CSV writer with automatic row-count rollover.

When a CSV file reaches CSV_MAX_ROWS data rows a new timestamped file is
created automatically.  The writer resumes from the most recently created
file that still has capacity on application restart.
"""

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from config import CSV_BASE_NAME, CSV_MAX_ROWS

logger = logging.getLogger(__name__)

# Canonical column order — every record dict is normalised to this before writing.
CSV_HEADERS: list[str] = [
    "filename",
    "app_no",
    "name",
    "address_line_1",
    "address_line_2",
    "address_line_3",
    "mob_no",
    "email",
    "q1_chanted_before",
    "q2_familiarity_level",
    "q3_formal_training",
    "q4_training_mode",
    "q5_preferred_language",
    "q6_comments",
    "app_no_confidence",
    "name_confidence",
    "address_line_1_confidence",
    "address_line_2_confidence",
    "address_line_3_confidence",
    "mob_no_confidence",
    "email_confidence",
    "overall_confidence",
    "ocr_engine_used",
    "flagged",
    "flag_reason",
    "processed_at",
]


class CSVWriter:
    """Append extracted records to CSV files; roll over at CSV_MAX_ROWS rows.

    Thread-safe via an internal lock — safe to call ``write_record`` from
    multiple worker threads simultaneously.
    """

    def __init__(self, output_folder: Path) -> None:
        self._folder = output_folder
        self._folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current_file: Path | None = None
        self._current_row_count: int = 0
        self._resolve_current_file()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_current_file(self) -> None:
        """Pick up the most recent CSV that still has row capacity."""
        existing = sorted(self._folder.glob(f"{CSV_BASE_NAME}_2*.csv"))
        for csv_path in reversed(existing):
            count = self._count_data_rows(csv_path)
            if count < CSV_MAX_ROWS:
                self._current_file = csv_path
                self._current_row_count = count
                logger.info("Resuming CSV '%s' at row %d.", csv_path.name, count)
                return
        self._create_new_file()

    def _count_data_rows(self, csv_path: Path) -> int:
        """Count non-header rows; return 0 on any read error."""
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.reader(fh)
                next(reader, None)   # skip header
                return sum(1 for _ in reader)
        except OSError as exc:
            logger.error("Cannot count rows in %s: %s", csv_path, exc)
            return 0

    def _create_new_file(self) -> None:
        """Create a new timestamped CSV file and write the header row."""
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self._current_file = self._folder / f"{CSV_BASE_NAME}_{stamp}.csv"
        self._current_row_count = 0
        try:
            with open(self._current_file, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
                writer.writeheader()
            logger.info("Created new CSV: %s", self._current_file.name)
        except OSError as exc:
            logger.error("Failed to create CSV file: %s", exc)
            raise

    # ── Public API ────────────────────────────────────────────────────────────

    def write_record(self, record: dict) -> None:
        """Append one extraction record.  Rolls over to a new file when full.

        Missing keys are written as empty strings.  ``processed_at`` is always
        overwritten with the current UTC timestamp.

        Raises:
            OSError: if the underlying file cannot be written.
        """
        with self._lock:
            if self._current_row_count >= CSV_MAX_ROWS:
                logger.info("CSV max rows (%d) reached — rolling over.", CSV_MAX_ROWS)
                self._create_new_file()

            row = {col: record.get(col, "") for col in CSV_HEADERS}
            row["processed_at"] = datetime.now(tz=timezone.utc).isoformat()

            try:
                with open(self._current_file, "a", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
                    writer.writerow(row)
                self._current_row_count += 1
                logger.debug(
                    "Wrote record for '%s' (row %d).",
                    record.get("filename", "?"),
                    self._current_row_count,
                )
            except OSError as exc:
                logger.error("Write failed for '%s': %s", record.get("filename"), exc)
                raise

    @property
    def current_file(self) -> Path | None:
        """Path to the CSV file currently being written to."""
        return self._current_file

    @property
    def current_row_count(self) -> int:
        """Number of data rows in the current file."""
        with self._lock:
            return self._current_row_count
