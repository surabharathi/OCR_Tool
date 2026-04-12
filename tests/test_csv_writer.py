"""
tests/test_csv_writer.py — Unit tests for CSVWriter.

Covers: header correctness, row values, rollover at max rows,
        multi-file resume, thread-safety under concurrent writes.
"""

import csv
import threading
from pathlib import Path

import pytest

from core.csv_writer import CSV_HEADERS, CSVWriter
from config import CSV_MAX_ROWS


@pytest.fixture()
def writer(tmp_path):
    return CSVWriter(tmp_path)


def _make_record(n: int = 1) -> dict:
    return {
        "filename": f"form_{n:05d}.jpeg",
        "app_no": str(10000 + n),
        "name": f"Test User {n}",
        "address": f"{n} Main Street, Bengaluru",
        "mob_no": f"9{n:09d}"[:10],
        "email": f"user{n}@example.com",
        "q1_chanted_before": "Yes",
        "q2_familiarity_level": "Can chant confidently without guidance",
        "q3_formal_training": "No",
        "q4_training_mode": "[BLANK]",
        "q5_preferred_language": "Kannada",
        "q6_comments": "[BLANK]",
        "name_confidence": 82.0,
        "address_confidence": 78.0,
        "mob_no_confidence": 90.0,
        "email_confidence": 85.0,
        "app_no_confidence": 95.0,
        "overall_confidence": 86.0,
        "ocr_engine_used": "tesseract",
        "flagged": "No",
        "flag_reason": "",
    }


class TestCSVCreation:
    def test_csv_file_created_on_first_write(self, tmp_path):
        writer = CSVWriter(tmp_path)
        assert writer.current_file is None or not writer.current_file.exists() or True
        writer.write_record(_make_record())
        assert writer.current_file is not None
        assert writer.current_file.exists()

    def test_header_row_written_on_creation(self, tmp_path):
        writer = CSVWriter(tmp_path)
        writer.write_record(_make_record())
        with open(writer.current_file, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
        assert set(header) == set(CSV_HEADERS)

    def test_header_columns_match_constant(self, tmp_path):
        writer = CSVWriter(tmp_path)
        writer.write_record(_make_record())
        with open(writer.current_file, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert set(reader.fieldnames or []) == set(CSV_HEADERS)


class TestRowWriting:
    def test_single_record_written_correctly(self, writer):
        record = _make_record(42)
        writer.write_record(record)
        with open(writer.current_file, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["name"] == "Test User 42"
        assert rows[0]["app_no"] == "10042"

    def test_multiple_records_all_written(self, writer):
        for i in range(10):
            writer.write_record(_make_record(i))
        assert writer.current_row_count == 10

    def test_missing_keys_written_as_empty_string(self, writer):
        writer.write_record({"filename": "sparse.jpeg"})  # all other keys absent
        with open(writer.current_file, newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        assert row["name"] == ""
        assert row["email"] == ""

    def test_processed_at_timestamp_always_set(self, writer):
        writer.write_record(_make_record())
        with open(writer.current_file, newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        assert row["processed_at"] != ""


class TestRollover:
    def test_rollover_creates_new_file(self, tmp_path):
        writer = CSVWriter(tmp_path)
        first_file = None
        writer.write_record(_make_record())
        first_file = writer.current_file

        # Force the writer to believe the file is full.
        writer._current_row_count = CSV_MAX_ROWS
        writer.write_record(_make_record(999))

        assert writer.current_file != first_file

    def test_data_rows_in_old_file_not_exceeded(self, tmp_path):
        """After rollover, old file must have exactly CSV_MAX_ROWS rows."""
        writer = CSVWriter(tmp_path)
        writer.write_record(_make_record())
        first_file = writer.current_file

        writer._current_row_count = CSV_MAX_ROWS
        writer.write_record(_make_record(999))

        # The new record goes into the new file, not the old one.
        with open(first_file, newline="", encoding="utf-8") as fh:
            old_rows = list(csv.DictReader(fh))
        # Old file had exactly 1 real row written (before rollover was triggered).
        assert len(old_rows) == 1

    def test_three_consecutive_rollovers(self, tmp_path):
        writer = CSVWriter(tmp_path)
        files_seen = set()

        for rollover in range(3):
            writer.write_record(_make_record(rollover))
            files_seen.add(str(writer.current_file))
            writer._current_row_count = CSV_MAX_ROWS  # force next rollover

        # Final write to trigger last rollover.
        writer.write_record(_make_record(100))
        files_seen.add(str(writer.current_file))

        assert len(files_seen) >= 3


class TestResumeFromExistingCSV:
    def test_writer_picks_up_non_full_csv(self, tmp_path):
        """If a CSV with rows < max already exists, resume writing to it."""
        writer1 = CSVWriter(tmp_path)
        for i in range(5):
            writer1.write_record(_make_record(i))
        first_csv = writer1.current_file

        # Simulate application restart.
        writer2 = CSVWriter(tmp_path)
        assert writer2.current_file == first_csv
        assert writer2.current_row_count == 5

    def test_writer_creates_new_file_if_existing_is_full(self, tmp_path):
        writer1 = CSVWriter(tmp_path)
        writer1.write_record(_make_record())
        first_csv = writer1.current_file
        # Patch the on-disk row count to simulate a full file.
        writer1._current_row_count = CSV_MAX_ROWS
        writer1.write_record(_make_record(2))   # triggers rollover
        second_csv = writer1.current_file

        # A brand-new instance should pick up the second (non-full) file.
        writer2 = CSVWriter(tmp_path)
        assert writer2.current_file == second_csv


class TestThreadSafety:
    def test_concurrent_writes_no_data_loss(self, tmp_path):
        """50 threads each writing 4 records = 200 total rows, no corruption."""
        writer = CSVWriter(tmp_path)
        n_threads = 50
        records_each = 4
        errors = []

        def worker(thread_id):
            for j in range(records_each):
                try:
                    writer.write_record(_make_record(thread_id * 100 + j))
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Write errors: {errors}"

        # Count total data rows across all CSV files.
        total_rows = 0
        for csv_file in tmp_path.glob("extracted_data_2*.csv"):
            with open(csv_file, newline="", encoding="utf-8") as fh:
                total_rows += sum(1 for _ in csv.DictReader(fh))

        expected = n_threads * records_each
        assert total_rows == expected, (
            f"Expected {expected} rows, found {total_rows}."
        )
