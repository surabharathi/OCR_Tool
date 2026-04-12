"""
tests/test_stress.py — Stress tests for the single-engine pipeline.
Run with:  pytest -m slow
"""

import csv
import gc
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from core.csv_writer import CSVWriter
from core.orchestrator import Orchestrator
from core.resume_tracker import ResumeTracker
from tests.conftest import REAL_IMAGES, make_result, make_failed_result

pytestmark = pytest.mark.slow


def _populate_folder(folder: Path, n: int) -> list[Path]:
    if not REAL_IMAGES:
        pytest.skip("No fixture images.")
    files = []
    for i in range(n):
        dst = folder / f"stress_{i:05d}.jpeg"
        shutil.copy(REAL_IMAGES[i % len(REAL_IMAGES)], dst)
        files.append(dst)
    return files


class TestLargeVolume:
    def test_1000_images_all_processed(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _populate_folder(img_dir, 1000)

        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.return_value = make_result()

            summary = Orchestrator(image_folder=img_dir).run()

        assert summary["processed"] == 1000
        assert summary["failed"] == 0

    def test_1000_images_csv_row_count_correct(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _populate_folder(img_dir, 1000)

        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.return_value = make_result()

            Orchestrator(image_folder=img_dir).run()

        total = sum(
            sum(1 for _ in csv.DictReader(open(f, newline="", encoding="utf-8")))
            for f in img_dir.glob("extracted_data_2*.csv")
        )
        assert total == 1000

    def test_mocked_throughput_at_least_10_per_second(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _populate_folder(img_dir, 200)

        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.return_value = make_result()

            start = time.monotonic()
            summary = Orchestrator(image_folder=img_dir).run()
            elapsed = time.monotonic() - start

        rate = summary["processed"] / elapsed
        assert rate >= 10, f"Mocked throughput {rate:.1f} img/s should be ≥10."


class TestCSVRolloverUnderLoad:
    def test_60000_rows_across_two_files(self, tmp_path):
        writer = CSVWriter(tmp_path)
        for i in range(60_000):
            writer.write_record({"filename": f"img_{i}.jpeg", "name": f"User {i}"})

        files = sorted(tmp_path.glob("extracted_data_2*.csv"))
        assert len(files) >= 2

        totals = []
        for f in files:
            with open(f, newline="", encoding="utf-8") as fh:
                totals.append(sum(1 for _ in csv.DictReader(fh)))

        assert sum(totals) == 60_000
        assert all(t <= 50_000 for t in totals)


class TestResumeAfterCrash:
    def test_resume_skips_already_processed(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _populate_folder(img_dir, 100)

        call_log = []

        def extract_and_log(path):
            call_log.append(path.name)
            return make_result()

        # Run 1 — process 100
        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.side_effect = extract_and_log
            Orchestrator(image_folder=img_dir).run()

        assert len(call_log) == 100

        # Run 2 — tracker already has all files, nothing extracted
        call_log.clear()
        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.side_effect = extract_and_log
            summary2 = Orchestrator(image_folder=img_dir).run()

        assert summary2["processed"] == 0

    def test_mid_run_crash_resume_processes_remainder(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        all_images = _populate_folder(img_dir, 100)

        # Pre-populate tracker as if 50 images were done
        tracker = ResumeTracker(img_dir)
        for img in all_images[:50]:
            tracker.mark_processed(img.name)

        call_log = []

        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.side_effect = lambda p: (call_log.append(p.name), make_result())[1]

            orch = Orchestrator(image_folder=img_dir)
            orch.tracker = tracker
            summary = orch.run()

        assert len(call_log) == 50
        assert summary["processed"] == 50


class TestMemoryStability:
    def test_no_unbounded_growth_over_500_records(self, tmp_path):
        writer = CSVWriter(tmp_path)
        for i in range(500):
            writer.write_record({
                "filename": f"img_{i}.jpeg",
                "name": f"User {i}",
                "address": "A" * 200,
            })
        assert writer.current_row_count <= 500
        gc.collect()
