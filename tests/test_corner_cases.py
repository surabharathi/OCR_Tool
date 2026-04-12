"""
tests/test_corner_cases.py — Corner and edge case tests (single-engine pipeline).
"""

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from core.csv_writer import CSVWriter
from core.orchestrator import Orchestrator
from extractors.base_extractor import ExtractionResult
from tests.conftest import REAL_IMAGES, make_result, make_failed_result


class TestBlankAndUnreadableFields:
    def test_blank_fields_written_as_sentinel(self, tmp_image_folder):
        fields = {**{k: v for k, v in make_result().fields.items()},
                  "email": "[BLANK]", "q6_comments": "[BLANK]"}

        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.return_value = make_result(fields=fields)

            Orchestrator(image_folder=tmp_image_folder).run()

        csvs = list(tmp_image_folder.glob("extracted_data_2*.csv"))
        with open(csvs[0], newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                assert row["email"] == "[BLANK]"

    def test_unreadable_field_has_zero_confidence(self):
        from extractors.olm_ocr_extractor import _compute_confidences
        confs = _compute_confidences({"name": "[UNREADABLE]"})
        assert confs["name"] == 0.0


class TestCorruptAndNonImageFiles:
    def test_non_image_file_ignored(self, tmp_image_folder):
        (tmp_image_folder / "readme.txt").write_text("not an image")

        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.return_value = make_result()

            summary = Orchestrator(image_folder=tmp_image_folder).run()

        assert summary["total"] == len(REAL_IMAGES)

    def test_corrupt_jpeg_logged_not_crashed(self, tmp_image_folder):
        (tmp_image_folder / "corrupt.jpeg").write_bytes(b"\xff\xd8\xff garbage")

        def extract_side_effect(path):
            if "corrupt" in path.name:
                return ExtractionResult(
                    fields={}, confidences={}, engine="olmocr",
                    error="Inference error: corrupt image"
                )
            return make_result()

        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.side_effect = extract_side_effect

            summary = Orchestrator(image_folder=tmp_image_folder).run()

        assert summary["failed"] >= 1
        assert summary["processed"] >= 1


class TestOllamaFailures:
    def test_extraction_error_marks_failed(self, tmp_image_folder):
        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.return_value = make_failed_result("Ollama timeout")

            summary = Orchestrator(image_folder=tmp_image_folder).run()

        assert summary["failed"] == len(REAL_IMAGES)
        assert summary["processed"] == 0

    def test_ollama_unavailable_raises_runtime_error(self, tmp_image_folder):
        with patch("core.orchestrator.OlmOCRExtractor") as MockClass:
            instance = MockClass.return_value
            instance.health_check.return_value = False

            orch = Orchestrator(image_folder=tmp_image_folder)
            with pytest.raises(RuntimeError, match="Ollama"):
                orch.run()


class TestCSVRollover:
    def test_rollover_creates_new_file(self, tmp_csv_folder):
        from config import CSV_MAX_ROWS
        writer = CSVWriter(tmp_csv_folder)
        first_file = writer.current_file
        writer.write_record({"filename": "x.jpeg"})
        first_file = writer.current_file
        writer._current_row_count = CSV_MAX_ROWS
        writer.write_record({"filename": "overflow.jpeg"})
        assert writer.current_file != first_file

    def test_three_rollovers(self, tmp_csv_folder):
        from config import CSV_MAX_ROWS
        writer = CSVWriter(tmp_csv_folder)
        seen = set()
        for _ in range(3):
            writer.write_record({"filename": "x.jpeg"})
            seen.add(str(writer.current_file))
            writer._current_row_count = CSV_MAX_ROWS
        writer.write_record({"filename": "last.jpeg"})
        seen.add(str(writer.current_file))
        assert len(seen) >= 3


class TestProgressCallback:
    def test_crashing_callback_does_not_abort_scan(self, tmp_image_folder):
        def bad_callback(evt):
            raise RuntimeError("callback bug")

        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.return_value = make_result()

            summary = Orchestrator(
                image_folder=tmp_image_folder,
                progress_callback=bad_callback,
            ).run()

        assert summary["processed"] == len(REAL_IMAGES)


class TestStopSignal:
    def test_stop_respected_promptly(self, tmp_image_folder):
        call_count = 0
        orch = None

        def extract_and_stop(path):
            nonlocal call_count
            call_count += 1
            orch.stop()
            return make_result()

        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.side_effect = extract_and_stop

            orch = Orchestrator(image_folder=tmp_image_folder)
            orch.run()

        assert call_count <= 5
