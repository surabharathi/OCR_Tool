"""
tests/test_happy_path.py — Happy path tests for the single-engine pipeline.
"""

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from core.csv_writer import CSV_HEADERS, CSVWriter
from core.orchestrator import Orchestrator
from extractors.base_extractor import ExtractionResult
from extractors.olm_ocr_extractor import _compute_confidences, _parse_response, _sanitise_fields
from tests.conftest import (
    MOCK_CONFS_GOOD,
    MOCK_FIELDS_GOOD,
    REAL_IMAGES,
    make_result,
    make_failed_result,
)


class TestFixtures:
    def test_real_images_available(self):
        assert len(REAL_IMAGES) >= 1, "No fixture images found."

    def test_fixture_images_are_readable(self):
        for img in REAL_IMAGES:
            assert img.stat().st_size > 0


class TestExtractionResult:
    def test_avg_text_field_confidence(self):
        result = make_result()
        avg = result.avg_text_field_confidence
        assert 80.0 <= avg <= 95.0

    def test_failed_result_detected(self):
        assert make_failed_result().is_failed()

    def test_successful_result_not_failed(self):
        assert not make_result().is_failed()

    def test_repr_contains_engine(self):
        assert "olmocr" in repr(make_result())


class TestOlmOCRResponseParsing:
    def test_clean_json_parsed(self):
        raw = '{"app_no": "123", "name": "Test User", "mob_no": "9876543210"}'
        fields = _parse_response(raw)
        assert fields["app_no"] == "123"
        assert fields["name"] == "Test User"

    def test_json_with_markdown_fences_parsed(self):
        raw = '```json\n{"app_no": "456", "name": "Alice"}\n```'
        fields = _parse_response(raw)
        assert fields["app_no"] == "456"

    def test_json_with_preamble_parsed(self):
        raw = 'Here is the extracted data:\n{"app_no": "789", "name": "Bob"}'
        fields = _parse_response(raw)
        assert fields["app_no"] == "789"

    def test_invalid_json_raises(self):
        import json
        with pytest.raises(json.JSONDecodeError):
            _parse_response("this is not json at all !!!")

    def test_sanitise_strips_country_code(self):
        fields = {"mob_no": "919876543210", "name": "Test"}
        sanitised = _sanitise_fields(fields)
        assert sanitised["mob_no"] == "9876543210"

    def test_sanitise_none_becomes_blank(self):
        fields = {"mob_no": None, "name": "Test"}
        sanitised = _sanitise_fields(fields)
        assert sanitised["mob_no"] == "[BLANK]"


class TestConfidenceScoring:
    def test_valid_mobile_gets_high_confidence(self):
        confs = _compute_confidences({"mob_no": "9876543210"})
        assert confs["mob_no"] >= 90.0

    def test_invalid_mobile_gets_low_confidence(self):
        confs = _compute_confidences({"mob_no": "12345"})
        assert confs["mob_no"] < 70.0

    def test_valid_email_gets_high_confidence(self):
        confs = _compute_confidences({"email": "test@example.com"})
        assert confs["email"] >= 88.0

    def test_unreadable_gets_zero_confidence(self):
        confs = _compute_confidences({"name": "[UNREADABLE]"})
        assert confs["name"] == 0.0

    def test_blank_gets_mid_confidence(self):
        confs = _compute_confidences({"email": "[BLANK]"})
        assert confs["email"] == 50.0

    def test_valid_app_no_gets_high_confidence(self):
        confs = _compute_confidences({"app_no": "18608"})
        assert confs["app_no"] >= 95.0


class TestOrchestratorHappyPath:
    def test_all_images_processed(self, tmp_image_folder):
        events = []
        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.return_value = make_result()

            orch = Orchestrator(
                image_folder=tmp_image_folder,
                progress_callback=events.append,
            )
            summary = orch.run()

        assert summary["type"] == "scan_complete"
        assert summary["processed"] == len(REAL_IMAGES)
        assert summary["failed"] == 0

    def test_skips_duplicates(self, tmp_image_folder):
        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=True), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"

            orch = Orchestrator(image_folder=tmp_image_folder)
            summary = orch.run()

        assert summary["skipped_duplicates"] == len(REAL_IMAGES)
        assert summary["processed"] == 0

    def test_raises_when_ollama_unavailable(self, tmp_image_folder):
        with patch("core.orchestrator.OlmOCRExtractor") as MockClass:
            instance = MockClass.return_value
            instance.health_check.return_value = False

            orch = Orchestrator(image_folder=tmp_image_folder)
            with pytest.raises(RuntimeError, match="Ollama"):
                orch.run()

    def test_empty_folder_completes_cleanly(self, empty_image_folder):
        with patch("core.orchestrator.OlmOCRExtractor") as MockClass:
            instance = MockClass.return_value
            instance.health_check.return_value = True

            orch = Orchestrator(image_folder=empty_image_folder)
            summary = orch.run()

        assert summary["processed"] == 0
        assert summary["failed"] == 0


class TestCSVOutput:
    def test_csv_created_with_correct_headers(self, tmp_image_folder):
        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.return_value = make_result()

            Orchestrator(image_folder=tmp_image_folder).run()

        csvs = list(tmp_image_folder.glob("extracted_data_2*.csv"))
        assert len(csvs) >= 1
        with open(csvs[0], newline="", encoding="utf-8") as fh:
            assert set(csv.DictReader(fh).fieldnames or []) == set(CSV_HEADERS)

    def test_csv_row_count_matches_processed(self, tmp_image_folder):
        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.return_value = make_result()

            summary = Orchestrator(image_folder=tmp_image_folder).run()

        csvs = list(tmp_image_folder.glob("extracted_data_2*.csv"))
        with open(csvs[0], newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == summary["processed"]

    def test_csv_field_values_correct(self, tmp_image_folder):
        with patch("core.orchestrator.OlmOCRExtractor") as MockClass, \
             patch("core.orchestrator.is_duplicate", return_value=False), \
             patch("core.orchestrator.mark_as_processed"):
            instance = MockClass.return_value
            instance.health_check.return_value = True
            instance.engine_name = "olmocr:reducto/rolmocr"
            instance.extract.return_value = make_result()

            Orchestrator(image_folder=tmp_image_folder).run()

        csvs = list(tmp_image_folder.glob("extracted_data_2*.csv"))
        with open(csvs[0], newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                assert row["name"] == MOCK_FIELDS_GOOD["name"]
                assert row["mob_no"] == MOCK_FIELDS_GOOD["mob_no"]
                assert row["ocr_engine_used"] == "olmocr:reducto/rolmocr"
