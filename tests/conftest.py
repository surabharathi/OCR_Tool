"""
tests/conftest.py — Shared pytest fixtures and mock factories.

The OCR engine is now OlmOCRExtractor (local VLM via Ollama).
All tests mock the Ollama client so no real GPU or model is required.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from extractors.base_extractor import ExtractionResult

# ── Paths ─────────────────────────────────────────────────────────────────────
FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_IMAGES = sorted(FIXTURES_DIR.glob("*.jpeg")) + sorted(FIXTURES_DIR.glob("*.jpg"))

# ── Canonical mock field sets ─────────────────────────────────────────────────

MOCK_FIELDS_GOOD = {
    "app_no":                "18608",
    "name":                  "Veena V",
    "address":               "Malleshwaram, Bengaluru - 560003",
    "mob_no":                "9980512902",
    "email":                 "hvveen@yahoo.com",
    "q1_chanted_before":     "Yes",
    "q2_familiarity_level":  "Can chant confidently without guidance",
    "q3_formal_training":    "No",
    "q4_training_mode":      "[BLANK]",
    "q5_preferred_language": "[BLANK]",
    "q6_comments":           "[BLANK]",
}

MOCK_CONFS_GOOD = {
    "app_no":   95.0,
    "name":     85.0,
    "address":  85.0,
    "mob_no":   92.0,
    "email":    90.0,
    "q1_chanted_before":     88.0,
    "q2_familiarity_level":  88.0,
    "q3_formal_training":    88.0,
    "q4_training_mode":      50.0,
    "q5_preferred_language": 50.0,
    "q6_comments":           50.0,
}

MOCK_FIELDS_FAILED = {
    "app_no":  "[UNREADABLE]",
    "name":    "[UNREADABLE]",
    "address": "[UNREADABLE]",
    "mob_no":  "[UNREADABLE]",
    "email":   "[BLANK]",
    "q1_chanted_before":     "[NEEDS_REVIEW]",
    "q2_familiarity_level":  "[NEEDS_REVIEW]",
    "q3_formal_training":    "[NEEDS_REVIEW]",
    "q4_training_mode":      "[NEEDS_REVIEW]",
    "q5_preferred_language": "[NEEDS_REVIEW]",
    "q6_comments":           "[BLANK]",
}

MOCK_CONFS_FAILED = {k: 0.0 for k in MOCK_FIELDS_FAILED}


# ── Factory helpers ───────────────────────────────────────────────────────────

def make_result(fields=None, confs=None, engine="olmocr:reducto/rolmocr",
                error=None) -> ExtractionResult:
    return ExtractionResult(
        fields=fields or MOCK_FIELDS_GOOD,
        confidences=confs or MOCK_CONFS_GOOD,
        engine=engine,
        raw_text='{"app_no": "18608"}',
        error=error,
    )


def make_failed_result(reason="Illegible image") -> ExtractionResult:
    return ExtractionResult(
        fields={}, confidences={}, engine="olmocr:reducto/rolmocr",
        error=reason,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_image_folder(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for img in REAL_IMAGES:
        shutil.copy(img, img_dir / img.name)
    return img_dir


@pytest.fixture()
def empty_image_folder(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    return folder


@pytest.fixture()
def tmp_csv_folder(tmp_path):
    folder = tmp_path / "output"
    folder.mkdir()
    return folder


@pytest.fixture()
def mock_olm_extractor_good():
    """OlmOCRExtractor that always succeeds with good fields."""
    with patch("core.orchestrator.OlmOCRExtractor") as MockClass:
        instance = MockClass.return_value
        instance.health_check.return_value = True
        instance.engine_name = "olmocr:reducto/rolmocr"
        instance.extract.return_value = make_result()
        yield instance


@pytest.fixture()
def mock_olm_extractor_failed():
    """OlmOCRExtractor that always returns an error result."""
    with patch("core.orchestrator.OlmOCRExtractor") as MockClass:
        instance = MockClass.return_value
        instance.health_check.return_value = True
        instance.engine_name = "olmocr:reducto/rolmocr"
        instance.extract.return_value = make_failed_result()
        yield instance
