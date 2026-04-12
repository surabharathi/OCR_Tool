"""
tests/test_deduplication.py — Unit tests for content-based deduplication.
"""

import shutil
from pathlib import Path

import pytest

from core.deduplication import (
    compute_file_hash,
    get_processed_count,
    is_duplicate,
    mark_as_processed,
)
from tests.conftest import REAL_IMAGES


class TestComputeFileHash:
    def test_same_file_same_hash(self, tmp_path):
        f = tmp_path / "a.jpeg"
        f.write_bytes(b"hello world")
        assert compute_file_hash(f) == compute_file_hash(f)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.jpeg"
        f2 = tmp_path / "b.jpeg"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_hash_is_64_char_hex(self, tmp_path):
        f = tmp_path / "x.jpeg"
        f.write_bytes(b"data")
        h = compute_file_hash(f)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_missing_file_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            compute_file_hash(tmp_path / "ghost.jpeg")


class TestIsDuplicate:
    def test_new_file_is_not_duplicate(self, tmp_path):
        img = tmp_path / "test.jpeg"
        img.write_bytes(b"unique content 123")
        assert is_duplicate(img, tmp_path) is False

    def test_same_content_detected_as_duplicate(self, tmp_path):
        img = tmp_path / "original.jpeg"
        img.write_bytes(b"shared content")
        mark_as_processed(img, tmp_path)

        img2 = tmp_path / "copy.jpeg"
        img2.write_bytes(b"shared content")
        assert is_duplicate(img2, tmp_path) is True

    def test_different_name_same_content_is_duplicate(self, tmp_path):
        """Rename does not fool the hash-based deduplicator."""
        original = tmp_path / "form_001.jpeg"
        original.write_bytes(b"form data")
        mark_as_processed(original, tmp_path)

        renamed = tmp_path / "form_001_copy.jpeg"
        renamed.write_bytes(b"form data")
        assert is_duplicate(renamed, tmp_path) is True

    def test_same_name_different_content_not_duplicate(self, tmp_path):
        """Re-scanned file with new content should not be skipped."""
        img = tmp_path / "form.jpeg"
        img.write_bytes(b"version 1")
        mark_as_processed(img, tmp_path)

        img.write_bytes(b"version 2 - different scan")
        assert is_duplicate(img, tmp_path) is False


class TestMarkAsProcessed:
    def test_mark_increments_count(self, tmp_path):
        before = get_processed_count(tmp_path)
        img = tmp_path / "new.jpeg"
        img.write_bytes(b"fresh image")
        mark_as_processed(img, tmp_path)
        assert get_processed_count(tmp_path) == before + 1

    def test_mark_twice_does_not_duplicate_entry(self, tmp_path):
        img = tmp_path / "once.jpeg"
        img.write_bytes(b"data")
        mark_as_processed(img, tmp_path)
        mark_as_processed(img, tmp_path)
        assert get_processed_count(tmp_path) == 1

    def test_store_persists_across_instances(self, tmp_path):
        img = tmp_path / "persist.jpeg"
        img.write_bytes(b"persistent")
        mark_as_processed(img, tmp_path)

        # Simulate a new application run — the deduplicator re-reads from disk.
        assert is_duplicate(img, tmp_path) is True


class TestGetProcessedCount:
    def test_empty_folder_returns_zero(self, tmp_path):
        assert get_processed_count(tmp_path) == 0

    def test_count_reflects_marked_files(self, tmp_path):
        for i in range(5):
            img = tmp_path / f"img_{i}.jpeg"
            img.write_bytes(f"content {i}".encode())
            mark_as_processed(img, tmp_path)
        assert get_processed_count(tmp_path) == 5


class TestRealImageDeduplication:
    """Integration test using actual fixture images."""

    def test_real_images_are_not_duplicates_of_each_other(self, tmp_path):
        if len(REAL_IMAGES) < 2:
            pytest.skip("Need at least 2 fixture images.")

        # Copy two distinct images to temp folder.
        img_a = tmp_path / "a.jpeg"
        img_b = tmp_path / "b.jpeg"
        shutil.copy(REAL_IMAGES[0], img_a)
        shutil.copy(REAL_IMAGES[1], img_b)

        mark_as_processed(img_a, tmp_path)
        # img_b has different content — should NOT be flagged as duplicate.
        assert is_duplicate(img_b, tmp_path) is False

    def test_real_image_copy_detected_as_duplicate(self, tmp_path):
        if not REAL_IMAGES:
            pytest.skip("No fixture images available.")

        original = tmp_path / "original.jpeg"
        copy = tmp_path / "rescan.jpeg"
        shutil.copy(REAL_IMAGES[0], original)
        shutil.copy(REAL_IMAGES[0], copy)

        mark_as_processed(original, tmp_path)
        assert is_duplicate(copy, tmp_path) is True
