"""
tests/test_resume.py — Unit tests for the ResumeTracker crash-recovery journal.
"""

import json
from pathlib import Path

import pytest

from core.resume_tracker import ResumeTracker


@pytest.fixture()
def tracker(tmp_path):
    return ResumeTracker(tmp_path)


class TestInitialState:
    def test_fresh_tracker_has_empty_lists(self, tracker):
        stats = tracker.get_stats()
        assert stats["processed"] == 0
        assert stats["skipped_duplicates"] == 0
        assert stats["failed"] == 0
        assert stats["pending_claude_review"] == 0
        assert stats["claude_reviewed"] == 0

    def test_progress_file_created_on_first_save(self, tmp_path):
        t = ResumeTracker(tmp_path)
        t.mark_processed("a.jpeg")
        progress_file = tmp_path / "progress.json"
        assert progress_file.exists()


class TestMarkProcessed:
    def test_mark_increments_count(self, tracker):
        tracker.mark_processed("form1.jpeg")
        assert tracker.get_stats()["processed"] == 1

    def test_double_mark_does_not_duplicate(self, tracker):
        tracker.mark_processed("form1.jpeg")
        tracker.mark_processed("form1.jpeg")
        assert tracker.get_stats()["processed"] == 1

    def test_is_already_handled_after_mark(self, tracker):
        tracker.mark_processed("form1.jpeg")
        assert tracker.is_already_handled("form1.jpeg") is True


class TestMarkSkippedDuplicate:
    def test_skipped_counted_separately(self, tracker):
        tracker.mark_skipped_duplicate("dup.jpeg")
        stats = tracker.get_stats()
        assert stats["skipped_duplicates"] == 1
        assert stats["processed"] == 0

    def test_skipped_file_is_already_handled(self, tracker):
        tracker.mark_skipped_duplicate("dup.jpeg")
        assert tracker.is_already_handled("dup.jpeg") is True


class TestMarkFailed:
    def test_failed_file_recorded_with_reason(self, tmp_path):
        tracker = ResumeTracker(tmp_path)
        tracker.mark_failed("bad.jpeg", "Tesseract crashed")
        stats = tracker.get_stats()
        assert stats["failed"] == 1

    def test_failed_file_not_marked_as_already_handled(self, tracker):
        """Failed files should be retried on resume — not skipped."""
        tracker.mark_failed("bad.jpeg", "error")
        # Failed files are NOT in the 'already handled' set so they get retried.
        assert tracker.is_already_handled("bad.jpeg") is False


class TestPendingClaudeQueue:
    def test_add_pending_shows_in_get_pending(self, tracker):
        tracker.add_pending_claude("low.jpeg", {"name": "[UNREADABLE]"}, 22.5)
        pending = tracker.get_pending_claude()
        assert len(pending) == 1
        assert pending[0]["file"] == "low.jpeg"
        assert pending[0]["avg_confidence"] == 22.5

    def test_pending_file_is_already_handled(self, tracker):
        tracker.add_pending_claude("low.jpeg", {}, 15.0)
        assert tracker.is_already_handled("low.jpeg") is True

    def test_duplicate_pending_not_added_twice(self, tracker):
        tracker.add_pending_claude("low.jpeg", {}, 15.0)
        tracker.add_pending_claude("low.jpeg", {}, 15.0)
        assert len(tracker.get_pending_claude()) == 1

    def test_mark_claude_reviewed_removes_from_pending(self, tracker):
        tracker.add_pending_claude("low.jpeg", {}, 15.0)
        tracker.mark_claude_reviewed("low.jpeg")
        assert len(tracker.get_pending_claude()) == 0
        assert tracker.get_stats()["claude_reviewed"] == 1


class TestPersistenceAcrossRestarts:
    def test_state_survives_reinstantiation(self, tmp_path):
        t1 = ResumeTracker(tmp_path)
        t1.mark_processed("img1.jpeg")
        t1.mark_skipped_duplicate("img2.jpeg")
        t1.add_pending_claude("img3.jpeg", {}, 30.0)

        # Simulate application restart.
        t2 = ResumeTracker(tmp_path)
        stats = t2.get_stats()
        assert stats["processed"] == 1
        assert stats["skipped_duplicates"] == 1
        assert stats["pending_claude_review"] == 1

    def test_corrupted_progress_file_starts_fresh(self, tmp_path):
        progress_file = tmp_path / "progress.json"
        progress_file.write_text("{ invalid json !!!")

        # Should not raise — starts fresh.
        t = ResumeTracker(tmp_path)
        assert t.get_stats()["processed"] == 0


class TestReset:
    def test_reset_clears_all_state(self, tracker):
        tracker.mark_processed("a.jpeg")
        tracker.mark_skipped_duplicate("b.jpeg")
        tracker.add_pending_claude("c.jpeg", {}, 10.0)
        tracker.reset()
        stats = tracker.get_stats()
        assert all(v == 0 for v in stats.values())

    def test_is_already_handled_false_after_reset(self, tracker):
        tracker.mark_processed("a.jpeg")
        tracker.reset()
        assert tracker.is_already_handled("a.jpeg") is False


class TestThreadSafety:
    def test_concurrent_marks_do_not_corrupt_state(self, tmp_path):
        """Multiple threads marking different files must not lose entries."""
        import threading

        tracker = ResumeTracker(tmp_path)
        n = 50
        errors = []

        def worker(idx):
            try:
                tracker.mark_processed(f"img_{idx:04d}.jpeg")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert tracker.get_stats()["processed"] == n
