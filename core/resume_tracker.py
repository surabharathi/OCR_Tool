"""
core/resume_tracker.py — JSON-backed progress journal.

Tracks which files have been processed/skipped/failed so that interrupted
runs can resume without re-processing already-completed images.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import PROGRESS_FILE

logger = logging.getLogger(__name__)


class ResumeTracker:
    """Thread-safe progress journal stored as a JSON file on disk.

    Schema of the JSON file::

        {
          "started_at": "<ISO datetime>",
          "processed":  ["file1.jpg", ...],
          "skipped_duplicates": ["file2.jpg", ...],
          "failed": [{"file": "f.jpg", "reason": "...", "at": "..."}],
          "pending_claude_review": [
              {"file": "f.jpg", "tesseract_fields": {...}, "avg_confidence": 45.2}
          ],
          "claude_reviewed": ["f.jpg", ...]
        }
    """

    def __init__(self, image_folder: Path) -> None:
        self._path = image_folder / PROGRESS_FILE
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                logger.info("Resumed progress from %s", self._path)
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Cannot read progress file (%s); starting fresh.", exc)
        return {
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "processed": [],
            "skipped_duplicates": [],
            "failed": [],
            "pending_claude_review": [],
            "claude_reviewed": [],
        }

    def _save(self) -> None:
        """Persist current state.  Caller must hold self._lock."""
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
        except OSError as exc:
            logger.error("Failed to save progress file: %s", exc)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def is_already_handled(self, filename: str) -> bool:
        """Return True when the file has been processed, skipped, or is pending Claude."""
        with self._lock:
            pending_files = [e["file"] for e in self._data["pending_claude_review"]]
            return (
                filename in self._data["processed"]
                or filename in self._data["skipped_duplicates"]
                or filename in pending_files
                or filename in self._data["claude_reviewed"]
            )

    def get_pending_claude(self) -> list[dict]:
        with self._lock:
            return list(self._data["pending_claude_review"])

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "processed": len(self._data["processed"]),
                "skipped_duplicates": len(self._data["skipped_duplicates"]),
                "failed": len(self._data["failed"]),
                "pending_claude_review": len(self._data["pending_claude_review"]),
                "claude_reviewed": len(self._data["claude_reviewed"]),
            }

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def mark_processed(self, filename: str) -> None:
        with self._lock:
            if filename not in self._data["processed"]:
                self._data["processed"].append(filename)
            # Remove any prior failed entry — file succeeded on retry.
            self._data["failed"] = [
                e for e in self._data["failed"] if e["file"] != filename
            ]
            self._save()

    def mark_skipped_duplicate(self, filename: str) -> None:
        with self._lock:
            if filename not in self._data["skipped_duplicates"]:
                self._data["skipped_duplicates"].append(filename)
            self._save()

    def mark_failed(self, filename: str, reason: str) -> None:
        with self._lock:
            # Remove any previous entry for this file so re-runs don't duplicate.
            self._data["failed"] = [
                e for e in self._data["failed"] if e["file"] != filename
            ]
            entry = {
                "file": filename,
                "reason": reason,
                "at": datetime.now(tz=timezone.utc).isoformat(),
            }
            self._data["failed"].append(entry)
            self._save()

    def add_pending_claude(
        self, filename: str, tesseract_fields: dict, avg_confidence: float
    ) -> None:
        """Enqueue a file for user-approved Claude review."""
        with self._lock:
            existing = [e["file"] for e in self._data["pending_claude_review"]]
            if filename not in existing:
                self._data["pending_claude_review"].append(
                    {
                        "file": filename,
                        "tesseract_fields": tesseract_fields,
                        "avg_confidence": round(avg_confidence, 1),
                    }
                )
            self._save()

    def mark_claude_reviewed(self, filename: str) -> None:
        with self._lock:
            if filename not in self._data["claude_reviewed"]:
                self._data["claude_reviewed"].append(filename)
            # Remove from pending queue
            self._data["pending_claude_review"] = [
                e for e in self._data["pending_claude_review"] if e["file"] != filename
            ]
            self._save()

    def reset(self) -> None:
        """Wipe all progress — use only when the user explicitly requests a fresh run."""
        with self._lock:
            self._data = {
                "started_at": datetime.now(tz=timezone.utc).isoformat(),
                "processed": [],
                "skipped_duplicates": [],
                "failed": [],
                "pending_claude_review": [],
                "claude_reviewed": [],
            }
            self._save()
        logger.info("Progress tracker reset.")
