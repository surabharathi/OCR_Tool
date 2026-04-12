"""
core/deduplication.py — Content-based deduplication using SHA-256 hashes.

The hash store is a JSON file (config.HASH_STORE_FILE) written next to the
image folder so it persists across application restarts.
"""

import hashlib
import json
import logging
from pathlib import Path

from config import HASH_STORE_FILE

logger = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _store_path(image_folder: Path) -> Path:
    return image_folder / HASH_STORE_FILE


def _load_store(image_folder: Path) -> dict:
    """Load the hash → filename mapping from disk.  Returns empty dict on any error."""
    path = _store_path(image_folder)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load hash store (%s); starting fresh. Error: %s", path, exc)
        return {}


def _save_store(image_folder: Path, store: dict) -> None:
    """Persist the hash store to disk, logging any write failure."""
    path = _store_path(image_folder)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2)
    except OSError as exc:
        logger.error("Failed to save hash store to %s: %s", path, exc)


# ── Public API ────────────────────────────────────────────────────────────────


def compute_file_hash(file_path: Path) -> str:
    """Return the SHA-256 hex digest of *file_path*'s binary content.

    Raises:
        OSError: if the file cannot be read.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    logger.debug("Hash for %s: %s…", file_path.name, digest[:12])
    return digest


def is_duplicate(file_path: Path, image_folder: Path) -> bool:
    """Deduplication is disabled: always treat files as unique."""
    logger.debug("Deduplication disabled: %s will be processed", file_path.name)
    return False


def mark_as_processed(file_path: Path, image_folder: Path) -> None:
    """No-op when deduplication is disabled."""
    logger.debug("Deduplication disabled: not recording %s", file_path.name)


def get_processed_count(image_folder: Path) -> int:
    """Return how many unique files have been recorded as processed."""
    return len(_load_store(image_folder))
