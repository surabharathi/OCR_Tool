"""
logging_setup.py — Centralised logging configuration.

Call ``configure_logging()`` once at application start-up.
Every module then does:  logger = logging.getLogger(__name__)
"""

import logging
import logging.handlers
from pathlib import Path

from config import LOG_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT


def configure_logging(log_dir: str | None = None, level: int = logging.DEBUG) -> None:
    """
    Set up root logger with:
      • RotatingFileHandler  → logs/<date>_app.log  (DEBUG and above)
      • StreamHandler        → console              (INFO and above)

    Args:
        log_dir: Directory for log files.  Defaults to config.LOG_DIR.
        level:   Minimum level captured to file.
    """
    log_directory = Path(log_dir or LOG_DIR)
    log_directory.mkdir(parents=True, exist_ok=True)

    log_file = log_directory / "app.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── File handler (rotating, DEBUG) ────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # ── Console handler (DEBUG+) ───────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Ensure console and file handlers are attached even if Flask/Werkzeug already configured logging.
    has_console = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
    has_file = any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root_logger.handlers)

    if not has_file:
        root_logger.addHandler(file_handler)
    if not has_console:
        root_logger.addHandler(console_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    logging.info("Logging configured. File: %s", log_file.resolve())
