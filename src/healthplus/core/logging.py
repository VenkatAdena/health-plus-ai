"""Centralized logging configuration.

Call `configure_logging()` exactly once at application startup.
Every module then obtains its logger the standard way:

    import logging
    logger = logging.getLogger(__name__)

Two destinations:
- Console: Rich handler, human-friendly, great during development.
- File:    rotating plain-text log, the persistent record a production
           system ships to a log aggregator.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_FILE_NAME = "healthplus.log"
_MAX_BYTES = 5 * 1024 * 1024  # rotate after 5 MB
_BACKUP_COUNT = 3


def configure_logging(level: str = "INFO", log_dir: Path = Path("logs")) -> None:
    """Configure the root logger with console + rotating file handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)

    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    file_handler = RotatingFileHandler(
        log_dir / _LOG_FILE_NAME,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # idempotent: re-running never duplicates handlers
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Third-party HTTP libraries flood INFO with request noise.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
