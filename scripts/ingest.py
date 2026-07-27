"""Ingest PDF documents into the knowledge base.

Usage:  python scripts/ingest.py data/samples/*.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from healthplus.config import get_settings
from healthplus.core import configure_logging
from healthplus.application import KnowledgeBaseService

logger = logging.getLogger("healthplus.ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the knowledge base")
    parser.add_argument("paths", nargs="+", type=Path, help="PDF files to ingest")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_directories()
    configure_logging(level=settings.log_level, log_dir=settings.log_dir)

    kb = KnowledgeBaseService()

    table = Table(title="Ingestion Report")
    table.add_column("Source", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Doc ID")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Time (s)", justify="right")

    batch = kb.ingest_many(args.paths)
    for report in batch.succeeded:
        table.add_row(
            report.source,
            report.category,
            report.doc_id,
            str(report.pages),
            str(report.chunks),
            str(report.duration_seconds),
        )

    Console().print(table)
    if batch.failed:
        error_table = Table(title="Failed Documents")
        error_table.add_column("Source", style="red")
        error_table.add_column("Reason")
        for failure in batch.failed:
            error_table.add_row(failure.source, failure.error)
        Console().print(error_table)
    logger.info("Knowledge base now holds %d chunks", kb.chunk_count)
    logger.info("Chunks by category: %s", kb.category_counts())
    logger.info(
        "Batch complete: %d succeeded, %d failed in %.2fs",
        len(batch.succeeded),
        len(batch.failed),
        batch.duration_seconds,
    )
    return 1 if batch.failed else 0


if __name__ == "__main__":
    sys.exit(main())
