"""Semantic search against the knowledge base.

Usage:
    python scripts/search.py "how much does an MRI cost"
    python scripts/search.py "refund policy" --category policies
    python scripts/search.py "cardiology timings" --source Departments.pdf
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from healthplus.config import get_settings
from healthplus.core import configure_logging
from healthplus.application import KnowledgeBaseService
from healthplus.knowledge_base import DocumentCategory


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the knowledge base")
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--source", default=None, help="Filter by source filename")
    parser.add_argument(
        "--category",
        default=None,
        choices=[c.value for c in DocumentCategory],
        help="Filter by content category",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_directories()
    configure_logging(level=settings.log_level, log_dir=settings.log_dir)

    kb = KnowledgeBaseService()
    results = kb.search(
        args.query, top_k=args.top_k, source=args.source, category=args.category
    )

    title = f'Results for: "{args.query}"'
    if args.category:
        title += f"  [filter: category={args.category}]"
    table = Table(title=title)
    table.add_column("Score", justify="right", style="green")
    table.add_column("Category", style="magenta")
    table.add_column("Source", style="cyan")
    table.add_column("Page", justify="right")
    table.add_column("Excerpt", max_width=60)

    for hit in results:
        excerpt = hit.text[:160].replace("\n", " ") + "…"
        table.add_row(
            str(hit.score), hit.category, hit.source, str(hit.page_number), excerpt
        )

    Console().print(table)


if __name__ == "__main__":
    main()
