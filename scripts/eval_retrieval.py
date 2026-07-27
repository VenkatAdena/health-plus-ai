"""Retrieval-quality evaluation against the live knowledge base.

Runs the golden query set and reports, per query: the top-1 hit, the
expected category, and whether the expectation was met at top-1 and
top-3. Top-3 is the pass/fail metric (the RAG context window shows
Claude several blocks); top-1 is reported because watching it drift
is an early warning even while top-3 still passes.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --top-k 5

Exits 1 when the top-3 hit-rate drops below the floor, so this can gate
a CI pipeline or a pre-release check.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from healthplus.application import KnowledgeBaseService
from healthplus.config import get_settings
from healthplus.core import configure_logging

# The golden set. Some queries accept more than one category because the
# corpus legitimately answers them from several documents. This list is
# deliberately duplicated in tests/test_retrieval_quality.py — scripts/
# is not importable from tests, and the list is small enough that
# duplication beats a shared-module contortion. Keep the two in sync.
GOLDEN_SET: list[tuple[str, set[str]]] = [
    ("How much does an MRI scan cost?", {"pricing"}),
    ("How do I book an appointment?", {"sops", "faqs"}),
    ("Which doctors are cardiologists?", {"doctors"}),
    ("What health checkup packages are available?", {"health_packages"}),
    ("What is the refund policy?", {"policies"}),
    ("How long does it take to get test reports?", {"reports", "faqs"}),
    ("What blood tests do you offer?", {"test_catalog"}),
    ("What are the visiting hours for patients?", {"policies"}),
    ("Do you accept insurance or cashless payment?", {"policies"}),
    ("How much does a full body checkup package cost?", {"pricing", "health_packages"}),
]

HIT_RATE_FLOOR = 0.7


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against the golden query set"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many results to consider for the top-k hit metric (default 3)",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_directories()
    configure_logging(level=settings.log_level, log_dir=settings.log_dir)

    kb = KnowledgeBaseService()
    console = Console()

    table = Table(title=f"Retrieval quality — golden set ({len(GOLDEN_SET)} queries)")
    table.add_column("Query", max_width=42)
    table.add_column("Top-1 source", style="cyan")
    table.add_column("Top-1 cat", style="magenta")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Expected", style="magenta")
    table.add_column("Top-1", justify="center")
    table.add_column(f"Top-{args.top_k}", justify="center")

    top1_hits = 0
    topk_hits = 0
    for query, accepted in GOLDEN_SET:
        results = kb.search(query, top_k=args.top_k)
        best = results[0] if results else None
        top1_hit = best is not None and best.category in accepted
        topk_hit = any(r.category in accepted for r in results)
        top1_hits += int(top1_hit)
        topk_hits += int(topk_hit)

        table.add_row(
            query,
            best.source if best else "—",
            best.category if best else "—",
            f"{best.score:.3f}" if best else "—",
            "|".join(sorted(accepted)),
            "[green]HIT[/green]" if top1_hit else "[red]miss[/red]",
            "[green]HIT[/green]" if topk_hit else "[red]MISS[/red]",
        )

    console.print(table)

    total = len(GOLDEN_SET)
    topk_rate = topk_hits / total
    console.print(
        f"Top-1 hit-rate: {top1_hits}/{total} ({top1_hits / total:.0%})   "
        f"Top-{args.top_k} hit-rate: {topk_hits}/{total} ({topk_rate:.0%})   "
        f"floor: {HIT_RATE_FLOOR:.0%}"
    )

    if topk_rate < HIT_RATE_FLOOR:
        console.print(
            f"[red]FAIL[/red] top-{args.top_k} hit-rate is below the floor"
        )
        sys.exit(1)
    console.print("[green]PASS[/green] retrieval quality meets the floor")


if __name__ == "__main__":
    main()
