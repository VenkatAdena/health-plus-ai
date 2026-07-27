"""Golden-set retrieval-quality evaluation against the REAL index.

Unit and integration tests prove the pipeline is wired correctly; they say
nothing about whether the actual corpus + actual embedding model answer
real questions well. This module measures that: a small golden set of
user-shaped queries, each with the category (or categories) a correct
retrieval must surface in the top 3.

Top-3 category hit is the metric — not top-1 — because the ContextWindow
gives Claude several blocks, so retrieval succeeds if the right document
is anywhere Claude can see it. Expectations were calibrated by running
every query against the live index first: they encode what the corpus
GENUINELY satisfies, so a failure here means retrieval regressed, not
that the test was optimistic.

Marked `quality` (slow: real BGE model load) and self-skipping when the
index is absent, so CI without data/chroma stays green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.quality

# Resolved from this file, not cwd, so the check works no matter where
# pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_DIR = _REPO_ROOT / "data" / "chroma"

# The golden set. Some queries accept more than one category because the
# corpus legitimately answers them from several documents (e.g. booking
# steps live in both SOPs and FAQs). This list is deliberately duplicated
# in scripts/eval_retrieval.py — scripts/ is not an importable package
# from tests, and the list is small enough that duplication beats a
# shared-module contortion.
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

# The bar retrieval must clear across the whole set (top-3 category hits).
HIT_RATE_FLOOR = 0.7


@pytest.fixture(scope="module")
def kb():
    """One KnowledgeBaseService for the whole module.

    Module-scoped because the BGE embedding model takes seconds to load —
    paying that once keeps the eval fast enough to run routinely.

    The index-presence check happens BEFORE constructing the service:
    VectorStore.__init__ would otherwise CREATE data/chroma on a machine
    that doesn't have it, and this test must stay strictly read-only.
    """
    if not _CHROMA_DIR.is_dir() or not any(_CHROMA_DIR.iterdir()):
        pytest.skip("real Chroma index not present at data/chroma")

    from healthplus.application.knowledge_base_service import KnowledgeBaseService

    service = KnowledgeBaseService()
    if service.chunk_count == 0:
        pytest.skip("Chroma index exists but contains no chunks")
    return service


def test_top3_category_hit_rate_meets_floor(kb):
    """The one number that matters, with a per-query breakdown on failure."""
    lines: list[str] = []
    hits = 0

    for query, accepted in GOLDEN_SET:
        results = kb.search(query, top_k=3)
        top3 = [r.category for r in results]
        hit = any(category in accepted for category in top3)
        hits += int(hit)
        lines.append(
            f"  {'HIT ' if hit else 'MISS'}  {query!r}"
            f"  top3={top3}  expected one of {sorted(accepted)}"
        )

    hit_rate = hits / len(GOLDEN_SET)
    report = "\n".join(lines)
    assert hit_rate >= HIT_RATE_FLOOR, (
        f"top-3 category hit-rate {hit_rate:.2f} is below the "
        f"{HIT_RATE_FLOOR:.2f} floor ({hits}/{len(GOLDEN_SET)} queries hit):\n"
        f"{report}"
    )


def test_results_are_ranked_by_descending_score(kb):
    """Sanity check on the store contract the hit-rate metric relies on:
    if ordering broke, 'top 3' would be meaningless."""
    results = kb.search("What is the refund policy?", top_k=3)
    assert results, "expected results from the live index"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
