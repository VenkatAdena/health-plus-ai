"""Unit tests for KnowledgeBaseRetriever — a fake KB, no ChromaDB, no model."""

from types import SimpleNamespace

from healthplus.application.retriever import KnowledgeBaseRetriever
from healthplus.knowledge_base.models import SearchResult


def make_result(score: float, chunk_id: str = "c1") -> SearchResult:
    return SearchResult(
        text="Full body checkup costs 4999.",
        score=score,
        source="Pricing.pdf",
        category="pricing",
        page_number=3,
        chunk_id=chunk_id,
    )


class FakeKnowledgeBase:
    """Records the search call and returns a canned result list."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def search(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
        category: str | None = None,
    ) -> list[SearchResult]:
        self.calls.append({"query": query, "top_k": top_k, "category": category})
        return self.results


def make_settings(top_k: int = 5, min_score: float = 0.30) -> SimpleNamespace:
    return SimpleNamespace(retrieval_top_k=top_k, retrieval_min_score=min_score)


def test_drops_results_below_min_score() -> None:
    kb = FakeKnowledgeBase(
        [
            make_result(0.85, "a"),
            make_result(0.29, "b"),  # below the floor — noise
            make_result(0.42, "c"),
        ]
    )
    retriever = KnowledgeBaseRetriever(kb=kb, settings=make_settings(min_score=0.30))
    kept = retriever.retrieve("checkup price")
    assert [r.chunk_id for r in kept] == ["a", "c"]


def test_result_exactly_at_floor_is_kept() -> None:
    kb = FakeKnowledgeBase([make_result(0.30, "a")])
    retriever = KnowledgeBaseRetriever(kb=kb, settings=make_settings(min_score=0.30))
    assert len(retriever.retrieve("checkup price")) == 1


def test_top_k_defaults_to_settings_value() -> None:
    kb = FakeKnowledgeBase([])
    retriever = KnowledgeBaseRetriever(kb=kb, settings=make_settings(top_k=7))
    retriever.retrieve("checkup price")
    assert kb.calls[0]["top_k"] == 7


def test_explicit_top_k_overrides_settings() -> None:
    kb = FakeKnowledgeBase([])
    retriever = KnowledgeBaseRetriever(kb=kb, settings=make_settings(top_k=7))
    retriever.retrieve("checkup price", top_k=2)
    assert kb.calls[0]["top_k"] == 2


def test_category_filter_is_passed_through() -> None:
    kb = FakeKnowledgeBase([])
    retriever = KnowledgeBaseRetriever(kb=kb, settings=make_settings())
    retriever.retrieve("checkup price", category="pricing")
    assert kb.calls[0]["category"] == "pricing"


def test_all_results_dropped_returns_empty_list() -> None:
    kb = FakeKnowledgeBase([make_result(0.05, "a"), make_result(0.10, "b")])
    retriever = KnowledgeBaseRetriever(kb=kb, settings=make_settings(min_score=0.30))
    assert retriever.retrieve("unrelated question") == []
