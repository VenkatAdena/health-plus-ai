"""Unit tests for ContextWindowManager and RagContext — pure logic, no I/O."""

from healthplus.application.context_manager import ContextWindowManager, RagContext
from healthplus.knowledge_base.models import SearchResult


def make_result(
    chunk_id: str,
    text: str = "Full body checkup costs 4999.",
    source: str = "Pricing.pdf",
    category: str = "pricing",
    page_number: int = 3,
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(
        text=text,
        score=score,
        source=source,
        category=category,
        page_number=page_number,
        chunk_id=chunk_id,
    )


def test_blocks_are_numbered_and_formatted() -> None:
    context = ContextWindowManager(max_chars=8000).assemble(
        [make_result("a"), make_result("b", source="FAQs.pdf", category="faqs", page_number=1)]
    )
    expected = (
        "[1] Pricing.pdf — page 3 (pricing)\n"
        "Full body checkup costs 4999.\n"
        "\n"
        "[2] FAQs.pdf — page 1 (faqs)\n"
        "Full body checkup costs 4999."
    )
    assert context.context_text == expected
    assert context.truncated is False


def test_duplicate_chunk_ids_are_removed_preserving_order() -> None:
    context = ContextWindowManager(max_chars=8000).assemble(
        [make_result("a"), make_result("a"), make_result("b")]
    )
    assert [r.chunk_id for r in context.sources] == ["a", "b"]
    # Numbering stays contiguous even after the duplicate is skipped.
    assert "[2]" in context.context_text
    assert "[3]" not in context.context_text


def test_truncates_when_budget_would_be_exceeded() -> None:
    results = [
        make_result("a", text="x" * 100),
        make_result("b", text="x" * 100),
        make_result("c", text="x" * 100),
    ]
    # Budget fits the first two blocks but not the third.
    manager = ContextWindowManager(max_chars=300)
    context = manager.assemble(results)
    assert [r.chunk_id for r in context.sources] == ["a", "b"]
    assert context.truncated is True
    assert len(context.context_text) <= 300


def test_sources_match_citation_order() -> None:
    results = [make_result("a"), make_result("b"), make_result("c")]
    context = ContextWindowManager(max_chars=8000).assemble(results)
    for i, result in enumerate(context.sources):
        assert f"[{i + 1}] {result.source}" in context.context_text
    assert [r.chunk_id for r in context.sources] == ["a", "b", "c"]


def test_empty_input_yields_empty_context() -> None:
    context = ContextWindowManager(max_chars=8000).assemble([])
    assert context.context_text == ""
    assert context.sources == []
    assert context.truncated is False
    assert context.is_empty


def test_is_empty_is_false_when_sources_exist() -> None:
    context = ContextWindowManager(max_chars=8000).assemble([make_result("a")])
    assert not context.is_empty


def test_single_block_larger_than_budget_yields_empty_truncated_context() -> None:
    context = ContextWindowManager(max_chars=50).assemble(
        [make_result("a", text="x" * 200)]
    )
    assert context.sources == []
    assert context.context_text == ""
    assert context.truncated is True


def test_rag_context_model_defaults() -> None:
    context = RagContext(context_text="", sources=[])
    assert context.truncated is False
    assert context.is_empty
