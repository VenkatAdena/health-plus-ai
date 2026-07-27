"""Unit tests for DefaultQueryProcessor — pure logic, no I/O."""

import pytest

from healthplus.application.query_processor import (
    MAX_QUERY_CHARS,
    DefaultQueryProcessor,
)


def test_collapses_internal_whitespace() -> None:
    processor = DefaultQueryProcessor()
    assert processor.process("what  is\tthe\n\nprice?") == "what is the price?"


def test_strips_leading_and_trailing_whitespace() -> None:
    processor = DefaultQueryProcessor()
    assert processor.process("   hello world   ") == "hello world"


def test_clean_query_passes_through_unchanged() -> None:
    processor = DefaultQueryProcessor()
    assert processor.process("hello world") == "hello world"


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError):
        DefaultQueryProcessor().process("")


def test_whitespace_only_query_is_rejected() -> None:
    with pytest.raises(ValueError):
        DefaultQueryProcessor().process("   \n\t  ")


def test_query_over_limit_is_rejected() -> None:
    with pytest.raises(ValueError):
        DefaultQueryProcessor().process("x" * (MAX_QUERY_CHARS + 1))


def test_query_at_limit_is_accepted() -> None:
    query = "x" * MAX_QUERY_CHARS
    assert DefaultQueryProcessor().process(query) == query


def test_length_is_checked_after_normalization() -> None:
    # Whitespace padding must not count against the limit.
    padded = "  " + "x" * MAX_QUERY_CHARS + "  "
    assert DefaultQueryProcessor().process(padded) == "x" * MAX_QUERY_CHARS
