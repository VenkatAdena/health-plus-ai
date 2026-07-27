"""Tests for cleanup performed between extraction and chunking."""

from healthplus.document_pipeline import TextCleaner


def test_normalizes_unicode_and_whitespace() -> None:
    raw = "MRI\u00a0 scan\t\tprice\r\n\r\n\r\n₹４５００"
    assert TextCleaner().clean(raw) == "MRI scan price\n\n₹4500"


def test_removes_soft_hyphens() -> None:
    assert TextCleaner().clean("appoint\u00adment") == "appointment"


def test_repairs_words_hyphenated_across_lines() -> None:
    assert TextCleaner().clean("depart-\nment details") == "department details"


def test_keeps_real_compound_words() -> None:
    assert TextCleaner().clean("Follow-up visit") == "Follow-up visit"


def test_preserves_paragraph_boundaries() -> None:
    assert TextCleaner().clean("First.\n\nSecond.") == "First.\n\nSecond."
