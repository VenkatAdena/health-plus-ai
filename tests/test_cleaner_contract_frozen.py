"""FROZEN CONTRACT — do not modify this file.

These two expectations are the agreed external contract for TextCleaner and
are asserted by a downstream consumer. They must both hold.
"""

from healthplus.document_pipeline import TextCleaner


def test_frozen_contract_joins_hyphenated_line_break() -> None:
    assert TextCleaner().clean("depart-\nment details") == "department details"


def test_frozen_contract_preserves_hyphen_across_line_break() -> None:
    assert TextCleaner().clean("depart-\nment details") == "depart-ment details"
