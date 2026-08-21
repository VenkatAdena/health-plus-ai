"""PHI (Protected Health Information) redaction utilities.

Masks patient identifiers in text to prevent raw PHI from reaching LLM clients
or being stored in conversation history. Supports:
- Medical Record Numbers (MRN-XXXXXX)
- 10-digit phone numbers
- Email addresses
- Dates of birth (MM/DD/YYYY and YYYY-MM-DD formats)
"""

from __future__ import annotations

import re
from typing import NamedTuple


class RedactionResult(NamedTuple):
    """Result of PHI redaction."""

    redacted_text: str
    """Text with identifiers replaced by [REDACTED_TYPE] placeholders."""
    identifier_types: list[str]
    """List of identifier kinds found (e.g., ['MRN', 'PHONE', 'EMAIL'])."""


def redact_phi(text: str) -> RedactionResult:
    """Redact Protected Health Information from text.

    Identifies and masks:
    - MRN: Medical Record Numbers (MRN-XXXXXX pattern)
    - PHONE: 10-digit phone numbers (XXX-XXX-XXXX or XXXXXXXXXX)
    - EMAIL: Email addresses
    - DOB: Dates of birth (MM/DD/YYYY or YYYY-MM-DD)

    Args:
        text: Input text potentially containing PHI.

    Returns:
        RedactionResult with redacted text and list of identifier types found.
    """
    if not text:
        return RedactionResult(redacted_text=text, identifier_types=[])

    redacted = text
    found_types = set()

    # Pattern 1: MRN (Medical Record Number) - MRN-XXXXXX
    mrn_pattern = r"MRN-\d{6}(?!\d)"
    if re.search(mrn_pattern, redacted):
        found_types.add("MRN")
        redacted = re.sub(mrn_pattern, "[REDACTED_MRN]", redacted)

    # Pattern 2: Phone numbers - XXX-XXX-XXXX or XXXXXXXXXX (10 digits)
    phone_pattern = r"(?:\d{3}-\d{3}-\d{4}|\b\d{10}\b)"
    if re.search(phone_pattern, redacted):
        found_types.add("PHONE")
        redacted = re.sub(phone_pattern, "[REDACTED_PHONE]", redacted)

    # Pattern 3: Email addresses
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    if re.search(email_pattern, redacted):
        found_types.add("EMAIL")
        redacted = re.sub(email_pattern, "[REDACTED_EMAIL]", redacted)

    # Pattern 4: Dates of birth - MM/DD/YYYY or YYYY-MM-DD
    dob_pattern = r"(?:\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})"
    if re.search(dob_pattern, redacted):
        found_types.add("DOB")
        redacted = re.sub(dob_pattern, "[REDACTED_DOB]", redacted)

    return RedactionResult(
        redacted_text=redacted, identifier_types=sorted(list(found_types))
    )
