"""Unit tests for PHI redaction functionality."""

from __future__ import annotations

import pytest

from healthplus.core.phi_redaction import redact_phi


class TestMRNRedaction:
    """Tests for Medical Record Number (MRN) redaction."""

    def test_redacts_single_mrn(self) -> None:
        result = redact_phi("Patient MRN-123456 admitted today")
        assert result.redacted_text == "Patient [REDACTED_MRN] admitted today"
        assert "MRN" in result.identifier_types

    def test_redacts_multiple_mrns(self) -> None:
        result = redact_phi("MRN-111111 and MRN-222222 are different")
        assert result.redacted_text == "[REDACTED_MRN] and [REDACTED_MRN] are different"
        assert result.identifier_types.count("MRN") == 1  # deduplicated

    def test_ignores_invalid_mrn_format(self) -> None:
        result = redact_phi("MRN-12345 is too short, MRN-1234567 is too long")
        assert result.redacted_text == "MRN-12345 is too short, MRN-1234567 is too long"
        assert "MRN" not in result.identifier_types


class TestPhoneRedaction:
    """Tests for phone number redaction."""

    def test_redacts_formatted_phone(self) -> None:
        result = redact_phi("Call me at 555-123-4567 tomorrow")
        assert result.redacted_text == "Call me at [REDACTED_PHONE] tomorrow"
        assert "PHONE" in result.identifier_types

    def test_redacts_unformatted_phone(self) -> None:
        result = redact_phi("My number is 5551234567")
        assert result.redacted_text == "My number is [REDACTED_PHONE]"
        assert "PHONE" in result.identifier_types

    def test_redacts_multiple_phones(self) -> None:
        result = redact_phi("555-123-4567 or 666-789-0123")
        assert result.redacted_text == "[REDACTED_PHONE] or [REDACTED_PHONE]"
        assert result.identifier_types.count("PHONE") == 1  # deduplicated

    def test_ignores_short_numbers(self) -> None:
        result = redact_phi("Call 911 or 411 for help")
        assert result.redacted_text == "Call 911 or 411 for help"
        assert "PHONE" not in result.identifier_types

    def test_ignores_numbers_in_middle_of_word(self) -> None:
        result = redact_phi("The code is abc1234567def")
        assert result.redacted_text == "The code is abc1234567def"
        assert "PHONE" not in result.identifier_types


class TestEmailRedaction:
    """Tests for email address redaction."""

    def test_redacts_single_email(self) -> None:
        result = redact_phi("Contact john.doe@example.com for details")
        assert result.redacted_text == "Contact [REDACTED_EMAIL] for details"
        assert "EMAIL" in result.identifier_types

    def test_redacts_multiple_emails(self) -> None:
        result = redact_phi("Email alice@test.org or bob@test.com")
        assert result.redacted_text == "Email [REDACTED_EMAIL] or [REDACTED_EMAIL]"
        assert result.identifier_types.count("EMAIL") == 1  # deduplicated

    def test_redacts_email_with_plus_addressing(self) -> None:
        result = redact_phi("Send to patient+notes@hospital.org")
        assert result.redacted_text == "Send to [REDACTED_EMAIL]"
        assert "EMAIL" in result.identifier_types

    def test_ignores_invalid_email_format(self) -> None:
        result = redact_phi("Not an email: @example.com or user@")
        assert result.redacted_text == "Not an email: @example.com or user@"
        assert "EMAIL" not in result.identifier_types


class TestDOBRedaction:
    """Tests for Date of Birth redaction."""

    def test_redacts_dob_slash_format(self) -> None:
        result = redact_phi("Patient DOB: 01/15/1985")
        assert result.redacted_text == "Patient DOB: [REDACTED_DOB]"
        assert "DOB" in result.identifier_types

    def test_redacts_dob_dash_format(self) -> None:
        result = redact_phi("Born on 1985-01-15")
        assert result.redacted_text == "Born on [REDACTED_DOB]"
        assert "DOB" in result.identifier_types

    def test_redacts_multiple_dates(self) -> None:
        result = redact_phi("DOB: 12/25/1990 and admission: 03/15/2024")
        assert result.redacted_text == "DOB: [REDACTED_DOB] and admission: [REDACTED_DOB]"
        assert result.identifier_types.count("DOB") == 1  # deduplicated

    def test_ignores_invalid_date_format(self) -> None:
        result = redact_phi("Date 1985/01/15 or 01-15-1985 are not standard")
        assert result.redacted_text == "Date 1985/01/15 or 01-15-1985 are not standard"
        assert "DOB" not in result.identifier_types


class TestMultipleIdentifiers:
    """Tests for text with multiple identifier types."""

    def test_redacts_all_identifier_types(self) -> None:
        text = (
            "Patient MRN-654321 with DOB 06/20/1975 "
            "can be reached at 555-987-6543 or jane.smith@clinic.org"
        )
        result = redact_phi(text)
        assert "[REDACTED_MRN]" in result.redacted_text
        assert "[REDACTED_DOB]" in result.redacted_text
        assert "[REDACTED_PHONE]" in result.redacted_text
        assert "[REDACTED_EMAIL]" in result.redacted_text
        assert set(result.identifier_types) == {"DOB", "EMAIL", "MRN", "PHONE"}

    def test_identifier_types_sorted(self) -> None:
        text = "Email: test@example.com, Phone: 555-123-4567, MRN-999999"
        result = redact_phi(text)
        assert result.identifier_types == ["EMAIL", "MRN", "PHONE"]

    def test_complex_message_with_mixed_content(self) -> None:
        text = (
            "Follow-up for patient MRN-111111 (DOB: 03/10/1980). "
            "Contact at 555-111-2222 or patient@email.com. "
            "Appointment on 2024-05-15."
        )
        result = redact_phi(text)
        assert "[REDACTED_MRN]" in result.redacted_text
        assert "[REDACTED_DOB]" in result.redacted_text
        assert "[REDACTED_PHONE]" in result.redacted_text
        assert "[REDACTED_EMAIL]" in result.redacted_text
        assert "[REDACTED_DOB]" in result.redacted_text  # appointment date also redacted


class TestNoIdentifiers:
    """Tests for text with no identifiers."""

    def test_empty_string(self) -> None:
        result = redact_phi("")
        assert result.redacted_text == ""
        assert result.identifier_types == []

    def test_none_like_empty_string(self) -> None:
        result = redact_phi("   ")
        assert result.redacted_text == "   "
        assert result.identifier_types == []

    def test_text_with_no_phi(self) -> None:
        result = redact_phi("What are the symptoms of diabetes?")
        assert result.redacted_text == "What are the symptoms of diabetes?"
        assert result.identifier_types == []

    def test_text_with_numbers_but_no_phi(self) -> None:
        result = redact_phi("The patient took 2 tablets 3 times daily for 5 days")
        assert result.redacted_text == "The patient took 2 tablets 3 times daily for 5 days"
        assert result.identifier_types == []

    def test_text_with_similar_but_invalid_patterns(self) -> None:
        result = redact_phi(
            "MRN-12345 (too short), 555-1234 (too short), "
            "user@domain (no TLD), 1985/01/15 (wrong format)"
        )
        assert result.redacted_text == (
            "MRN-12345 (too short), 555-1234 (too short), "
            "user@domain (no TLD), 1985/01/15 (wrong format)"
        )
        assert result.identifier_types == []


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_consecutive_identifiers(self) -> None:
        result = redact_phi("MRN-123456 555-123-4567")
        assert result.redacted_text == "[REDACTED_MRN] [REDACTED_PHONE]"
        assert set(result.identifier_types) == {"MRN", "PHONE"}

    def test_identifier_at_start_of_text(self) -> None:
        result = redact_phi("MRN-123456 is the patient identifier")
        assert result.redacted_text == "[REDACTED_MRN] is the patient identifier"
        assert "MRN" in result.identifier_types

    def test_identifier_at_end_of_text(self) -> None:
        result = redact_phi("The patient email is test@example.com")
        assert result.redacted_text == "The patient email is [REDACTED_EMAIL]"
        assert "EMAIL" in result.identifier_types

    def test_identifier_with_punctuation(self) -> None:
        result = redact_phi("Call 555-123-4567. That's the number.")
        assert result.redacted_text == "Call [REDACTED_PHONE]. That's the number."
        assert "PHONE" in result.identifier_types

    def test_very_long_text_with_identifiers(self) -> None:
        long_text = (
            "This is a very long medical record. " * 100
            + "Patient MRN-123456 with DOB 01/01/1990 "
            + "can be reached at 555-123-4567 or test@example.com"
        )
        result = redact_phi(long_text)
        assert "[REDACTED_MRN]" in result.redacted_text
        assert "[REDACTED_DOB]" in result.redacted_text
        assert "[REDACTED_PHONE]" in result.redacted_text
        assert "[REDACTED_EMAIL]" in result.redacted_text
        assert set(result.identifier_types) == {"DOB", "EMAIL", "MRN", "PHONE"}

    def test_unicode_text_with_identifiers(self) -> None:
        result = redact_phi("Patient MRN-123456 with notes: नमस्ते")
        assert "[REDACTED_MRN]" in result.redacted_text
        assert "नमस्ते" in result.redacted_text
        assert "MRN" in result.identifier_types

    def test_newlines_and_special_chars_preserved(self) -> None:
        text = "MRN-123456\nPhone: 555-123-4567\nEmail: test@example.com"
        result = redact_phi(text)
        assert "\n" in result.redacted_text
        assert "[REDACTED_MRN]" in result.redacted_text
        assert "[REDACTED_PHONE]" in result.redacted_text
        assert "[REDACTED_EMAIL]" in result.redacted_text
