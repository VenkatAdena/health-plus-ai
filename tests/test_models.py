"""Unit tests for domain models — the category taxonomy contract."""

import pytest

from healthplus.knowledge_base import DocumentCategory


def test_category_resolves_from_filename() -> None:
    assert DocumentCategory.from_filename("Pricing.pdf") == DocumentCategory.PRICING
    assert DocumentCategory.from_filename("FAQs.pdf") == DocumentCategory.FAQS
    assert DocumentCategory.from_filename("doctors.pdf") == DocumentCategory.DOCTORS


def test_category_resolution_is_case_insensitive() -> None:
    assert DocumentCategory.from_filename("POLICIES.pdf") == DocumentCategory.POLICIES


def test_category_resolution_normalizes_spaces_and_hyphens() -> None:
    assert (
        DocumentCategory.from_filename("Health Packages.pdf")
        == DocumentCategory.HEALTH_PACKAGES
    )
    assert (
        DocumentCategory.from_filename("Test-Catalog.pdf")
        == DocumentCategory.TEST_CATALOG
    )


def test_unknown_filename_raises_with_allowed_list() -> None:
    with pytest.raises(ValueError, match="pricing"):
        DocumentCategory.from_filename("RandomNotes.pdf")


def test_taxonomy_matches_architecture_diagram() -> None:
    # The public architecture diagram promises 8 content categories.
    assert len(DocumentCategory) == 8
