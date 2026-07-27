"""Domain models for the knowledge base.

These are the contracts between pipeline stages: the loader produces a
Document, the chunker turns it into Chunks, the vector store returns
SearchResults. Because they are Pydantic models, malformed data fails
loudly at the stage boundary instead of corrupting the store silently.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class DocumentCategory(StrEnum):
    """The content taxonomy of the knowledge base.

    Every document belongs to exactly one category. Categories power
    metadata-filtered search today and UI filters / routing later.
    Adding a category is a deliberate act: extend this enum, and the
    whole pipeline (validation included) picks it up.
    """

    SOPS = "sops"
    DOCTORS = "doctors"
    TEST_CATALOG = "test_catalog"
    PRICING = "pricing"
    FAQS = "faqs"
    HEALTH_PACKAGES = "health_packages"
    REPORTS = "reports"
    POLICIES = "policies"

    @classmethod
    def from_filename(cls, filename: str) -> "DocumentCategory":
        """Resolve a category from a filename by convention (Pricing.pdf -> pricing)."""
        stem = Path(filename).stem.lower().replace("-", "_").replace(" ", "_")
        try:
            return cls(stem)
        except ValueError:
            allowed = ", ".join(c.value for c in cls)
            raise ValueError(
                f"Cannot derive a category from {filename!r} — file stem must "
                f"be one of: {allowed}"
            ) from None


class PageContent(BaseModel):
    """Text of a single page. Page numbers are 1-based, as a human cites them."""

    page_number: int = Field(ge=1)
    text: str


class DocumentMetadata(BaseModel):
    source: str
    category: DocumentCategory
    title: str | None = None
    page_count: int = Field(ge=1)
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Document(BaseModel):
    """A fully loaded document, before chunking."""

    doc_id: str  # content hash — identical files dedupe to the same id
    metadata: DocumentMetadata
    pages: list[PageContent]


class Chunk(BaseModel):
    """The retrieval unit. Metadata here is what makes citations possible later."""

    chunk_id: str  # deterministic — re-ingesting upserts instead of duplicating
    doc_id: str
    text: str
    source: str
    category: DocumentCategory
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)


class SearchResult(BaseModel):
    """One semantic-search hit, ready to be shown or cited."""

    text: str
    score: float  # cosine similarity: 1.0 = identical meaning
    source: str
    category: str
    page_number: int
    chunk_id: str
