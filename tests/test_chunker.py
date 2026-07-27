"""Unit tests for DocumentChunker — pure logic, no I/O, runs in milliseconds."""

from healthplus.knowledge_base.models import (
    Document,
    DocumentCategory,
    DocumentMetadata,
    PageContent,
)
from healthplus.document_pipeline import DocumentChunker


def make_document() -> Document:
    return Document(
        doc_id="abc123",
        metadata=DocumentMetadata(
            source="Doctors.pdf",
            category=DocumentCategory.DOCTORS,
            page_count=2,
        ),
        pages=[
            PageContent(page_number=1, text="First paragraph.\n\n" + "alpha " * 300),
            PageContent(page_number=2, text="Second page content. " * 40),
        ],
    )


def test_chunks_preserve_page_numbers() -> None:
    chunks = DocumentChunker(chunk_size=500, chunk_overlap=50).chunk(make_document())
    assert chunks, "expected at least one chunk"
    assert {c.page_number for c in chunks} == {1, 2}
    assert all(c.source == "Doctors.pdf" for c in chunks)


def test_chunks_inherit_document_category() -> None:
    chunks = DocumentChunker(chunk_size=500, chunk_overlap=50).chunk(make_document())
    assert all(c.category == DocumentCategory.DOCTORS for c in chunks)


def test_chunk_ids_are_deterministic() -> None:
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    first = [c.chunk_id for c in chunker.chunk(make_document())]
    second = [c.chunk_id for c in chunker.chunk(make_document())]
    assert first == second


def test_chunk_ids_are_unique() -> None:
    chunks = DocumentChunker(chunk_size=300, chunk_overlap=30).chunk(make_document())
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_pages_are_skipped() -> None:
    doc = Document(
        doc_id="empty1",
        metadata=DocumentMetadata(
            source="FAQs.pdf",
            category=DocumentCategory.FAQS,
            page_count=1,
        ),
        pages=[PageContent(page_number=1, text="")],
    )
    assert DocumentChunker(chunk_size=500, chunk_overlap=50).chunk(doc) == []


def test_chunks_respect_max_size() -> None:
    chunks = DocumentChunker(chunk_size=200, chunk_overlap=20).chunk(make_document())
    assert all(len(c.text) <= 200 for c in chunks)
