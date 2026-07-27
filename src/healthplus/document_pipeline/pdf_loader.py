"""PDF text extraction via PyMuPDF."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import fitz  # PyMuPDF

from healthplus.core.exceptions import DocumentLoadError, EmptyDocumentError
from healthplus.knowledge_base.models import (
    Document,
    DocumentCategory,
    DocumentMetadata,
    PageContent,
)
from healthplus.document_pipeline.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class PDFLoader:
    """Turns a PDF file on disk into a validated Document model."""

    def __init__(self, cleaner: TextCleaner | None = None) -> None:
        self._cleaner = cleaner or TextCleaner()

    def load(self, path: Path, category: DocumentCategory | None = None) -> Document:
        if not path.is_file():
            raise DocumentLoadError(f"File not found: {path}")

        # Category is explicit when the caller knows it (future admin upload
        # UI) and derived from the filename by convention otherwise.
        if category is None:
            try:
                category = DocumentCategory.from_filename(path.name)
            except ValueError as exc:
                raise DocumentLoadError(str(exc)) from exc

        file_bytes = path.read_bytes()
        # Content hash, not filename: the same file uploaded twice (or renamed)
        # maps to the same doc_id, which makes ingestion idempotent.
        doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]

        try:
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            raise DocumentLoadError(f"Cannot parse {path.name}: {exc}") from exc

        with pdf:
            pages = [
                PageContent(page_number=i + 1, text=self._cleaner.clean(page.get_text()))
                for i, page in enumerate(pdf)
            ]
            title = (pdf.metadata or {}).get("title") or None

        if not any(page.text for page in pages):
            raise EmptyDocumentError(
                f"{path.name} has no extractable text — likely a scanned "
                "document that requires OCR"
            )

        logger.info(
            "Loaded %s (doc_id=%s, category=%s, pages=%d)",
            path.name,
            doc_id,
            category.value,
            len(pages),
        )
        return Document(
            doc_id=doc_id,
            metadata=DocumentMetadata(
                source=path.name,
                category=category,
                title=title,
                page_count=len(pages),
            ),
            pages=pages,
        )
