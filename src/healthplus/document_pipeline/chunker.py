"""Split documents into retrieval-sized chunks."""

from __future__ import annotations

import hashlib
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from healthplus.knowledge_base.models import Chunk, Document

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits page text into overlapping chunks, preserving page provenance.

    We chunk page-by-page (never across page boundaries) so every chunk has
    an unambiguous page number — the foundation of source citations.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Try to break at paragraph, then line, then sentence boundaries
            # before falling back to hard character cuts.
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in document.pages:
            if not page.text:
                continue
            for index, text in enumerate(self._splitter.split_text(page.text)):
                chunks.append(
                    Chunk(
                        chunk_id=self._chunk_id(
                            document.doc_id, page.page_number, index
                        ),
                        doc_id=document.doc_id,
                        text=text,
                        source=document.metadata.source,
                        category=document.metadata.category,
                        page_number=page.page_number,
                        chunk_index=index,
                    )
                )
        logger.info(
            "Chunked %s: %d pages -> %d chunks",
            document.metadata.source,
            document.metadata.page_count,
            len(chunks),
        )
        return chunks

    @staticmethod
    def _chunk_id(doc_id: str, page_number: int, index: int) -> str:
        # Deterministic: the same document always yields the same ids,
        # so re-ingestion upserts in place instead of creating duplicates.
        raw = f"{doc_id}:{page_number}:{index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
