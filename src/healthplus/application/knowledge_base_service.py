"""Knowledge base facade.

The rest of the application (CLI today, Claude RAG chain and Streamlit UI
later) interacts with the knowledge base ONLY through this service. The
pipeline stages are injected, so tests can substitute fakes for any of them.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pydantic import BaseModel

from healthplus.config import Settings, get_settings
from healthplus.core.exceptions import HealthPlusError
from healthplus.document_pipeline import DocumentChunker, PDFLoader
from healthplus.knowledge_base.models import DocumentCategory, SearchResult
from healthplus.vector_database import EmbeddingService, VectorStore

logger = logging.getLogger(__name__)


class IngestionReport(BaseModel):
    source: str
    category: str
    doc_id: str
    pages: int
    chunks: int
    duration_seconds: float


class IngestionFailure(BaseModel):
    source: str
    error: str


class BatchIngestionReport(BaseModel):
    """Result of a resilient batch: one bad PDF does not stop the others."""

    succeeded: list[IngestionReport]
    failed: list[IngestionFailure]
    duration_seconds: float

    @property
    def is_successful(self) -> bool:
        return not self.failed


class KnowledgeBaseService:
    def __init__(
        self,
        loader: PDFLoader | None = None,
        chunker: DocumentChunker | None = None,
        embedder: EmbeddingService | None = None,
        store: VectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._loader = loader or PDFLoader()
        self._chunker = chunker or DocumentChunker(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        self._embedder = embedder or EmbeddingService(settings.embedding_model)
        self._store = store or VectorStore(
            persist_dir=settings.chroma_dir,
            collection_name=settings.chroma_collection,
        )

    def ingest_pdf(
        self, path: Path, category: DocumentCategory | None = None
    ) -> IngestionReport:
        """Full pipeline: load -> chunk -> embed -> store.

        `category` is explicit when the caller knows it (the upload UI asks
        the user); otherwise it is derived from the filename by convention.
        """
        start = time.perf_counter()
        document = self._loader.load(path, category=category)
        chunks = self._chunker.chunk(document)
        embeddings = self._embedder.embed_texts([c.text for c in chunks])
        self._store.upsert(chunks, embeddings)

        report = IngestionReport(
            source=document.metadata.source,
            category=document.metadata.category.value,
            doc_id=document.doc_id,
            pages=document.metadata.page_count,
            chunks=len(chunks),
            duration_seconds=round(time.perf_counter() - start, 2),
        )
        logger.info("Ingestion complete: %s", report.model_dump())
        return report

    def ingest_many(self, paths: list[Path]) -> BatchIngestionReport:
        """Ingest PDFs independently and collect successes and failures."""
        start = time.perf_counter()
        succeeded: list[IngestionReport] = []
        failed: list[IngestionFailure] = []

        for path in paths:
            try:
                succeeded.append(self.ingest_pdf(path))
            except HealthPlusError as exc:
                logger.error("Skipping %s: %s", path.name, exc)
                failed.append(IngestionFailure(source=path.name, error=str(exc)))

        return BatchIngestionReport(
            succeeded=succeeded,
            failed=failed,
            duration_seconds=round(time.perf_counter() - start, 2),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
        category: str | None = None,
    ) -> list[SearchResult]:
        """Semantic search over the knowledge base, optionally filtered."""
        logger.info(
            "Search: %r (top_k=%d, source=%s, category=%s)",
            query,
            top_k,
            source,
            category,
        )
        embedding = self._embedder.embed_query(query)
        return self._store.query(
            embedding, top_k=top_k, source=source, category=category
        )

    @property
    def chunk_count(self) -> int:
        return self._store.count()

    def category_counts(self) -> dict[str, int]:
        return self._store.category_counts()
