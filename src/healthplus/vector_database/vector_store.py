"""ChromaDB adapter.

This is the ONLY module in the codebase allowed to import chromadb.
Everything else talks to VectorStore, so replacing Chroma with pgvector,
Pinecone, or Qdrant later touches exactly one file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from healthplus.core.exceptions import KnowledgeBaseError
from healthplus.knowledge_base.models import Chunk, SearchResult

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Vector store ready: collection='%s' (%d chunks) at %s",
            collection_name,
            self._collection.count(),
            persist_dir,
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise KnowledgeBaseError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must be the same length"
            )
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "category": c.category.value,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )
        logger.info("Upserted %d chunks (collection now %d)", len(chunks), self.count())

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        source: str | None = None,
        category: str | None = None,
    ) -> list[SearchResult]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=self._build_where(source=source, category=category),
        )
        hits: list[SearchResult] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                SearchResult(
                    text=text,
                    # Chroma returns cosine *distance*; similarity = 1 - distance.
                    score=round(1.0 - distance, 4),
                    source=str(meta["source"]),
                    category=str(meta.get("category", "unknown")),
                    page_number=int(meta["page_number"]),
                    chunk_id=chunk_id,
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    def category_counts(self) -> dict[str, int]:
        """Chunk count per category — the admin dashboard's first metric."""
        records = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in records["metadatas"]:
            key = str(meta.get("category", "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _build_where(**filters: str | None) -> dict | None:
        """Compose Chroma's where-clause from optional equality filters.

        Chroma wants {"field": value} for one condition but
        {"$and": [{...}, {...}]} for several — this hides that quirk.
        """
        clauses = [{field: value} for field, value in filters.items() if value]
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
