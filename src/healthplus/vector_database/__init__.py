"""Layer 4: embedding generation and ChromaDB persistence adapters."""

from healthplus.vector_database.embeddings import EmbeddingService
from healthplus.vector_database.vector_store import VectorStore

__all__ = ["EmbeddingService", "VectorStore"]
