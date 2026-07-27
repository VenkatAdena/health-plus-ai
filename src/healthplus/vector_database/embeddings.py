"""Text -> vector embedding adapter for the vector-database layer."""

from __future__ import annotations

import logging
import time

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wraps a SentenceTransformer model behind a two-method interface.

    The model is loaded lazily: constructing this service is free, and the
    multi-second model load only happens on first actual use. Swapping the
    embedding model is a config change, not a code change.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def _sentence_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model '%s' ...", self._model_name)
            start = time.perf_counter()
            self._model = self._load_model()
            logger.info(
                "Embedding model ready in %.1fs (dim=%d)",
                time.perf_counter() - start,
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    def _load_model(self) -> SentenceTransformer:
        # Cache-first: a cached model loads with zero network calls, which is
        # faster and keeps us working offline / behind corporate proxies.
        # Only fall back to downloading when the model isn't cached yet.
        try:
            return SentenceTransformer(self._model_name, local_files_only=True)
        except Exception:
            logger.info("Model not in local cache — downloading from the Hub")
            return SentenceTransformer(self._model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks (batch)."""
        # normalize_embeddings=True gives unit vectors, so cosine similarity
        # comparisons are meaningful and consistent.
        vectors = self._sentence_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return self.embed_texts([text])[0]
