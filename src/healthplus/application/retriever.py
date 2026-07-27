"""Retrieval with a relevance floor.

The vector store always returns the top_k nearest chunks — even when the
nearest chunk is barely related to the question. Similarity is relative;
usefulness is absolute. This retriever applies the configured minimum
score (settings.retrieval_min_score) so weak matches never reach the
prompt: an LLM given noise as "context" will confidently cite the noise.
"""

from __future__ import annotations

import logging

from healthplus.application.knowledge_base_service import KnowledgeBaseService
from healthplus.config import Settings, get_settings
from healthplus.knowledge_base.models import SearchResult

logger = logging.getLogger(__name__)


class KnowledgeBaseRetriever:
    """Fetches search hits and keeps only the ones worth citing."""

    def __init__(
        self,
        kb: KnowledgeBaseService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._kb = kb or KnowledgeBaseService(settings=self._settings)

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Search the knowledge base and drop hits below the relevance floor."""
        k = top_k or self._settings.retrieval_top_k
        results = self._kb.search(query, top_k=k, category=category)

        floor = self._settings.retrieval_min_score
        kept = [result for result in results if result.score >= floor]

        logger.info(
            "Retrieval: kept %d of %d result(s); dropped %d below min_score=%.2f",
            len(kept),
            len(results),
            len(results) - len(kept),
            floor,
        )
        return kept
