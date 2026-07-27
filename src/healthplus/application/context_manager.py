"""Context assembly under a character budget.

The retriever hands back scored chunks; this module turns them into the
single context block a prompt can actually use. Two constraints drive the
design: the prompt has a finite budget (settings.context_max_chars), and
every claim in the answer must be traceable to a citation — so `sources`
holds ONLY the results that made it into the text, in citation order
(sources[i] is citation [i+1]).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from healthplus.config import get_settings
from healthplus.knowledge_base.models import SearchResult

logger = logging.getLogger(__name__)

# Blocks are separated by one blank line, i.e. two newline characters.
_BLOCK_SEPARATOR = "\n\n"


class RagContext(BaseModel):
    """The assembled context plus exactly the results it was built from."""

    context_text: str
    sources: list[SearchResult]
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        """True when retrieval produced nothing usable — the caller should
        answer "I don't know" instead of letting the LLM improvise."""
        return not self.sources


class ContextWindowManager:
    """Formats search results into a numbered, budget-capped context block."""

    def __init__(self, max_chars: int | None = None) -> None:
        self._max_chars = max_chars or get_settings().context_max_chars

    def assemble(self, results: list[SearchResult]) -> RagContext:
        """Dedupe, number, and pack results until the budget runs out."""
        blocks: list[str] = []
        sources: list[SearchResult] = []
        seen_chunk_ids: set[str] = set()
        truncated = False
        used_chars = 0

        for result in results:
            # The same chunk can surface twice (e.g. overlapping queries in
            # a multi-query setup); citing it twice adds cost, not signal.
            if result.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(result.chunk_id)

            citation_number = len(sources) + 1
            block = (
                f"[{citation_number}] {result.source} — "
                f"page {result.page_number} ({result.category})\n"
                f"{result.text}"
            )

            # Cost of adding this block: the block itself, plus a separator
            # if it is not the first one.
            separator_cost = len(_BLOCK_SEPARATOR) if blocks else 0
            if used_chars + separator_cost + len(block) > self._max_chars:
                truncated = True
                break

            blocks.append(block)
            sources.append(result)
            used_chars += separator_cost + len(block)

        logger.info(
            "Context assembled: %d block(s), %d chars (budget %d), truncated=%s",
            len(sources),
            used_chars,
            self._max_chars,
            truncated,
        )
        return RagContext(
            context_text=_BLOCK_SEPARATOR.join(blocks),
            sources=sources,
            truncated=truncated,
        )
