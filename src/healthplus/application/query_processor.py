"""Query normalization for the RAG pipeline.

Raw user input arrives messy: leading/trailing spaces, doubled spaces from
copy-paste, stray newlines from chat clients. We clean it up ONCE, here,
so every downstream stage (embedding, logging, prompt assembly) sees the
same canonical string. We also reject inputs that would waste an embedding
call (empty) or blow past what the embedding model and the prompt can
sensibly handle (too long).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Hard ceiling on query length. Anything longer is almost certainly a
# pasted document, not a question — and it would degrade the embedding
# and eat the prompt budget.
MAX_QUERY_CHARS = 2000


class DefaultQueryProcessor:
    """Cleans and validates a user query before it enters the pipeline."""

    def process(self, query: str) -> str:
        """Return the normalized query, or raise ValueError if unusable.

        str.split() with no arguments splits on ANY run of whitespace
        (spaces, tabs, newlines) and discards leading/trailing runs, so
        joining with a single space both collapses and strips in one step.
        """
        normalized = " ".join(query.split())

        if not normalized:
            raise ValueError("Query must not be empty or whitespace-only.")

        if len(normalized) > MAX_QUERY_CHARS:
            raise ValueError(
                f"Query is {len(normalized)} characters long; "
                f"the maximum is {MAX_QUERY_CHARS}."
            )

        logger.debug("Normalized query: %r", normalized)
        return normalized
