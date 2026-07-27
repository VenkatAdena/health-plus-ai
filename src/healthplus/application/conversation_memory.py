"""Per-conversation chat history.

Process-lifetime, in-memory storage is a deliberate v1 trade-off: it is
exactly right for a single-process Streamlit app and keeps Day-scope small.
Production would swap this for Redis or Postgres behind the same interface —
which is why the interface (append/history/clear) is the contract, not dicts.
"""

from __future__ import annotations

import logging

from healthplus.config import get_settings

logger = logging.getLogger(__name__)

_VALID_ROLES = {"user", "assistant"}


class InMemoryConversationMemory:
    """Windowed chat history keyed by conversation id."""

    def __init__(self, max_turns: int | None = None) -> None:
        # One "turn" = a user message plus the assistant reply, so the
        # message window is max_turns * 2.
        self._max_messages = (max_turns or get_settings().memory_max_turns) * 2
        self._conversations: dict[str, list[dict]] = {}

    def append(self, conversation_id: str, role: str, content: str) -> None:
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}, got {role!r}")
        self._conversations.setdefault(conversation_id, []).append(
            {"role": role, "content": content}
        )

    def history(self, conversation_id: str) -> list[dict]:
        """Return a windowed COPY — callers must not mutate stored state."""
        messages = self._conversations.get(conversation_id, [])
        return [dict(m) for m in messages[-self._max_messages :]]

    def clear(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)
        logger.info("Cleared conversation %s", conversation_id)
