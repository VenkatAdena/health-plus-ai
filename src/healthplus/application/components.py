"""Named application boundaries from the enterprise architecture.

These Protocols are the ports the presentation layer programs against.
Each has exactly one production implementation today; the Protocol exists
so tests can substitute fakes and production can swap implementations
(e.g. Redis-backed memory) without touching ChatService.
"""

from __future__ import annotations

from typing import Iterator, Protocol

from healthplus.knowledge_base.models import SearchResult


class QueryProcessor(Protocol):
    def process(self, query: str) -> str: ...


class Retriever(Protocol):
    def retrieve(
        self, query: str, category: str | None = None, top_k: int | None = None
    ) -> list[SearchResult]: ...


class ContextManager(Protocol):
    def assemble(self, results: list[SearchResult]) -> object:
        """Returns a RagContext: .context_text, .sources, .truncated, .is_empty."""
        ...


class PromptBuilder(Protocol):
    def build_system(self, context_text: str) -> str: ...

    def build_messages(self, query: str, history: list[dict]) -> list[dict]: ...


class ConversationMemory(Protocol):
    def append(self, conversation_id: str, role: str, content: str) -> None: ...

    def history(self, conversation_id: str) -> list[dict]: ...

    def clear(self, conversation_id: str) -> None: ...


class LLMClient(Protocol):
    def stream_reply(self, system: str, messages: list[dict]) -> Iterator[str]: ...
