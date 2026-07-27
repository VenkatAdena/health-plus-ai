"""The RAG chat use-case: one user question in, one grounded answer out.

This is the conductor for the whole architecture diagram:

    query -> QueryProcessor -> Retriever -> ContextManager
          -> PromptBuilder -> ClaudeClient (stream) -> memory

Every collaborator is injected and duck-typed against the contracts in
application/components.py, so each stage can be faked in tests and swapped
in production without touching this file.
"""

from __future__ import annotations

import logging
from typing import Iterator

from healthplus.knowledge_base.models import SearchResult

logger = logging.getLogger(__name__)


class ChatAnswer:
    """One in-flight answer: sources known up front, tokens streamed lazily.

    A plain class (not Pydantic) because `tokens` is a live generator —
    it cannot be validated or serialized, only consumed once.
    """

    def __init__(self, sources: list[SearchResult], tokens: Iterator[str]) -> None:
        self.sources = sources
        self.tokens = tokens


class ChatService:
    """Orchestrates retrieval-augmented chat turns."""

    def __init__(
        self,
        query_processor,
        retriever,
        context_manager,
        prompt_builder,
        memory,
        llm,
    ) -> None:
        """All collaborators are required — wiring lives in build_chat_service().

        Duck-typed contracts:
        - query_processor.process(query) -> str
        - retriever.retrieve(query, category=None) -> list[SearchResult]
        - context_manager.assemble(results) -> RagContext (.context_text, .sources)
        - prompt_builder.build_system(context_text) -> str;
          .build_messages(query, history) -> list[dict]
        - memory.append(cid, role, content); .history(cid) -> list[dict]
        - llm.stream_reply(system, messages) -> Iterator[str]
        """
        self._query_processor = query_processor
        self._retriever = retriever
        self._context_manager = context_manager
        self._prompt_builder = prompt_builder
        self._memory = memory
        self._llm = llm

    def answer(
        self, query: str, conversation_id: str, category: str | None = None
    ) -> ChatAnswer:
        """Run one RAG turn; returns sources immediately, tokens lazily.

        Empty retrieval still calls Claude: the empty-context system prompt
        makes it answer honestly ("that's not in my documents"), and it keeps
        plain conversation ("hello", "thanks") natural. The trade-off is one
        API call for questions we could refuse for free — acceptable at this
        scale, and revisit with a similarity-based short-circuit if cost bites.
        """
        clean_query = self._query_processor.process(query)
        results = self._retriever.retrieve(clean_query, category=category)
        context = self._context_manager.assemble(results)

        system = self._prompt_builder.build_system(context.context_text)
        history = self._memory.history(conversation_id)
        messages = self._prompt_builder.build_messages(clean_query, history)

        logger.info(
            "Chat turn: conversation=%s, sources=%d, history_messages=%d",
            conversation_id,
            len(context.sources),
            len(history),
        )

        def _stream() -> Iterator[str]:
            # Memory is written INSIDE the generator, after the stream is
            # fully consumed: an abandoned or failed stream must not record
            # a turn that the user never actually received.
            pieces: list[str] = []
            for token in self._llm.stream_reply(system, messages):
                pieces.append(token)
                yield token
            self._memory.append(conversation_id, "user", clean_query)
            self._memory.append(conversation_id, "assistant", "".join(pieces))

        return ChatAnswer(sources=list(context.sources), tokens=_stream())


def build_chat_service(settings=None, provider: str = "claude") -> ChatService:
    """Wire the default production chat service.

    The one place that knows the concrete classes; everything else depends
    on the ports in application/components.py.

    ``provider`` selects the LLM backend: ``"claude"`` (default) or ``"openai"``.
    """
    from healthplus.application.context_manager import ContextWindowManager
    from healthplus.application.conversation_memory import InMemoryConversationMemory
    from healthplus.application.knowledge_base_service import KnowledgeBaseService
    from healthplus.application.prompt_builder import HealthcarePromptBuilder
    from healthplus.application.query_processor import DefaultQueryProcessor
    from healthplus.application.retriever import KnowledgeBaseRetriever
    from healthplus.llm import ClaudeClient, OpenAIClient

    kb = KnowledgeBaseService(settings=settings)

    if provider == "openai":
        llm = OpenAIClient(settings=settings)
    else:
        llm = ClaudeClient(settings=settings)

    return ChatService(
        query_processor=DefaultQueryProcessor(),
        retriever=KnowledgeBaseRetriever(kb=kb, settings=settings),
        context_manager=ContextWindowManager(),
        prompt_builder=HealthcarePromptBuilder(),
        memory=InMemoryConversationMemory(),
        llm=llm,
    )
