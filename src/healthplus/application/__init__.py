"""Layer 3: use-case orchestration for ingestion, retrieval, prompts, and chat."""

from healthplus.application.chat_service import (
    ChatAnswer,
    ChatService,
    build_chat_service,
)
from healthplus.application.context_manager import ContextWindowManager, RagContext
from healthplus.application.conversation_memory import InMemoryConversationMemory
from healthplus.application.knowledge_base_service import (
    BatchIngestionReport,
    IngestionFailure,
    IngestionReport,
    KnowledgeBaseService,
)
from healthplus.application.prompt_builder import HealthcarePromptBuilder
from healthplus.application.query_processor import MAX_QUERY_CHARS, DefaultQueryProcessor
from healthplus.application.retriever import KnowledgeBaseRetriever

__all__ = [
    "BatchIngestionReport",
    "ChatAnswer",
    "ChatService",
    "ContextWindowManager",
    "DefaultQueryProcessor",
    "HealthcarePromptBuilder",
    "InMemoryConversationMemory",
    "IngestionFailure",
    "IngestionReport",
    "KnowledgeBaseRetriever",
    "KnowledgeBaseService",
    "MAX_QUERY_CHARS",
    "RagContext",
    "build_chat_service",
]
