"""End-to-end RAG integration: real pipeline components over a real (but
isolated) ChromaDB store.

The unit tests fake every collaborator; this module fakes only the two
boundaries we cannot exercise hermetically — the embedding model (slow,
multi-second load) and the Claude API (no key, no network). Everything
else is REAL: DefaultQueryProcessor -> KnowledgeBaseRetriever ->
KnowledgeBaseService -> VectorStore (Chroma in tmp_path) ->
ContextWindowManager -> HealthcarePromptBuilder -> InMemoryConversationMemory
-> ChatService. That proves the pieces actually compose: metadata survives
the round-trip through Chroma, citations number correctly, and history
threads across turns.

The fake embedder hashes tokens into a small vector space, so texts that
share words land near each other — similarity is crude but MEANINGFUL,
which lets us assert that the semantically-right chunk wins retrieval.
"""

from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import pytest

from healthplus.application.chat_service import ChatService
from healthplus.application.context_manager import ContextWindowManager
from healthplus.application.conversation_memory import InMemoryConversationMemory
from healthplus.application.knowledge_base_service import KnowledgeBaseService
from healthplus.application.prompt_builder import HealthcarePromptBuilder
from healthplus.application.query_processor import DefaultQueryProcessor
from healthplus.application.retriever import KnowledgeBaseRetriever
from healthplus.knowledge_base.models import Chunk, DocumentCategory
from healthplus.vector_database.vector_store import VectorStore

_DIMS = 32


class FakeEmbeddingService:
    """Deterministic bag-of-words embeddings — no model, no network.

    Each token is hashed (md5, NOT built-in hash(), which is salted per
    process) into one of _DIMS buckets; the vector is then unit-normalized
    so cosine similarity behaves like the real normalized BGE embeddings.
    Texts sharing tokens get high similarity; disjoint texts get ~0.
    The SAME instance embeds both the stored chunks and the query, so
    nearest-neighbour search in Chroma is semantically meaningful.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        vector = [0.0] * _DIMS
        for raw in text.lower().split():
            token = raw.strip(".,?!:;()'\"")
            if not token:
                continue
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % _DIMS
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


class FakeLLM:
    """Records what ChatService sends and streams a canned reply."""

    def __init__(self, tokens: tuple[str, ...] = ("An MRI", " costs INR 8000 [1].")):
        self._tokens = tokens
        self.requests: list[dict] = []

    def stream_reply(self, system, messages):
        self.requests.append({"system": system, "messages": messages})
        yield from self._tokens


def _chunk(chunk_id: str, text: str, source: str, category: DocumentCategory, page: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{category.value}",
        text=text,
        source=source,
        category=category,
        page_number=page,
        chunk_index=0,
    )


# Three chunks with (deliberately) disjoint vocabularies, except that the
# pricing chunk shares "an mri scan cost" with the test query — so the
# bag-of-words fake embedder ranks it first by a wide margin.
PRICING_CHUNK = _chunk(
    "pricing-0",
    "An MRI scan cost is INR 8000 at HealthPlus radiology.",
    "Pricing.pdf",
    DocumentCategory.PRICING,
    page=3,
)
DOCTORS_CHUNK = _chunk(
    "doctors-0",
    "Dr. Sample 1 leads cardiology and consults every weekday morning.",
    "Doctors.pdf",
    DocumentCategory.DOCTORS,
    page=1,
)
POLICIES_CHUNK = _chunk(
    "policies-0",
    "Refunds for cancelled bookings arrive within seven working days.",
    "Policies.pdf",
    DocumentCategory.POLICIES,
    page=2,
)


@pytest.fixture()
def rag(tmp_path):
    """A fully wired ChatService over an isolated store in tmp_path.

    data/chroma is never touched: the VectorStore persists under pytest's
    tmp_path, and the collection name is test-only.
    """
    embedder = FakeEmbeddingService()
    store = VectorStore(
        persist_dir=tmp_path / "chroma", collection_name="test_rag_integration"
    )

    chunks = [PRICING_CHUNK, DOCTORS_CHUNK, POLICIES_CHUNK]
    store.upsert(chunks, embedder.embed_texts([c.text for c in chunks]))

    kb = KnowledgeBaseService(embedder=embedder, store=store)
    retriever = KnowledgeBaseRetriever(
        kb=kb,
        # SimpleNamespace instead of real Settings: the retriever only reads
        # these two fields, and a low floor keeps the test about RANKING
        # (which chunk wins), not about absolute fake-similarity values.
        settings=SimpleNamespace(retrieval_top_k=3, retrieval_min_score=0.2),
    )
    llm = FakeLLM()
    service = ChatService(
        query_processor=DefaultQueryProcessor(),
        retriever=retriever,
        context_manager=ContextWindowManager(max_chars=8000),
        prompt_builder=HealthcarePromptBuilder(),
        memory=InMemoryConversationMemory(max_turns=10),
        llm=llm,
    )
    return SimpleNamespace(service=service, llm=llm)


def test_query_retrieves_the_semantically_matching_chunk(rag):
    answer = rag.service.answer("How much does an MRI scan cost?", "c1")
    assert answer.sources, "expected at least one source from the real store"
    assert answer.sources[0].chunk_id == PRICING_CHUNK.chunk_id


def test_system_prompt_contains_chunk_text_and_citation_header(rag):
    answer = rag.service.answer("How much does an MRI scan cost?", "c1")
    list(answer.tokens)  # the prompt is only sent when the stream runs

    system = rag.llm.requests[0]["system"]
    assert PRICING_CHUNK.text in system
    # The exact header format ContextWindowManager promises: number, source,
    # page, category — this is what makes [n] citations verifiable.
    assert "[1] Pricing.pdf — page 3 (pricing)" in system


def test_sources_carry_metadata_through_the_chroma_round_trip(rag):
    answer = rag.service.answer("How much does an MRI scan cost?", "c1")
    top = answer.sources[0]
    assert top.source == "Pricing.pdf"
    assert top.page_number == 3
    assert top.category == "pricing"


def test_second_turn_includes_first_turn_in_history(rag):
    first = rag.service.answer("How much does an MRI scan cost?", "c1")
    reply = "".join(first.tokens)  # consume so memory records the turn

    second = rag.service.answer("Is that price inclusive of the report?", "c1")
    list(second.tokens)

    messages = rag.llm.requests[1]["messages"]
    assert messages[0] == {
        "role": "user",
        "content": "How much does an MRI scan cost?",
    }
    assert messages[1] == {"role": "assistant", "content": reply}
    assert messages[-1] == {
        "role": "user",
        "content": "Is that price inclusive of the report?",
    }


def test_empty_query_raises_before_touching_retrieval(rag):
    with pytest.raises(ValueError):
        rag.service.answer("   ", "c1")
    assert rag.llm.requests == []  # nothing was sent to the LLM
