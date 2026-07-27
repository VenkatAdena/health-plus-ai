"""ChatService orchestration, tested with fakes for every collaborator."""

from __future__ import annotations

from healthplus.application.chat_service import ChatService
from healthplus.application.context_manager import RagContext
from healthplus.application.conversation_memory import InMemoryConversationMemory
from healthplus.application.prompt_builder import HealthcarePromptBuilder
from healthplus.knowledge_base.models import SearchResult


def _result(text="MRI costs INR 8000", source="Pricing.pdf"):
    return SearchResult(
        text=text,
        score=0.9,
        source=source,
        category="pricing",
        page_number=1,
        chunk_id="chunk-1",
    )


class FakeQueryProcessor:
    def process(self, query):
        return query.strip()


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, category=None):
        self.calls.append({"query": query, "category": category})
        return self.results


class FakeContextManager:
    def assemble(self, results):
        return RagContext(
            context_text="\n\n".join(r.text for r in results),
            sources=list(results),
        )


class FakeLLM:
    def __init__(self, tokens=("Hello", " world")):
        self._tokens = tokens
        self.requests = []

    def stream_reply(self, system, messages):
        self.requests.append({"system": system, "messages": messages})
        yield from self._tokens


def _service(results=None, llm=None, memory=None):
    return (
        ChatService(
            query_processor=FakeQueryProcessor(),
            retriever=FakeRetriever(results if results is not None else [_result()]),
            context_manager=FakeContextManager(),
            prompt_builder=HealthcarePromptBuilder(),
            memory=memory or InMemoryConversationMemory(max_turns=10),
            llm=llm or FakeLLM(),
        )
    )


def test_sources_are_available_before_streaming():
    answer = _service().answer("  What does an MRI cost?  ", "c1")
    assert len(answer.sources) == 1
    assert answer.sources[0].source == "Pricing.pdf"


def test_tokens_stream_in_order():
    answer = _service().answer("What does an MRI cost?", "c1")
    assert "".join(answer.tokens) == "Hello world"


def test_memory_written_only_after_stream_consumed():
    memory = InMemoryConversationMemory(max_turns=10)
    service = _service(memory=memory)
    answer = service.answer("What does an MRI cost?", "c1")

    assert memory.history("c1") == []  # nothing recorded yet
    list(answer.tokens)  # consume the stream
    history = memory.history("c1")
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"] == "Hello world"


def test_history_passed_to_llm_excludes_inflight_turn():
    memory = InMemoryConversationMemory(max_turns=10)
    memory.append("c1", "user", "earlier question")
    memory.append("c1", "assistant", "earlier answer")
    llm = FakeLLM()
    service = _service(memory=memory, llm=llm)

    list(service.answer("new question", "c1").tokens)
    sent = llm.requests[0]["messages"]
    # 2 prior + the current user turn; the in-flight turn is not duplicated
    assert len(sent) == 3
    assert sent[-1] == {"role": "user", "content": "new question"}


def test_category_is_forwarded_to_retriever():
    retriever = FakeRetriever([_result()])
    service = ChatService(
        query_processor=FakeQueryProcessor(),
        retriever=retriever,
        context_manager=FakeContextManager(),
        prompt_builder=HealthcarePromptBuilder(),
        memory=InMemoryConversationMemory(max_turns=10),
        llm=FakeLLM(),
    )
    service.answer("MRI cost", "c1", category="pricing")
    assert retriever.calls[0]["category"] == "pricing"


def test_empty_retrieval_still_answers_with_empty_context():
    llm = FakeLLM(tokens=("I don't know",))
    answer = _service(results=[], llm=llm).answer("Who won the World Cup?", "c1")
    assert answer.sources == []
    list(answer.tokens)
    assert "(no relevant documents were found" in llm.requests[0]["system"]
