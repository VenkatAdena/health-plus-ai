"""Conversation memory: windowing, isolation, and copy semantics."""

from __future__ import annotations

import pytest

from healthplus.application.conversation_memory import InMemoryConversationMemory


def test_append_and_history_round_trip():
    memory = InMemoryConversationMemory(max_turns=10)
    memory.append("c1", "user", "hi")
    memory.append("c1", "assistant", "hello")
    assert memory.history("c1") == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_history_is_windowed_to_max_turns():
    memory = InMemoryConversationMemory(max_turns=2)  # window = 4 messages
    for i in range(5):
        memory.append("c1", "user", f"q{i}")
        memory.append("c1", "assistant", f"a{i}")
    history = memory.history("c1")
    assert len(history) == 4
    assert history[0]["content"] == "q3"  # oldest surviving message
    assert history[-1]["content"] == "a4"


def test_conversations_are_isolated():
    memory = InMemoryConversationMemory(max_turns=10)
    memory.append("c1", "user", "hi")
    assert memory.history("c2") == []


def test_history_returns_a_copy():
    memory = InMemoryConversationMemory(max_turns=10)
    memory.append("c1", "user", "hi")
    memory.history("c1")[0]["content"] = "tampered"
    assert memory.history("c1")[0]["content"] == "hi"


def test_invalid_role_rejected():
    memory = InMemoryConversationMemory(max_turns=10)
    with pytest.raises(ValueError, match="role"):
        memory.append("c1", "system", "not allowed")


def test_clear_removes_conversation():
    memory = InMemoryConversationMemory(max_turns=10)
    memory.append("c1", "user", "hi")
    memory.clear("c1")
    assert memory.history("c1") == []
