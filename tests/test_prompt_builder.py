"""The system prompt is a safety contract — these tests pin its guarantees."""

from __future__ import annotations

from healthplus.application.prompt_builder import HealthcarePromptBuilder


def test_system_prompt_contains_guardrails():
    system = HealthcarePromptBuilder().build_system("[1] Pricing.pdf — page 1 (pricing)\nMRI costs INR 8000")
    assert "ONLY from the numbered context blocks" in system
    assert "NEVER give medical diagnosis" in system
    assert "[n]" in system  # citation instruction
    assert "MRI costs INR 8000" in system  # context embedded


def test_empty_context_gets_explicit_marker():
    system = HealthcarePromptBuilder().build_system("")
    assert "(no relevant documents were found for this question)" in system


def test_messages_are_history_plus_current_turn():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    messages = HealthcarePromptBuilder().build_messages("What does an MRI cost?", history)
    assert messages[:2] == history
    assert messages[-1] == {"role": "user", "content": "What does an MRI cost?"}
    assert len(messages) == 3


def test_messages_does_not_mutate_history():
    history: list[dict] = []
    HealthcarePromptBuilder().build_messages("question", history)
    assert history == []
