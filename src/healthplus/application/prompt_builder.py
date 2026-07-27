"""Prompt construction for the RAG chat.

The system prompt is where a healthcare chatbot earns (or loses) trust:
it pins the model to the retrieved context, forces citations, and draws
a hard line at medical advice. The retrieved context lives in the SYSTEM
prompt — not the user message — so multi-turn history stays clean and
each turn gets fresh context without polluting the conversation record.
"""

from __future__ import annotations

SYSTEM_PROMPT_TEMPLATE = """\
You are the assistant for HealthPlus hospital's operational knowledge base. \
You help patients and staff with questions about appointments, doctors, \
pricing, policies, health packages, diagnostic tests, reports, and FAQs.

Rules you must always follow:
- Answer ONLY from the numbered context blocks below. Do not use outside knowledge about HealthPlus.
- Cite your sources inline as [n], where n matches the context block number you used.
- If the answer is not in the context, say so plainly and suggest contacting the hospital front desk.
- NEVER give medical diagnosis, treatment, or medication advice. Redirect such questions to a qualified doctor at HealthPlus.
- Be concise and professional.
- All prices are in INR (Indian Rupees).

Context:
{context}
"""

_EMPTY_CONTEXT_MARKER = "(no relevant documents were found for this question)"


class HealthcarePromptBuilder:
    """Builds the system prompt and message list for one chat turn."""

    def build_system(self, context_text: str) -> str:
        """Embed retrieved context into the guardrailed system prompt.

        An explicit empty-context marker (instead of a blank section) makes
        the "I don't know" guardrail fire reliably — the model sees a
        positive statement that nothing was found, not an ambiguous gap.
        """
        context = context_text.strip() or _EMPTY_CONTEXT_MARKER
        return SYSTEM_PROMPT_TEMPLATE.format(context=context)

    def build_messages(self, query: str, history: list[dict]) -> list[dict]:
        """Prior turns plus the current user question, in API message format.

        `history` must not yet contain the in-flight turn — memory is only
        written after the reply streams successfully (see ChatService).
        """
        return [*history, {"role": "user", "content": query}]
