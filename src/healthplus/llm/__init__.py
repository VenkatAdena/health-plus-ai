"""LLM adapter layer: the only modules that talk to external LLM APIs.

Keeping each SDK behind one class means the rest of the application depends
on our contract (stream_reply), not on any vendor's SDK — swapping models or
mocking in tests touches exactly one seam.
"""

from healthplus.llm.claude_client import ClaudeClient
from healthplus.llm.openai_client import OpenAIClient

__all__ = ["ClaudeClient", "OpenAIClient"]
