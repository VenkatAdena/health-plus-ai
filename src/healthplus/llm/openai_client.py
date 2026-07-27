"""OpenAI (ChatGPT) API adapter.

Design rules (mirrors claude_client.py):
- Fail fast: a missing API key raises at construction time, not on the
  first user question in production.
- Streaming only: the UI renders tokens as they arrive, so blocking
  completions would just be a worse special case of streaming.
- Errors are translated: callers catch our LLMError, never SDK exceptions.
"""

from __future__ import annotations

import logging
from typing import Iterator

import httpx
import openai as openai_sdk

from healthplus.config import Settings, get_settings
from healthplus.llm.claude_client import _get_ca_bundle
from healthplus.core.exceptions import ConfigurationError, LLMError

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Thin, testable wrapper around the OpenAI Chat Completions API."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: openai_sdk.OpenAI | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._model = settings.openai_model
        self._max_tokens = settings.claude_max_tokens

        if client is not None:
            self._client = client
        else:
            if settings.openai_api_key is None:
                raise ConfigurationError(
                    "OPENAI_API_KEY is not set. Add it to your .env file — "
                    "the chat feature cannot start without it."
                )
            http_client = httpx.Client(verify=_get_ca_bundle())
            self._client = openai_sdk.OpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                http_client=http_client,
            )

    def stream_reply(self, system: str, messages: list[dict]) -> Iterator[str]:
        """Stream OpenAI's answer as text deltas.

        The system prompt is prepended as a system-role message because
        OpenAI's chat completions API does not have a separate `system`
        parameter — the convention is a leading message with role "system".
        """
        logger.info(
            "OpenAI request: model=%s, messages=%d, system_chars=%d",
            self._model,
            len(messages),
            len(system),
        )
        openai_messages = [{"role": "system", "content": system}, *messages]
        total_chars = 0
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=openai_messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    total_chars += len(delta)
                    yield delta
        except openai_sdk.APIConnectionError as exc:
            raise LLMError(f"Could not reach the OpenAI API: {exc}") from exc
        except openai_sdk.APIStatusError as exc:
            raise LLMError(
                f"OpenAI API error ({exc.status_code}): {exc.message}"
            ) from exc
        logger.info("OpenAI reply complete: %d chars streamed", total_chars)
