"""Claude API adapter.

Design rules:
- Fail fast: a missing API key raises at construction time, not on the
  first user question in production.
- Streaming only: the UI renders tokens as they arrive, so blocking
  completions would just be a worse special case of streaming.
- Errors are translated: callers catch our LLMError, never SDK exceptions.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from typing import Iterator

import anthropic
import certifi
import httpx

from healthplus.config import Settings, get_settings
from healthplus.core.exceptions import ConfigurationError, LLMError

logger = logging.getLogger(__name__)

_CA_BUNDLE_PATH: str | None = None


def _get_ca_bundle() -> str:
    """Return a CA bundle path that includes certifi roots + macOS system keychain certs.

    The bundle is built once and cached for the process lifetime.  On non-macOS
    or when the `security` command is unavailable the plain certifi bundle is
    returned so the function is always safe to call.
    """
    global _CA_BUNDLE_PATH
    if _CA_BUNDLE_PATH is not None:
        return _CA_BUNDLE_PATH
    try:
        result = subprocess.run(
            ["security", "find-certificate", "-a", "-p", "/Library/Keychains/System.keychain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        system_pem = result.stdout
    except Exception:
        _CA_BUNDLE_PATH = certifi.where()
        return _CA_BUNDLE_PATH

    if not system_pem.strip():
        _CA_BUNDLE_PATH = certifi.where()
        return _CA_BUNDLE_PATH

    with tempfile.NamedTemporaryFile(suffix=".pem", delete=False, mode="w") as tmp:
        with open(certifi.where()) as cf:
            tmp.write(cf.read())
        tmp.write(system_pem)
        _CA_BUNDLE_PATH = tmp.name

    logger.debug("CA bundle built: %s", _CA_BUNDLE_PATH)
    return _CA_BUNDLE_PATH


class ClaudeClient:
    """Thin, testable wrapper around the Anthropic Messages API."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._model = settings.claude_model
        self._max_tokens = settings.claude_max_tokens

        if client is not None:
            # Injected client — tests use this to avoid real network calls.
            self._client = client
        else:
            if settings.anthropic_api_key is None:
                raise ConfigurationError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env "
                    "and add your key — the chat feature cannot start without it."
                )
            http_client = httpx.Client(verify=_get_ca_bundle())
            self._client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key.get_secret_value(),
                http_client=http_client,
            )

    def stream_reply(self, system: str, messages: list[dict]) -> Iterator[str]:
        """Stream Claude's answer as text deltas.

        `thinking` is explicitly disabled: claude-sonnet-5 runs adaptive
        thinking by default, which spends output tokens and delays the first
        visible token — wrong trade-off for a low-latency grounded chat where
        the hard reasoning already happened at retrieval time.
        """
        logger.info(
            "Claude request: model=%s, messages=%d, system_chars=%d",
            self._model,
            len(messages),
            len(system),
        )
        total_chars = 0
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
                thinking={"type": "disabled"},
            ) as stream:
                for text in stream.text_stream:
                    total_chars += len(text)
                    yield text
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Claude API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"Claude API error ({exc.status_code}): {exc.message}"
            ) from exc
        logger.info("Claude reply complete: %d chars streamed", total_chars)
