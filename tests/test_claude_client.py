"""ClaudeClient construction guards — no network, no real key."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from healthplus.core.exceptions import ConfigurationError
from healthplus.llm import ClaudeClient


def _settings(api_key=None):
    return SimpleNamespace(
        anthropic_api_key=api_key,
        claude_model="claude-sonnet-5",
        claude_max_tokens=1024,
    )


def test_missing_api_key_fails_fast():
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        ClaudeClient(settings=_settings(api_key=None))


def test_injected_client_skips_key_requirement():
    fake_sdk_client = object()
    client = ClaudeClient(settings=_settings(api_key=None), client=fake_sdk_client)
    assert client._client is fake_sdk_client
