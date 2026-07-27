"""Application exception hierarchy.

Every custom exception derives from HealthPlusError so callers can catch
"anything that went wrong in our code" with a single except clause, while
still distinguishing failure categories when they need to.
"""

from __future__ import annotations


class HealthPlusError(Exception):
    """Base class for all HealthPlus AI errors."""


class DocumentLoadError(HealthPlusError):
    """A document could not be read or parsed."""


class EmptyDocumentError(DocumentLoadError):
    """The document contained no extractable text (likely a scan — needs OCR)."""


class KnowledgeBaseError(HealthPlusError):
    """A knowledge-base operation (embedding, storage, search) failed."""


class ConfigurationError(HealthPlusError):
    """Required configuration is missing or invalid (fail fast, not mid-request)."""


class LLMError(HealthPlusError):
    """A Claude API call failed (network, rate limit, or server error)."""
