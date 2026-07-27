"""Application configuration.

All runtime configuration is loaded from environment variables (optionally
supplied via a `.env` file) and validated once at startup.

Rule for the whole codebase: modules import `get_settings()` — nothing else
ever reads `os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class Settings(BaseSettings):
    """Single, validated source of truth for runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="HEALTHPLUS_",
        extra="ignore",
    )

    # --- Application identity ---
    app_name: str = "HealthPlus AI"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # --- LLM ---
    # Optional at settings load time; ClaudeClient and OpenAIClient fail fast
    # at construction if the required key is missing.
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
    )
    claude_model: str = "claude-sonnet-5"
    # No temperature/top_p: claude-sonnet-5 rejects non-default sampling
    # parameters — output style is steered through the system prompt instead.
    claude_max_tokens: int = Field(default=1024, gt=0)

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    openai_model: str = "gpt-4o-mini"

    # --- Retrieval / RAG ---
    retrieval_top_k: int = Field(default=5, gt=0)
    # Cosine-similarity floor: hits below this are noise, not context.
    retrieval_min_score: float = Field(default=0.30, ge=-1.0, le=1.0)
    context_max_chars: int = Field(default=8000, gt=0)

    # --- Conversation ---
    memory_max_turns: int = Field(default=10, gt=0)

    # --- Storage ---
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")

    # --- Knowledge base ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_collection: str = "healthplus_documents"
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    # --- Logging ---
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, got {value!r}"
            )
        return level

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    @property
    def chroma_dir(self) -> Path:
        """Persistent home of the vector store (used from Day 4 onward)."""
        return self.data_dir / "chroma"

    def ensure_directories(self) -> None:
        """Create runtime directories that must exist before first use."""
        for directory in (self.data_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance (built once, then cached)."""
    return Settings()
