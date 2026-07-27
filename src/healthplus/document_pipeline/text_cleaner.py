"""Deterministic cleanup for text extracted by the document pipeline."""

from __future__ import annotations

import re
import unicodedata


class TextCleaner:
    """Normalize extraction artifacts without rewriting document meaning.

    PDF extraction commonly introduces soft hyphens, words split across line
    endings, repeated spaces and excessive blank lines. Cleaning these before
    chunking improves both chunk quality and, later, embedding quality.
    """

    _DEHYPHENATE = re.compile(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])")
    _LINE_WHITESPACE = re.compile(r"[ \t]+")
    _SPACE_AROUND_NEWLINE = re.compile(r"[ \t]*\n[ \t]*")
    _EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

    def clean(self, text: str) -> str:
        """Return stable, readable text while preserving paragraph breaks."""
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u00ad", "").replace("\r\n", "\n").replace("\r", "\n")
        text = self._DEHYPHENATE.sub("", text)
        text = self._LINE_WHITESPACE.sub(" ", text)
        text = self._SPACE_AROUND_NEWLINE.sub("\n", text)
        text = self._EXCESS_BLANK_LINES.sub("\n\n", text)
        return text.strip()
