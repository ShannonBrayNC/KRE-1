from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class NormalizedContent:
    text: str
    content_hash: str


def normalize_text(value: str) -> NormalizedContent:
    """Normalize text deterministically without changing semantic line breaks."""
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_WHITESPACE.sub(" ", line).rstrip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text).strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return NormalizedContent(text=text, content_hash=f"sha256:{digest}")
