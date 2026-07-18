from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from kre.embeddings.base import EmbeddingProvider, EmbeddingVector


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Offline provider for tests and local development; not for semantic quality."""

    name = "deterministic"

    def __init__(self, *, dimensions: int = 16, model: str = "sha256-v1") -> None:
        if dimensions < 1 or dimensions > 32:
            raise ValueError("dimensions must be between 1 and 32")
        self.dimensions = dimensions
        self.model = model

    async def embed(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("every embedding input must be a string")

        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> EmbeddingVector:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [((digest[index] / 255.0) * 2.0) - 1.0 for index in range(self.dimensions)]
        magnitude = math.sqrt(sum(value * value for value in raw))
        values = tuple(value / magnitude for value in raw) if magnitude else tuple(raw)
        return EmbeddingVector(values=values, model=self.model, dimensions=self.dimensions)
