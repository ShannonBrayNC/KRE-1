from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """One provider-neutral embedding result."""

    values: tuple[float, ...]
    model: str
    dimensions: int

    def __post_init__(self) -> None:
        if self.dimensions < 1:
            raise ValueError("dimensions must be at least 1")
        if len(self.values) != self.dimensions:
            raise ValueError("embedding length must match dimensions")


class EmbeddingProvider(ABC):
    """Provider-neutral contract for converting text into vectors."""

    name: str
    model: str
    dimensions: int

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]:
        """Return one ordered vector for every supplied text value."""
