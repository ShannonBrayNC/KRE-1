from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from kre.embeddings import EmbeddingVector


@dataclass(frozen=True, slots=True)
class SemanticRecord:
    """One indexed chunk vector with citation-ready identifiers."""

    document_id: UUID
    chunk_id: UUID
    sequence: int
    vector: EmbeddingVector


@dataclass(frozen=True, slots=True)
class SemanticSearchResult:
    """One cosine-ranked semantic match."""

    document_id: UUID
    chunk_id: UUID
    sequence: int
    score: float


class SemanticIndex(ABC):
    """Provider-neutral contract for vector indexing and retrieval."""

    @abstractmethod
    async def replace_document(self, document_id: UUID, records: Sequence[SemanticRecord]) -> None:
        """Atomically replace all semantic records for one document."""

    @abstractmethod
    async def delete_document(self, document_id: UUID) -> bool:
        """Delete all records for a document and return whether any existed."""

    @abstractmethod
    async def search(
        self,
        query: EmbeddingVector,
        *,
        limit: int = 10,
        minimum_score: float = -1.0,
    ) -> Sequence[SemanticSearchResult]:
        """Return deterministic cosine-ranked matches."""


class InMemorySemanticIndex(SemanticIndex):
    """Reference semantic index for local development and contract tests."""

    def __init__(self) -> None:
        self._records: dict[UUID, list[SemanticRecord]] = {}

    async def replace_document(self, document_id: UUID, records: Sequence[SemanticRecord]) -> None:
        if any(record.document_id != document_id for record in records):
            raise ValueError("every semantic record must belong to the target document")

        dimensions = {record.vector.dimensions for record in records}
        models = {record.vector.model for record in records}
        if len(dimensions) > 1 or len(models) > 1:
            raise ValueError("semantic records for a document must share model and dimensions")

        chunk_ids = [record.chunk_id for record in records]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("semantic record chunk identifiers must be unique")

        self._records[document_id] = list(records)

    async def delete_document(self, document_id: UUID) -> bool:
        return self._records.pop(document_id, None) is not None

    async def search(
        self,
        query: EmbeddingVector,
        *,
        limit: int = 10,
        minimum_score: float = -1.0,
    ) -> Sequence[SemanticSearchResult]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not -1.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between -1 and 1")

        results: list[SemanticSearchResult] = []
        for records in self._records.values():
            for record in records:
                if record.vector.model != query.model:
                    continue
                if record.vector.dimensions != query.dimensions:
                    continue

                score = self._cosine(query.values, record.vector.values)
                if score < minimum_score:
                    continue

                results.append(
                    SemanticSearchResult(
                        document_id=record.document_id,
                        chunk_id=record.chunk_id,
                        sequence=record.sequence,
                        score=round(score, 8),
                    )
                )

        results.sort(
            key=lambda item: (
                -item.score,
                str(item.document_id),
                item.sequence,
                str(item.chunk_id),
            )
        )
        return results[:limit]

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)
