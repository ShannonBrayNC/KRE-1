from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from kre.embeddings import EmbeddingProvider
from kre.search.semantic import SemanticIndex
from kre.storage import KnowledgeRepository


@dataclass(frozen=True, slots=True)
class SemanticRetrievalResult:
    """Resolved semantic match with citation and provenance context."""

    document_id: UUID
    chunk_id: UUID
    document_title: str
    sequence: int
    text: str
    score: float
    source_uri: str | None
    source_system: str
    source_version: str | None
    classification: str
    security_labels: tuple[str, ...]
    metadata: dict[str, Any]


class SemanticRetrievalService:
    """Embed a query, search the semantic index, and resolve canonical content."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        index: SemanticIndex,
        embeddings: EmbeddingProvider,
    ) -> None:
        self._repository = repository
        self._index = index
        self._embeddings = embeddings

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        minimum_score: float = -1.0,
    ) -> list[SemanticRetrievalResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        vectors = await self._embeddings.embed([normalized_query])
        if len(vectors) != 1:
            raise RuntimeError("embedding provider must return exactly one query vector")

        matches = await self._index.search(
            vectors[0],
            limit=limit,
            minimum_score=minimum_score,
        )

        results: list[SemanticRetrievalResult] = []
        chunks_by_document: dict[UUID, dict[UUID, Any]] = {}
        documents: dict[UUID, Any] = {}

        for match in matches:
            document = documents.get(match.document_id)
            if document is None:
                document = await self._repository.get_document(match.document_id)
                if document is None:
                    continue
                documents[match.document_id] = document

            document_chunks = chunks_by_document.get(match.document_id)
            if document_chunks is None:
                chunks = await self._repository.list_chunks(match.document_id)
                document_chunks = {chunk.id: chunk for chunk in chunks}
                chunks_by_document[match.document_id] = document_chunks

            chunk = document_chunks.get(match.chunk_id)
            if chunk is None:
                continue

            results.append(
                SemanticRetrievalResult(
                    document_id=document.id,
                    chunk_id=chunk.id,
                    document_title=document.title,
                    sequence=chunk.sequence,
                    text=chunk.text,
                    score=match.score,
                    source_uri=(
                        str(document.provenance.source_uri)
                        if document.provenance.source_uri is not None
                        else None
                    ),
                    source_system=document.provenance.source_system,
                    source_version=document.provenance.source_version,
                    classification=document.classification.value,
                    security_labels=tuple(document.security_labels),
                    metadata=dict(chunk.metadata),
                )
            )

        return results
