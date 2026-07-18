from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from kre.models import KnowledgeChunk, KnowledgeDocument
from kre.storage.base import KnowledgeRepository


class InMemoryKnowledgeRepository(KnowledgeRepository):
    """Reference repository used by tests, local development, and contract validation."""

    def __init__(self) -> None:
        self._documents: dict[UUID, KnowledgeDocument] = {}
        self._chunks: dict[UUID, list[KnowledgeChunk]] = {}

    async def upsert_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        self._documents[document.id] = document.model_copy(deep=True)
        return self._documents[document.id].model_copy(deep=True)

    async def get_document(self, document_id: UUID) -> KnowledgeDocument | None:
        document = self._documents.get(document_id)
        return document.model_copy(deep=True) if document else None

    async def delete_document(self, document_id: UUID) -> bool:
        existed = document_id in self._documents
        self._documents.pop(document_id, None)
        self._chunks.pop(document_id, None)
        return existed

    async def list_documents(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[KnowledgeDocument]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if offset < 0:
            raise ValueError("offset cannot be negative")

        documents = [self._documents[key] for key in sorted(self._documents, key=str)]
        return [item.model_copy(deep=True) for item in documents[offset : offset + limit]]

    async def replace_chunks(
        self,
        document_id: UUID,
        chunks: Sequence[KnowledgeChunk],
    ) -> Sequence[KnowledgeChunk]:
        if document_id not in self._documents:
            raise KeyError(f"document does not exist: {document_id}")
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("every chunk must belong to the target document")

        ordered = sorted(chunks, key=lambda chunk: chunk.sequence)
        sequences = [chunk.sequence for chunk in ordered]
        if len(sequences) != len(set(sequences)):
            raise ValueError("chunk sequence values must be unique")

        self._chunks[document_id] = [chunk.model_copy(deep=True) for chunk in ordered]
        return [chunk.model_copy(deep=True) for chunk in self._chunks[document_id]]

    async def list_chunks(self, document_id: UUID) -> Sequence[KnowledgeChunk]:
        return [chunk.model_copy(deep=True) for chunk in self._chunks.get(document_id, [])]