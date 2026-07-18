from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import UUID

from kre.models import KnowledgeChunk, KnowledgeDocument


class KnowledgeRepository(ABC):
    """Provider-neutral persistence contract for canonical KRE entities."""

    @abstractmethod
    async def upsert_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        """Create or replace a document by its stable identifier."""

    @abstractmethod
    async def get_document(self, document_id: UUID) -> KnowledgeDocument | None:
        """Return one document or None when it does not exist."""

    @abstractmethod
    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document and its chunks; return whether it existed."""

    @abstractmethod
    async def list_documents(self, *, limit: int = 100, offset: int = 0) -> Sequence[KnowledgeDocument]:
        """Return documents in deterministic identifier order."""

    @abstractmethod
    async def replace_chunks(
        self,
        document_id: UUID,
        chunks: Sequence[KnowledgeChunk],
    ) -> Sequence[KnowledgeChunk]:
        """Atomically replace all chunks belonging to one document."""

    @abstractmethod
    async def list_chunks(self, document_id: UUID) -> Sequence[KnowledgeChunk]:
        """Return document chunks ordered by sequence."""