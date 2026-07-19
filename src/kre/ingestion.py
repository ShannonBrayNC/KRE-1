from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from kre.embeddings import EmbeddingProvider
from kre.models import KnowledgeChunk, KnowledgeDocument
from kre.search import SemanticIndex, SemanticRecord
from kre.storage import KnowledgeRepository


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Outcome of one governed document ingestion operation."""

    document_id: UUID
    chunk_count: int
    embedding_model: str
    embedding_dimensions: int


class IngestionConsistencyError(RuntimeError):
    """Raised when ingestion fails and semantic cleanup cannot be completed."""


class KnowledgeIngestionService:
    """Coordinate canonical storage, embedding, and semantic indexing."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        semantic_index: SemanticIndex,
        embeddings: EmbeddingProvider,
    ) -> None:
        self._repository = repository
        self._semantic_index = semantic_index
        self._embeddings = embeddings

    async def ingest(
        self,
        document: KnowledgeDocument,
        chunks: Sequence[KnowledgeChunk],
    ) -> IngestionResult:
        """Persist a document and atomically replace its semantic projection."""

        ordered = self._validate_chunks(document.id, chunks)
        vectors = await self._embed_chunks(ordered)
        records = self._records(document.id, ordered, vectors)

        try:
            await self._repository.upsert_document(document)
            await self._repository.replace_chunks(document.id, ordered)
            await self._semantic_index.replace_document(document.id, records)
        except BaseException as primary:
            try:
                await self._semantic_index.delete_document(document.id)
            except BaseException as cleanup:
                raise IngestionConsistencyError(
                    f"ingestion failed and semantic cleanup failed for document {document.id}"
                ) from cleanup
            raise primary

        return IngestionResult(
            document_id=document.id,
            chunk_count=len(ordered),
            embedding_model=self._embeddings.model,
            embedding_dimensions=self._embeddings.dimensions,
        )

    async def reindex(self, document_id: UUID) -> IngestionResult:
        """Rebuild semantic records for an existing canonical document."""

        document = await self._repository.get_document(document_id)
        if document is None:
            raise KeyError(f"document does not exist: {document_id}")
        chunks = tuple(await self._repository.list_chunks(document_id))
        ordered = self._validate_chunks(document_id, chunks)
        vectors = await self._embed_chunks(ordered)
        await self._semantic_index.replace_document(
            document_id,
            self._records(document_id, ordered, vectors),
        )
        return IngestionResult(
            document_id=document_id,
            chunk_count=len(ordered),
            embedding_model=self._embeddings.model,
            embedding_dimensions=self._embeddings.dimensions,
        )

    async def delete(self, document_id: UUID) -> bool:
        """Delete semantic and canonical state without leaving stale vectors."""

        await self._semantic_index.delete_document(document_id)
        return await self._repository.delete_document(document_id)

    @staticmethod
    def _validate_chunks(
        document_id: UUID,
        chunks: Sequence[KnowledgeChunk],
    ) -> tuple[KnowledgeChunk, ...]:
        ordered = tuple(sorted(chunks, key=lambda chunk: chunk.sequence))
        if any(chunk.document_id != document_id for chunk in ordered):
            raise ValueError("every chunk must belong to the target document")
        sequences = [chunk.sequence for chunk in ordered]
        if len(sequences) != len(set(sequences)):
            raise ValueError("chunk sequence values must be unique")
        chunk_ids = [chunk.id for chunk in ordered]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk identifiers must be unique")
        return ordered

    async def _embed_chunks(
        self,
        chunks: tuple[KnowledgeChunk, ...],
    ):
        vectors = tuple(await self._embeddings.embed(tuple(chunk.text for chunk in chunks)))
        if len(vectors) != len(chunks):
            raise RuntimeError("embedding provider returned an unexpected vector count")
        if any(vector.model != self._embeddings.model for vector in vectors):
            raise RuntimeError("embedding provider returned an unexpected model")
        if any(vector.dimensions != self._embeddings.dimensions for vector in vectors):
            raise RuntimeError("embedding provider returned unexpected dimensions")
        return vectors

    @staticmethod
    def _records(document_id, chunks, vectors):
        return tuple(
            SemanticRecord(
                document_id=document_id,
                chunk_id=chunk.id,
                sequence=chunk.sequence,
                vector=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        )
