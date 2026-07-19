from __future__ import annotations

from collections.abc import AsyncContextManager, Mapping, Sequence
from typing import Any, Protocol
from uuid import UUID

from kre.models import KnowledgeChunk, KnowledgeDocument, Provenance
from kre.storage.base import KnowledgeRepository
from kre.storage.postgres_schema import PostgresSchemaConfig


class PostgresConnection(Protocol):
    async def execute(self, query: str, *args: Any) -> str: ...

    async def fetchrow(self, query: str, *args: Any) -> Mapping[str, Any] | None: ...

    async def fetch(self, query: str, *args: Any) -> Sequence[Mapping[str, Any]]: ...

    def transaction(self) -> AsyncContextManager[Any]: ...


class PostgresPool(Protocol):
    def acquire(self) -> AsyncContextManager[PostgresConnection]: ...


class PostgresKnowledgeRepository(KnowledgeRepository):
    """Async PostgreSQL adapter for canonical KRE documents and chunks."""

    def __init__(self, pool: PostgresPool, *, schema: str = "kre") -> None:
        PostgresSchemaConfig(schema=schema)
        self._pool = pool
        self._schema = schema

    async def upsert_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        query = f"""
        INSERT INTO {self._schema}.knowledge_documents (
            id, title, content, mime_type, language, owner, classification,
            security_labels, tags, created_at, modified_at, provenance,
            ingested_at, content_hash, source_system, source_uri, source_version
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
            $13, $14, $15, $16, $17
        )
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            mime_type = EXCLUDED.mime_type,
            language = EXCLUDED.language,
            owner = EXCLUDED.owner,
            classification = EXCLUDED.classification,
            security_labels = EXCLUDED.security_labels,
            tags = EXCLUDED.tags,
            created_at = EXCLUDED.created_at,
            modified_at = EXCLUDED.modified_at,
            provenance = EXCLUDED.provenance,
            ingested_at = EXCLUDED.ingested_at,
            content_hash = EXCLUDED.content_hash,
            source_system = EXCLUDED.source_system,
            source_uri = EXCLUDED.source_uri,
            source_version = EXCLUDED.source_version
        RETURNING *
        """
        provenance = document.provenance.model_dump(mode="json")
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                query,
                document.id,
                document.title,
                document.content,
                document.mime_type,
                document.language,
                document.owner,
                document.classification.value,
                list(document.security_labels),
                list(document.tags),
                document.created_at,
                document.modified_at,
                provenance,
                document.provenance.ingested_at,
                document.provenance.content_hash,
                document.provenance.source_system,
                str(document.provenance.source_uri) if document.provenance.source_uri else None,
                document.provenance.source_version,
            )
        if row is None:
            raise RuntimeError("document upsert did not return a row")
        return self._document(row)

    async def get_document(self, document_id: UUID) -> KnowledgeDocument | None:
        query = f"SELECT * FROM {self._schema}.knowledge_documents WHERE id = $1"
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(query, document_id)
        return self._document(row) if row is not None else None

    async def delete_document(self, document_id: UUID) -> bool:
        query = f"DELETE FROM {self._schema}.knowledge_documents WHERE id = $1"
        async with self._pool.acquire() as connection:
            status = await connection.execute(query, document_id)
        return self._affected(status) > 0

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
        query = (
            f"SELECT * FROM {self._schema}.knowledge_documents "
            "ORDER BY id LIMIT $1 OFFSET $2"
        )
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(query, limit, offset)
        return [self._document(row) for row in rows]

    async def replace_chunks(
        self,
        document_id: UUID,
        chunks: Sequence[KnowledgeChunk],
    ) -> Sequence[KnowledgeChunk]:
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("every chunk must belong to the target document")
        ordered = sorted(chunks, key=lambda chunk: chunk.sequence)
        sequences = [chunk.sequence for chunk in ordered]
        if len(sequences) != len(set(sequences)):
            raise ValueError("chunk sequence values must be unique")

        exists_query = f"SELECT id FROM {self._schema}.knowledge_documents WHERE id = $1"
        delete_query = f"DELETE FROM {self._schema}.knowledge_chunks WHERE document_id = $1"
        insert_query = f"""
        INSERT INTO {self._schema}.knowledge_chunks (
            id, document_id, sequence, text, token_count, section, page, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                if await connection.fetchrow(exists_query, document_id) is None:
                    raise KeyError(f"document does not exist: {document_id}")
                await connection.execute(delete_query, document_id)
                for chunk in ordered:
                    await connection.execute(
                        insert_query,
                        chunk.id,
                        chunk.document_id,
                        chunk.sequence,
                        chunk.text,
                        chunk.token_count,
                        chunk.section,
                        chunk.page,
                        dict(chunk.metadata),
                    )
        return [chunk.model_copy(deep=True) for chunk in ordered]

    async def list_chunks(self, document_id: UUID) -> Sequence[KnowledgeChunk]:
        query = (
            f"SELECT * FROM {self._schema}.knowledge_chunks "
            "WHERE document_id = $1 ORDER BY sequence, id"
        )
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(query, document_id)
        return [self._chunk(row) for row in rows]

    @staticmethod
    def _affected(status: str) -> int:
        try:
            return int(status.rsplit(" ", maxsplit=1)[-1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"unexpected PostgreSQL command status: {status}") from exc

    @staticmethod
    def _document(row: Mapping[str, Any]) -> KnowledgeDocument:
        provenance_data = dict(row["provenance"])
        provenance_data.update(
            {
                "ingested_at": row["ingested_at"],
                "content_hash": row["content_hash"],
                "source_system": row["source_system"],
                "source_uri": row["source_uri"],
                "source_version": row["source_version"],
            }
        )
        return KnowledgeDocument(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            mime_type=row["mime_type"],
            language=row["language"],
            owner=row["owner"],
            classification=row["classification"],
            security_labels=list(row["security_labels"]),
            tags=list(row["tags"]),
            created_at=row["created_at"],
            modified_at=row["modified_at"],
            provenance=Provenance.model_validate(provenance_data),
        )

    @staticmethod
    def _chunk(row: Mapping[str, Any]) -> KnowledgeChunk:
        return KnowledgeChunk(
            id=row["id"],
            document_id=row["document_id"],
            sequence=row["sequence"],
            text=row["text"],
            token_count=row["token_count"],
            section=row["section"],
            page=row["page"],
            metadata=dict(row["metadata"]),
        )
