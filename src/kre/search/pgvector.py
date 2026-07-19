from __future__ import annotations

import math
from collections.abc import Sequence
from uuid import UUID

from kre.embeddings import EmbeddingVector
from kre.search.semantic import (
    SemanticIndex,
    SemanticRecord,
    SemanticSearchResult,
)
from kre.storage.postgres import PostgresPool
from kre.storage.postgres_schema import PostgresSchemaConfig


class PgVectorSemanticIndex(SemanticIndex):
    """PostgreSQL/pgvector adapter for durable semantic indexing."""

    def __init__(
        self,
        pool: PostgresPool,
        *,
        schema: str = "kre",
        vector_dimensions: int = 1536,
    ) -> None:
        PostgresSchemaConfig(schema=schema, vector_dimensions=vector_dimensions)
        self._pool = pool
        self._schema = schema
        self._dimensions = vector_dimensions

    async def replace_document(
        self,
        document_id: UUID,
        records: Sequence[SemanticRecord],
    ) -> None:
        self._validate_records(document_id, records)
        delete_query = f"""
        DELETE FROM {self._schema}.semantic_embeddings AS embedding
        USING {self._schema}.knowledge_chunks AS chunk
        WHERE embedding.chunk_id = chunk.id AND chunk.document_id = $1
        """
        insert_query = f"""
        INSERT INTO {self._schema}.semantic_embeddings (
            chunk_id, model, dimensions, embedding
        ) VALUES ($1, $2, $3, $4::vector)
        ON CONFLICT (chunk_id, model) DO UPDATE SET
            dimensions = EXCLUDED.dimensions,
            embedding = EXCLUDED.embedding,
            created_at = now()
        """
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(delete_query, document_id)
                for record in records:
                    await connection.execute(
                        insert_query,
                        record.chunk_id,
                        record.vector.model,
                        record.vector.dimensions,
                        self._vector_literal(record.vector),
                    )

    async def delete_document(self, document_id: UUID) -> bool:
        query = f"""
        DELETE FROM {self._schema}.semantic_embeddings AS embedding
        USING {self._schema}.knowledge_chunks AS chunk
        WHERE embedding.chunk_id = chunk.id AND chunk.document_id = $1
        """
        async with self._pool.acquire() as connection:
            status = await connection.execute(query, document_id)
        return self._affected(status) > 0

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
        if query.dimensions != self._dimensions:
            raise ValueError("query dimensions do not match the configured pgvector index")
        if any(not math.isfinite(value) for value in query.values):
            raise ValueError("query vector values must be finite")

        sql = f"""
        SELECT
            chunk.document_id,
            embedding.chunk_id,
            chunk.sequence,
            1 - (embedding.embedding <=> $1::vector) AS score
        FROM {self._schema}.semantic_embeddings AS embedding
        JOIN {self._schema}.knowledge_chunks AS chunk ON chunk.id = embedding.chunk_id
        WHERE embedding.model = $2
          AND embedding.dimensions = $3
          AND 1 - (embedding.embedding <=> $1::vector) >= $4
        ORDER BY score DESC, chunk.document_id, chunk.sequence, embedding.chunk_id
        LIMIT $5
        """
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                sql,
                self._vector_literal(query),
                query.model,
                query.dimensions,
                minimum_score,
                limit,
            )
        return [
            SemanticSearchResult(
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
                sequence=row["sequence"],
                score=round(float(row["score"]), 8),
            )
            for row in rows
        ]

    def _validate_records(
        self,
        document_id: UUID,
        records: Sequence[SemanticRecord],
    ) -> None:
        if any(record.document_id != document_id for record in records):
            raise ValueError("every semantic record must belong to the target document")
        chunk_ids = [record.chunk_id for record in records]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("semantic record chunk identifiers must be unique")
        models = {record.vector.model for record in records}
        dimensions = {record.vector.dimensions for record in records}
        if len(models) > 1 or len(dimensions) > 1:
            raise ValueError("semantic records for a document must share model and dimensions")
        if dimensions and dimensions != {self._dimensions}:
            raise ValueError("semantic record dimensions do not match the configured pgvector index")
        if any(
            not math.isfinite(value)
            for record in records
            for value in record.vector.values
        ):
            raise ValueError("semantic record vector values must be finite")

    @staticmethod
    def _vector_literal(vector: EmbeddingVector) -> str:
        return "[" + ",".join(format(value, ".17g") for value in vector.values) + "]"

    @staticmethod
    def _affected(status: str) -> int:
        try:
            return int(status.rsplit(" ", maxsplit=1)[-1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"unexpected PostgreSQL command status: {status}") from exc
