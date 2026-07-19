from __future__ import annotations

from datetime import datetime, timezone
from os import environ

import pytest

from kre.embeddings import EmbeddingVector
from kre.models import Classification, KnowledgeChunk, KnowledgeDocument, Provenance
from kre.search import PgVectorSemanticIndex, SemanticRecord
from kre.storage import PostgresKnowledgeRepository, PostgresSchemaConfig, render_postgres_schema

asyncpg = pytest.importorskip("asyncpg")


@pytest.mark.asyncio
async def test_postgres_repository_and_pgvector_index_round_trip() -> None:
    dsn = environ.get("KRE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KRE_TEST_POSTGRES_DSN is not configured")

    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2)
    try:
        schema = "kre_integration"
        dimensions = 3
        async with pool.acquire() as connection:
            await connection.execute(
                render_postgres_schema(
                    PostgresSchemaConfig(schema=schema, vector_dimensions=dimensions)
                )
            )

        repository = PostgresKnowledgeRepository(pool, schema=schema)
        index = PgVectorSemanticIndex(pool, schema=schema, vector_dimensions=dimensions)
        document = KnowledgeDocument(
            title="Lantern durable retrieval",
            content="PostgreSQL and pgvector integration contract",
            classification=Classification.PUBLIC,
            security_labels=["integration"],
            provenance=Provenance(
                source_system="integration-test",
                connector="pytest",
                content_hash="sha256:integration",
                ingested_at=datetime.now(timezone.utc),
            ),
        )
        chunk = KnowledgeChunk(
            document_id=document.id,
            sequence=0,
            text=document.content,
            token_count=5,
            metadata={"suite": "postgres-pgvector"},
        )
        vector = EmbeddingVector(
            values=(1.0, 0.0, 0.0),
            model="integration-v1",
            dimensions=dimensions,
        )

        stored = await repository.upsert_document(document)
        assert stored == document
        assert await repository.replace_chunks(document.id, [chunk]) == [chunk]

        await index.replace_document(
            document.id,
            [
                SemanticRecord(
                    document_id=document.id,
                    chunk_id=chunk.id,
                    sequence=chunk.sequence,
                    vector=vector,
                )
            ],
        )
        results = await index.search(vector, limit=5, minimum_score=0.99)

        assert len(results) == 1
        assert results[0].document_id == document.id
        assert results[0].chunk_id == chunk.id
        assert results[0].score == 1.0

        assert await repository.delete_document(document.id) is True
        assert await repository.get_document(document.id) is None
        assert await repository.list_chunks(document.id) == []
        assert await index.search(vector, limit=5, minimum_score=-1.0) == []
    finally:
        await pool.close()
