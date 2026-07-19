from __future__ import annotations

import pytest

from kre.storage import PostgresSchemaConfig, render_postgres_schema


def test_postgres_schema_contains_canonical_and_vector_tables() -> None:
    sql = render_postgres_schema(PostgresSchemaConfig(vector_dimensions=8))

    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE SCHEMA IF NOT EXISTS kre" in sql
    assert "kre.knowledge_documents" in sql
    assert "kre.knowledge_chunks" in sql
    assert "kre.semantic_embeddings" in sql
    assert "embedding vector(8) NOT NULL" in sql
    assert "CHECK (dimensions = 8)" in sql
    assert "USING hnsw (embedding vector_cosine_ops)" in sql


def test_postgres_schema_enforces_lifecycle_and_identity_constraints() -> None:
    sql = render_postgres_schema(PostgresSchemaConfig())

    assert sql.count("ON DELETE CASCADE") == 2
    assert "UNIQUE (document_id, sequence)" in sql
    assert "PRIMARY KEY (chunk_id, model)" in sql
    assert "classification IN ('public', 'internal', 'confidential', 'restricted')" in sql
    assert "ix_knowledge_documents_source_hash" in sql
    assert "ix_knowledge_documents_source_identity" in sql
    assert "CREATE UNIQUE INDEX IF NOT EXISTS ux_knowledge_documents_source_hash" not in sql
    assert "document_id uuid NOT NULL REFERENCES kre.knowledge_documents" not in sql.split(
        "CREATE TABLE IF NOT EXISTS kre.semantic_embeddings", maxsplit=1
    )[1]


def test_postgres_schema_supports_governed_schema_names() -> None:
    sql = render_postgres_schema(
        PostgresSchemaConfig(schema="lantern_kre", vector_dimensions=3072)
    )

    assert "CREATE SCHEMA IF NOT EXISTS lantern_kre" in sql
    assert "lantern_kre.knowledge_documents" in sql
    assert "vector(3072)" in sql


@pytest.mark.parametrize("schema", ["", "KRE", "kre-prod", "1kre", "kre;drop"])
def test_postgres_schema_rejects_unsafe_identifiers(schema: str) -> None:
    with pytest.raises(ValueError, match="identifier"):
        PostgresSchemaConfig(schema=schema)


@pytest.mark.parametrize("dimensions", [0, -1, 16_001])
def test_postgres_schema_rejects_invalid_dimensions(dimensions: int) -> None:
    with pytest.raises(ValueError, match="vector_dimensions"):
        PostgresSchemaConfig(vector_dimensions=dimensions)
