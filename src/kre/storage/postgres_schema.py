from __future__ import annotations

import re
from dataclasses import dataclass

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class PostgresSchemaConfig:
    """Validated inputs for rendering the durable KRE database schema."""

    schema: str = "kre"
    vector_dimensions: int = 1536

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.schema):
            raise ValueError("schema must be a lowercase PostgreSQL identifier")
        if self.vector_dimensions < 1 or self.vector_dimensions > 16_000:
            raise ValueError("vector_dimensions must be between 1 and 16000")


def render_postgres_schema(config: PostgresSchemaConfig) -> str:
    """Render the initial PostgreSQL and pgvector migration contract."""

    schema = config.schema
    dimensions = config.vector_dimensions
    return f"""CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.knowledge_documents (
    id uuid PRIMARY KEY,
    title text NOT NULL,
    content text NOT NULL,
    mime_type text NOT NULL,
    language text NOT NULL,
    owner text NULL,
    classification text NOT NULL CHECK (
        classification IN ('public', 'internal', 'confidential', 'restricted')
    ),
    security_labels jsonb NOT NULL DEFAULT '[]'::jsonb,
    tags jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NULL,
    modified_at timestamptz NULL,
    provenance jsonb NOT NULL,
    ingested_at timestamptz NOT NULL,
    content_hash text NOT NULL,
    source_system text NOT NULL,
    source_uri text NULL,
    source_version text NULL
);

CREATE INDEX IF NOT EXISTS ix_knowledge_documents_source_hash
    ON {schema}.knowledge_documents (source_system, content_hash);

CREATE INDEX IF NOT EXISTS ix_knowledge_documents_source_identity
    ON {schema}.knowledge_documents (source_system, source_uri, source_version);

CREATE TABLE IF NOT EXISTS {schema}.knowledge_chunks (
    id uuid PRIMARY KEY,
    document_id uuid NOT NULL REFERENCES {schema}.knowledge_documents(id) ON DELETE CASCADE,
    sequence integer NOT NULL CHECK (sequence >= 0),
    text text NOT NULL,
    token_count integer NULL CHECK (token_count >= 0),
    section text NULL,
    page integer NULL CHECK (page >= 1),
    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    UNIQUE (document_id, sequence)
);

CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_document
    ON {schema}.knowledge_chunks (document_id, sequence);

CREATE TABLE IF NOT EXISTS {schema}.semantic_embeddings (
    chunk_id uuid NOT NULL REFERENCES {schema}.knowledge_chunks(id) ON DELETE CASCADE,
    model text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions = {dimensions}),
    embedding vector({dimensions}) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, model)
);

CREATE INDEX IF NOT EXISTS ix_semantic_embeddings_cosine_hnsw
    ON {schema}.semantic_embeddings
    USING hnsw (embedding vector_cosine_ops);
"""
