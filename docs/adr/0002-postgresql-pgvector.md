# ADR 0002: PostgreSQL and pgvector for durable KRE persistence

- Status: Accepted
- Date: 2026-07-19
- Decision owners: Lantern Platform / SignalForge governance

## Context

KRE requires durable canonical storage, transactional chunk replacement, provenance preservation, and semantic vector retrieval. The current in-memory repository and semantic index are reference implementations for tests and local development; they are not suitable for multi-process or horizontally scaled deployments.

KRE must preserve provider neutrality at the application boundary while establishing one operationally supportable default persistence platform.

## Decision

Use PostgreSQL as the default durable system of record and pgvector as the default semantic-vector extension.

The durable adapter will implement the existing `KnowledgeRepository` and `SemanticIndex` contracts. Application, search, connector, and embedding code will not import PostgreSQL-specific types.

The initial physical model contains:

- `knowledge_documents` for canonical documents, provenance, classification, labels, tags, and source timestamps.
- `knowledge_chunks` for ordered document chunks and chunk metadata.
- `semantic_embeddings` for model-qualified vectors linked to canonical chunks.

Document deletion cascades to chunks and embeddings. Chunk replacement must execute transactionally so readers never observe a partially replaced document. Embeddings are unique per `(chunk_id, model)` and retain explicit model and dimension metadata.

Cosine distance is the default semantic operator. An HNSW index using `vector_cosine_ops` is the initial approximate-nearest-neighbor strategy. Exact search remains available for validation and small corpora.

## Security and governance

Classification and security labels remain canonical document attributes. Security trimming remains an application boundary and is not delegated solely to vector similarity queries.

Production deployments should additionally apply:

- TLS for database connections.
- Managed identity or secret-store-backed credentials.
- Least-privilege database roles.
- Row-level security where tenant isolation requires database enforcement.
- Migration approval through SignalForge governance.
- Backup, restore, retention, and disaster-recovery policies appropriate to the hosted environment.

## Migration strategy

1. Create the `vector` extension and KRE schema.
2. Create canonical document and chunk tables.
3. Create semantic embedding table and indexes.
4. Implement async PostgreSQL repository and semantic-index adapters.
5. Run contract tests against ephemeral PostgreSQL with pgvector.
6. Add dual-write or controlled reindex tooling before production cutover.

Schema changes are forward-only migrations. Vector dimension changes require a new embedding column/table or a controlled rebuild; they must not silently reinterpret existing vectors.

## Consequences

### Positive

- One transactional system supports canonical data and semantic vectors.
- PostgreSQL operational tooling, backups, and observability are mature.
- Existing provider-neutral interfaces remain intact.
- pgvector supports exact and approximate cosine retrieval.

### Negative

- pgvector becomes an operational dependency for the default deployment.
- Embedding-model or dimension changes require explicit migration and reindexing.
- Large-scale search may later require a specialized search service.

## Alternatives considered

### Azure AI Search

Retained as a future `SemanticIndex` adapter. It provides managed hybrid-search capabilities but should not become the canonical document system of record.

### Separate vector database

Deferred. It adds another consistency boundary and operational surface before KRE has demonstrated a scale requirement that PostgreSQL cannot meet.

### PostgreSQL without pgvector

Rejected because it would require a second semantic store immediately and would complicate atomic lifecycle management between canonical chunks and vectors.
