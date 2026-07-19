from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from kre.application import SearchApplicationBackend
from kre.config import KRESettings
from kre.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from kre.search import (
    HybridSearchService,
    InMemorySemanticIndex,
    KeywordSearch,
    PgVectorSemanticIndex,
    SemanticIndex,
    SemanticRetrievalService,
)
from kre.storage import (
    InMemoryKnowledgeRepository,
    KnowledgeRepository,
    PostgresKnowledgeRepository,
)
from kre.storage.postgres import PostgresPool
from kre.telemetry import InMemoryRetrievalTelemetry, TelemetrySearchBackend

PostgresPoolFactory = Callable[[str], Awaitable[PostgresPool]]


@dataclass(frozen=True, slots=True)
class KREComponents:
    """Process-scoped KRE services produced by the composition root."""

    settings: KRESettings
    repository: KnowledgeRepository
    semantic_index: SemanticIndex
    embeddings: EmbeddingProvider
    telemetry: InMemoryRetrievalTelemetry
    search_backend: TelemetrySearchBackend


def build_components(settings: KRESettings) -> KREComponents:
    """Construct the synchronous in-memory dependency graph."""

    if settings.storage_provider.casefold() != "memory":
        raise RuntimeError("postgres storage requires build_components_async")
    return _assemble(
        settings,
        repository=InMemoryKnowledgeRepository(),
        semantic_index=InMemorySemanticIndex(),
    )


async def build_components_async(
    settings: KRESettings,
    *,
    postgres_pool_factory: PostgresPoolFactory | None = None,
) -> KREComponents:
    """Construct memory or durable PostgreSQL components from validated settings."""

    if settings.storage_provider.casefold() == "memory":
        return build_components(settings)
    if postgres_pool_factory is None:
        raise RuntimeError("postgres_pool_factory is required for postgres storage")

    pool = await postgres_pool_factory(settings.postgres_dsn or "")
    repository = PostgresKnowledgeRepository(pool, schema=settings.postgres_schema)
    semantic_index = PgVectorSemanticIndex(
        pool,
        schema=settings.postgres_schema,
        vector_dimensions=settings.embedding_dimensions,
    )
    return _assemble(settings, repository=repository, semantic_index=semantic_index)


def _assemble(
    settings: KRESettings,
    *,
    repository: KnowledgeRepository,
    semantic_index: SemanticIndex,
) -> KREComponents:
    embeddings = build_embedding_provider(settings)
    keyword = KeywordSearch(repository)
    semantic = SemanticRetrievalService(repository, semantic_index, embeddings)
    hybrid = HybridSearchService(keyword, semantic)
    application_backend = SearchApplicationBackend(repository, keyword, semantic, hybrid)
    telemetry = InMemoryRetrievalTelemetry()
    backend = TelemetrySearchBackend(application_backend, telemetry)
    return KREComponents(
        settings=settings,
        repository=repository,
        semantic_index=semantic_index,
        embeddings=embeddings,
        telemetry=telemetry,
        search_backend=backend,
    )


def build_embedding_provider(settings: KRESettings) -> EmbeddingProvider:
    """Select one embedding adapter without exposing provider details downstream."""

    if settings.embedding_provider.casefold() == "deterministic":
        return DeterministicEmbeddingProvider(
            dimensions=settings.embedding_dimensions,
            model=settings.embedding_model,
        )
    return OpenAICompatibleEmbeddingProvider(
        endpoint=settings.embedding_endpoint or "",
        api_key=settings.embedding_api_key or "",
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        api_version=settings.embedding_api_version,
        api_key_header=settings.embedding_api_key_header,
    )
