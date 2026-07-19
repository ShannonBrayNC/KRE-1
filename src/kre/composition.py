from __future__ import annotations

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
    SemanticIndex,
    SemanticRetrievalService,
)
from kre.storage import InMemoryKnowledgeRepository, KnowledgeRepository
from kre.telemetry import InMemoryRetrievalTelemetry, TelemetrySearchBackend


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
    """Construct the default KRE dependency graph from validated settings."""

    repository = InMemoryKnowledgeRepository()
    semantic_index = InMemorySemanticIndex()
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
