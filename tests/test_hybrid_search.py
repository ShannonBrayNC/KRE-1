from __future__ import annotations

import pytest

from kre.embeddings import DeterministicEmbeddingProvider
from kre.models import KnowledgeChunk, KnowledgeDocument, Provenance
from kre.search import (
    HybridSearchService,
    InMemorySemanticIndex,
    KeywordSearch,
    SemanticRecord,
    SemanticRetrievalService,
)
from kre.storage import InMemoryKnowledgeRepository


async def seed_hybrid_service() -> tuple[HybridSearchService, KnowledgeChunk, KnowledgeChunk]:
    repository = InMemoryKnowledgeRepository()
    embeddings = DeterministicEmbeddingProvider(dimensions=8)
    index = InMemorySemanticIndex()

    first_document = KnowledgeDocument(
        title="Lantern Governance",
        content="",
        provenance=Provenance(
            source_system="github",
            source_uri="https://github.com/ShannonBrayNC/KRE-1",
            connector="github",
            content_hash="sha256:first",
        ),
    )
    second_document = KnowledgeDocument(
        title="Operations",
        content="",
        provenance=Provenance(
            source_system="test",
            connector="test",
            content_hash="sha256:second",
        ),
    )
    first_chunk = KnowledgeChunk(
        document_id=first_document.id,
        sequence=0,
        text="SignalForge governs canonical Lantern registry decisions.",
    )
    second_chunk = KnowledgeChunk(
        document_id=second_document.id,
        sequence=0,
        text="Operational registry synchronization guidance.",
    )

    for document, chunk in (
        (first_document, first_chunk),
        (second_document, second_chunk),
    ):
        await repository.upsert_document(document)
        await repository.replace_chunks(document.id, [chunk])
        [vector] = await embeddings.embed([chunk.text])
        await index.replace_document(
            document.id,
            [SemanticRecord(document.id, chunk.id, chunk.sequence, vector)],
        )

    service = HybridSearchService(
        KeywordSearch(repository),
        SemanticRetrievalService(repository, index, embeddings),
    )
    return service, first_chunk, second_chunk


@pytest.mark.asyncio
async def test_hybrid_search_rewards_results_present_in_both_rankings() -> None:
    service, first_chunk, _ = await seed_hybrid_service()

    results = await service.search("canonical Lantern registry", limit=2)

    assert results[0].chunk_id == first_chunk.id
    assert results[0].keyword_rank == 1
    assert results[0].semantic_rank is not None
    assert results[0].keyword_score is not None
    assert results[0].semantic_score is not None
    assert results[0].source_system == "github"
    assert results[0].source_uri is not None


@pytest.mark.asyncio
async def test_hybrid_search_keeps_single_channel_candidates() -> None:
    service, _, second_chunk = await seed_hybrid_service()

    results = await service.search("operational synchronization", limit=2)

    assert any(result.chunk_id == second_chunk.id for result in results)
    assert all(result.score > 0 for result in results)


@pytest.mark.asyncio
async def test_hybrid_search_is_deterministic() -> None:
    service, _, _ = await seed_hybrid_service()

    first = await service.search("registry", limit=2)
    second = await service.search("registry", limit=2)

    assert first == second


@pytest.mark.asyncio
async def test_hybrid_search_validates_options() -> None:
    service, _, _ = await seed_hybrid_service()

    with pytest.raises(ValueError, match="empty"):
        await service.search("   ")
    with pytest.raises(ValueError, match="limit"):
        await service.search("registry", limit=0)
    with pytest.raises(ValueError, match="candidate_limit"):
        await service.search("registry", limit=5, candidate_limit=2)
    with pytest.raises(TypeError, match="string"):
        await service.search(42)  # type: ignore[arg-type]


def test_hybrid_search_validates_rank_constant() -> None:
    with pytest.raises(ValueError, match="rank_constant"):
        HybridSearchService(None, None, rank_constant=0)  # type: ignore[arg-type]
