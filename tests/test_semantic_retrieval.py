from __future__ import annotations

from collections.abc import Sequence

import pytest

from kre.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider, EmbeddingVector
from kre.models import Classification, KnowledgeChunk, KnowledgeDocument, Provenance
from kre.search import (
    InMemorySemanticIndex,
    SemanticRecord,
    SemanticRetrievalService,
)
from kre.storage import InMemoryKnowledgeRepository


async def seed_service() -> tuple[SemanticRetrievalService, KnowledgeDocument, KnowledgeChunk]:
    repository = InMemoryKnowledgeRepository()
    embeddings = DeterministicEmbeddingProvider(dimensions=8)
    index = InMemorySemanticIndex()
    document = KnowledgeDocument(
        title="Lantern Governance",
        content="",
        classification=Classification.CONFIDENTIAL,
        security_labels=["lantern-engineering"],
        provenance=Provenance(
            source_system="github",
            source_uri="https://github.com/ShannonBrayNC/KRE-1",
            connector="github",
            content_hash="sha256:test",
            source_version="abc123",
        ),
    )
    chunk = KnowledgeChunk(
        document_id=document.id,
        sequence=0,
        text="SignalForge governs canonical Lantern registry decisions.",
        metadata={"section": "governance"},
    )
    await repository.upsert_document(document)
    await repository.replace_chunks(document.id, [chunk])
    [vector] = await embeddings.embed([chunk.text])
    await index.replace_document(
        document.id,
        [SemanticRecord(document.id, chunk.id, chunk.sequence, vector)],
    )
    return SemanticRetrievalService(repository, index, embeddings), document, chunk


@pytest.mark.asyncio
async def test_semantic_retrieval_resolves_content_and_provenance() -> None:
    service, document, chunk = await seed_service()

    results = await service.search(chunk.text)

    assert len(results) == 1
    result = results[0]
    assert result.document_id == document.id
    assert result.chunk_id == chunk.id
    assert result.document_title == "Lantern Governance"
    assert result.text == chunk.text
    assert result.score == pytest.approx(1.0)
    assert result.source_system == "github"
    assert result.source_version == "abc123"
    assert result.classification == "confidential"
    assert result.security_labels == ("lantern-engineering",)
    assert result.metadata == {"section": "governance"}


@pytest.mark.asyncio
async def test_semantic_retrieval_skips_stale_index_records() -> None:
    repository = InMemoryKnowledgeRepository()
    embeddings = DeterministicEmbeddingProvider(dimensions=8)
    index = InMemorySemanticIndex()
    missing_document = KnowledgeDocument(
        title="Missing",
        content="",
        provenance=Provenance(
            source_system="test",
            connector="test",
            content_hash="sha256:missing",
        ),
    )
    missing_chunk = KnowledgeChunk(
        document_id=missing_document.id,
        sequence=0,
        text="Stale semantic record",
    )
    [vector] = await embeddings.embed([missing_chunk.text])
    await index.replace_document(
        missing_document.id,
        [SemanticRecord(missing_document.id, missing_chunk.id, 0, vector)],
    )

    service = SemanticRetrievalService(repository, index, embeddings)

    assert await service.search(missing_chunk.text) == []


@pytest.mark.asyncio
async def test_semantic_retrieval_validates_query_and_limit() -> None:
    service, _, _ = await seed_service()

    with pytest.raises(ValueError, match="empty"):
        await service.search("   ")
    with pytest.raises(ValueError, match="limit"):
        await service.search("knowledge", limit=0)
    with pytest.raises(TypeError, match="string"):
        await service.search(42)  # type: ignore[arg-type]


class BrokenEmbeddingProvider(EmbeddingProvider):
    name = "broken"
    model = "broken-v1"
    dimensions = 2

    async def embed(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]:
        return []


@pytest.mark.asyncio
async def test_semantic_retrieval_rejects_broken_provider_cardinality() -> None:
    service = SemanticRetrievalService(
        InMemoryKnowledgeRepository(),
        InMemorySemanticIndex(),
        BrokenEmbeddingProvider(),
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        await service.search("query")
