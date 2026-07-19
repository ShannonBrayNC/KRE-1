from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from kre.composition import build_components
from kre.config import KRESettings
from kre.ingestion import IngestionConsistencyError, KnowledgeIngestionService
from kre.models import Classification, KnowledgeChunk, KnowledgeDocument, Provenance
from kre.search import InMemorySemanticIndex


DOCUMENT_ID = UUID("40000000-0000-0000-0000-000000000001")
CHUNK_ID = UUID("50000000-0000-0000-0000-000000000001")


def make_document() -> KnowledgeDocument:
    return KnowledgeDocument(
        id=DOCUMENT_ID,
        title="Governed ingestion",
        content="Lantern durable ingestion and semantic reindex orchestration",
        classification=Classification.INTERNAL,
        provenance=Provenance(
            source_system="test",
            connector="pytest",
            content_hash="sha256:ingestion",
            ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    )


def make_chunk() -> KnowledgeChunk:
    return KnowledgeChunk(
        id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        sequence=0,
        text="Lantern durable ingestion and semantic reindex orchestration",
        token_count=7,
    )


@pytest.mark.asyncio
async def test_ingest_reindex_and_delete_use_composed_services() -> None:
    components = build_components(KRESettings(embedding_dimensions=8))
    document = make_document()
    chunk = make_chunk()

    result = await components.ingestion.ingest(document, (chunk,))

    assert result.document_id == DOCUMENT_ID
    assert result.chunk_count == 1
    assert result.embedding_dimensions == 8
    assert await components.repository.get_document(DOCUMENT_ID) == document
    assert await components.repository.list_chunks(DOCUMENT_ID) == [chunk]

    query = (await components.embeddings.embed((chunk.text,)))[0]
    matches = await components.semantic_index.search(query, limit=1, minimum_score=0.99)
    assert len(matches) == 1
    assert matches[0].document_id == DOCUMENT_ID
    assert matches[0].chunk_id == CHUNK_ID

    reindexed = await components.ingestion.reindex(DOCUMENT_ID)
    assert reindexed.chunk_count == 1
    assert await components.ingestion.delete(DOCUMENT_ID) is True
    assert await components.repository.get_document(DOCUMENT_ID) is None
    assert await components.semantic_index.search(query, limit=1) == []


class FailingIndex(InMemorySemanticIndex):
    def __init__(self, *, fail_cleanup: bool = False) -> None:
        super().__init__()
        self.fail_cleanup = fail_cleanup
        self.cleanup_attempted = False

    async def replace_document(self, document_id, records) -> None:
        raise RuntimeError("index write failed")

    async def delete_document(self, document_id) -> bool:
        self.cleanup_attempted = True
        if self.fail_cleanup:
            raise RuntimeError("index cleanup failed")
        return await super().delete_document(document_id)


@pytest.mark.asyncio
async def test_failed_index_write_attempts_semantic_cleanup() -> None:
    components = build_components(KRESettings())
    index = FailingIndex()
    service = KnowledgeIngestionService(
        components.repository,
        index,
        components.embeddings,
    )

    with pytest.raises(RuntimeError, match="index write failed"):
        await service.ingest(make_document(), (make_chunk(),))

    assert index.cleanup_attempted is True


@pytest.mark.asyncio
async def test_cleanup_failure_is_reported_as_consistency_error() -> None:
    components = build_components(KRESettings())
    service = KnowledgeIngestionService(
        components.repository,
        FailingIndex(fail_cleanup=True),
        components.embeddings,
    )

    with pytest.raises(IngestionConsistencyError, match="semantic cleanup failed"):
        await service.ingest(make_document(), (make_chunk(),))


@pytest.mark.asyncio
async def test_ingestion_rejects_invalid_chunks_and_missing_reindex() -> None:
    components = build_components(KRESettings())
    chunk = make_chunk().model_copy(update={"document_id": UUID(int=9)})

    with pytest.raises(ValueError, match="target document"):
        await components.ingestion.ingest(make_document(), (chunk,))
    with pytest.raises(KeyError, match="does not exist"):
        await components.ingestion.reindex(DOCUMENT_ID)
