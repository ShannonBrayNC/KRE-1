from __future__ import annotations

from uuid import uuid4

import pytest

from kre.models import KnowledgeChunk, KnowledgeDocument, Provenance
from kre.storage import InMemoryKnowledgeRepository


def make_document(title: str = "Document") -> KnowledgeDocument:
    return KnowledgeDocument(
        title=title,
        content="Governed knowledge",
        provenance=Provenance(
            source_system="test",
            connector="test",
            content_hash="sha256:test",
        ),
    )


@pytest.mark.asyncio
async def test_repository_upserts_and_returns_defensive_copies() -> None:
    repository = InMemoryKnowledgeRepository()
    document = make_document()

    stored = await repository.upsert_document(document)
    stored.title = "Mutated"

    loaded = await repository.get_document(document.id)

    assert loaded is not None
    assert loaded.title == "Document"


@pytest.mark.asyncio
async def test_repository_replaces_and_orders_chunks() -> None:
    repository = InMemoryKnowledgeRepository()
    document = make_document()
    await repository.upsert_document(document)

    chunks = [
        KnowledgeChunk(document_id=document.id, sequence=1, text="Second"),
        KnowledgeChunk(document_id=document.id, sequence=0, text="First"),
    ]

    stored = await repository.replace_chunks(document.id, chunks)

    assert [chunk.sequence for chunk in stored] == [0, 1]
    assert [chunk.text for chunk in await repository.list_chunks(document.id)] == ["First", "Second"]


@pytest.mark.asyncio
async def test_repository_rejects_invalid_chunk_replacement() -> None:
    repository = InMemoryKnowledgeRepository()
    document = make_document()
    await repository.upsert_document(document)

    with pytest.raises(ValueError, match="belong"):
        await repository.replace_chunks(
            document.id,
            [KnowledgeChunk(document_id=uuid4(), sequence=0, text="Wrong")],
        )

    with pytest.raises(ValueError, match="unique"):
        await repository.replace_chunks(
            document.id,
            [
                KnowledgeChunk(document_id=document.id, sequence=0, text="A"),
                KnowledgeChunk(document_id=document.id, sequence=0, text="B"),
            ],
        )


@pytest.mark.asyncio
async def test_repository_delete_cascades_chunks() -> None:
    repository = InMemoryKnowledgeRepository()
    document = make_document()
    await repository.upsert_document(document)
    await repository.replace_chunks(
        document.id,
        [KnowledgeChunk(document_id=document.id, sequence=0, text="Only")],
    )

    assert await repository.delete_document(document.id) is True
    assert await repository.get_document(document.id) is None
    assert await repository.list_chunks(document.id) == []
    assert await repository.delete_document(document.id) is False


@pytest.mark.asyncio
async def test_repository_lists_documents_with_validated_paging() -> None:
    repository = InMemoryKnowledgeRepository()
    documents = [make_document(str(index)) for index in range(3)]
    for document in documents:
        await repository.upsert_document(document)

    page = await repository.list_documents(limit=2, offset=1)
    expected = sorted(documents, key=lambda item: str(item.id))[1:3]

    assert [item.id for item in page] == [item.id for item in expected]

    with pytest.raises(ValueError, match="limit"):
        await repository.list_documents(limit=0)
    with pytest.raises(ValueError, match="offset"):
        await repository.list_documents(offset=-1)