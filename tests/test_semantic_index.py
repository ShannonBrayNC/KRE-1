from __future__ import annotations

from uuid import uuid4

import pytest

from kre.embeddings import EmbeddingVector
from kre.search import InMemorySemanticIndex, SemanticRecord


def vector(*values: float, model: str = "test-v1") -> EmbeddingVector:
    return EmbeddingVector(values=values, model=model, dimensions=len(values))


@pytest.mark.asyncio
async def test_semantic_index_ranks_cosine_similarity_deterministically() -> None:
    index = InMemorySemanticIndex()
    first_document = uuid4()
    second_document = uuid4()
    first_chunk = uuid4()
    second_chunk = uuid4()

    await index.replace_document(
        first_document,
        [SemanticRecord(first_document, first_chunk, 0, vector(1.0, 0.0))],
    )
    await index.replace_document(
        second_document,
        [SemanticRecord(second_document, second_chunk, 0, vector(0.8, 0.2))],
    )

    results = await index.search(vector(1.0, 0.0), limit=2)

    assert [result.chunk_id for result in results] == [first_chunk, second_chunk]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_semantic_index_replaces_and_deletes_document_records() -> None:
    index = InMemorySemanticIndex()
    document_id = uuid4()
    old_chunk = uuid4()
    new_chunk = uuid4()

    await index.replace_document(
        document_id,
        [SemanticRecord(document_id, old_chunk, 0, vector(1.0, 0.0))],
    )
    await index.replace_document(
        document_id,
        [SemanticRecord(document_id, new_chunk, 0, vector(0.0, 1.0))],
    )

    results = await index.search(vector(0.0, 1.0))

    assert [result.chunk_id for result in results] == [new_chunk]
    assert await index.delete_document(document_id) is True
    assert await index.search(vector(0.0, 1.0)) == []
    assert await index.delete_document(document_id) is False


@pytest.mark.asyncio
async def test_semantic_index_filters_incompatible_models_and_scores() -> None:
    index = InMemorySemanticIndex()
    document_id = uuid4()
    await index.replace_document(
        document_id,
        [SemanticRecord(document_id, uuid4(), 0, vector(1.0, 0.0, model="model-a"))],
    )

    assert await index.search(vector(1.0, 0.0, model="model-b")) == []
    assert await index.search(vector(-1.0, 0.0, model="model-a"), minimum_score=0.0) == []


@pytest.mark.asyncio
async def test_semantic_index_validates_records_and_query_options() -> None:
    index = InMemorySemanticIndex()
    document_id = uuid4()
    foreign_document = uuid4()
    chunk_id = uuid4()

    with pytest.raises(ValueError, match="belong"):
        await index.replace_document(
            document_id,
            [SemanticRecord(foreign_document, chunk_id, 0, vector(1.0, 0.0))],
        )

    with pytest.raises(ValueError, match="unique"):
        await index.replace_document(
            document_id,
            [
                SemanticRecord(document_id, chunk_id, 0, vector(1.0, 0.0)),
                SemanticRecord(document_id, chunk_id, 1, vector(0.0, 1.0)),
            ],
        )

    with pytest.raises(ValueError, match="share"):
        await index.replace_document(
            document_id,
            [
                SemanticRecord(document_id, uuid4(), 0, vector(1.0, 0.0)),
                SemanticRecord(document_id, uuid4(), 1, vector(1.0, 0.0, 0.0)),
            ],
        )

    with pytest.raises(ValueError, match="limit"):
        await index.search(vector(1.0, 0.0), limit=0)

    with pytest.raises(ValueError, match="minimum_score"):
        await index.search(vector(1.0, 0.0), minimum_score=1.1)
