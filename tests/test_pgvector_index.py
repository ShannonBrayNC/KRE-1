from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import pytest

from kre.embeddings import EmbeddingVector
from kre.search import PgVectorSemanticIndex, SemanticRecord


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.rows: list[dict[str, Any]] = []
        self.in_transaction = False

    @asynccontextmanager
    async def transaction(self):
        self.in_transaction = True
        try:
            yield
        finally:
            self.in_transaction = False

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        if "DELETE FROM" in query:
            return "DELETE 2"
        if "INSERT INTO" in query:
            assert self.in_transaction
            return "INSERT 0 1"
        raise AssertionError(query)

    async def fetch(self, query: str, *args: Any):
        self.executed.append((query, args))
        return list(self.rows)


class FakePool:
    def __init__(self) -> None:
        self.connection = FakeConnection()

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


def vector(*values: float, model: str = "embed-v1") -> EmbeddingVector:
    return EmbeddingVector(values=tuple(values), model=model, dimensions=len(values))


@pytest.mark.asyncio
async def test_pgvector_replaces_document_transactionally() -> None:
    pool = FakePool()
    index = PgVectorSemanticIndex(pool, vector_dimensions=2)
    document_id = uuid4()
    records = [
        SemanticRecord(document_id=document_id, chunk_id=uuid4(), sequence=0, vector=vector(1, 0)),
        SemanticRecord(document_id=document_id, chunk_id=uuid4(), sequence=1, vector=vector(0, 1)),
    ]

    await index.replace_document(document_id, records)

    assert pool.connection.in_transaction is False
    assert len(pool.connection.executed) == 3
    assert "USING kre.knowledge_chunks" in pool.connection.executed[0][0]
    assert pool.connection.executed[1][1][-1] == "[1,0]"
    assert pool.connection.executed[2][1][-1] == "[0,1]"


@pytest.mark.asyncio
async def test_pgvector_search_projects_deterministic_results() -> None:
    pool = FakePool()
    document_id = uuid4()
    chunk_id = uuid4()
    pool.connection.rows = [
        {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "sequence": 3,
            "score": 0.8123456789,
        }
    ]
    index = PgVectorSemanticIndex(pool, schema="lantern_kre", vector_dimensions=2)

    results = await index.search(vector(0.5, 0.25), limit=4, minimum_score=0.2)

    assert len(results) == 1
    assert results[0].document_id == document_id
    assert results[0].chunk_id == chunk_id
    assert results[0].sequence == 3
    assert results[0].score == 0.81234568
    query, args = pool.connection.executed[-1]
    assert "lantern_kre.semantic_embeddings" in query
    assert "ORDER BY score DESC" in query
    assert args == ("[0.5,0.25]", "embed-v1", 2, 0.2, 4)


@pytest.mark.asyncio
async def test_pgvector_delete_reports_affected_records() -> None:
    index = PgVectorSemanticIndex(FakePool(), vector_dimensions=2)

    assert await index.delete_document(uuid4()) is True


@pytest.mark.asyncio
async def test_pgvector_validates_dimensions_and_record_identity() -> None:
    index = PgVectorSemanticIndex(FakePool(), vector_dimensions=2)
    document_id = uuid4()

    with pytest.raises(ValueError, match="dimensions"):
        await index.search(vector(1.0))
    with pytest.raises(ValueError, match="target document"):
        await index.replace_document(
            document_id,
            [
                SemanticRecord(
                    document_id=uuid4(),
                    chunk_id=uuid4(),
                    sequence=0,
                    vector=vector(1.0, 0.0),
                )
            ],
        )
    duplicate_chunk = uuid4()
    with pytest.raises(ValueError, match="unique"):
        await index.replace_document(
            document_id,
            [
                SemanticRecord(document_id, duplicate_chunk, 0, vector(1.0, 0.0)),
                SemanticRecord(document_id, duplicate_chunk, 1, vector(0.0, 1.0)),
            ],
        )
