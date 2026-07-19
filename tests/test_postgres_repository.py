from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest

from kre.models import Classification, KnowledgeChunk, KnowledgeDocument, Provenance
from kre.storage import PostgresKnowledgeRepository


class FakeConnection:
    def __init__(self) -> None:
        self.rows: dict[Any, dict[str, Any]] = {}
        self.chunk_rows: dict[Any, list[dict[str, Any]]] = {}
        self.in_transaction = False

    @asynccontextmanager
    async def transaction(self):
        self.in_transaction = True
        try:
            yield
        finally:
            self.in_transaction = False

    async def fetchrow(self, query: str, *args: Any):
        if "INSERT INTO" in query and "knowledge_documents" in query:
            row = _document_row(*args)
            self.rows[args[0]] = row
            return row
        if "SELECT id FROM" in query:
            return {"id": args[0]} if args[0] in self.rows else None
        if "knowledge_documents WHERE id" in query:
            return self.rows.get(args[0])
        return None

    async def fetch(self, query: str, *args: Any):
        if "knowledge_documents" in query:
            limit, offset = args
            rows = [self.rows[key] for key in sorted(self.rows, key=str)]
            return rows[offset : offset + limit]
        return list(self.chunk_rows.get(args[0], []))

    async def execute(self, query: str, *args: Any) -> str:
        if "DELETE FROM" in query and "knowledge_documents" in query:
            existed = self.rows.pop(args[0], None) is not None
            self.chunk_rows.pop(args[0], None)
            return f"DELETE {1 if existed else 0}"
        if "DELETE FROM" in query and "knowledge_chunks" in query:
            self.chunk_rows[args[0]] = []
            return "DELETE 1"
        if "INSERT INTO" in query and "knowledge_chunks" in query:
            assert self.in_transaction
            row = {
                "id": args[0], "document_id": args[1], "sequence": args[2],
                "text": args[3], "token_count": args[4], "section": args[5],
                "page": args[6], "metadata": args[7],
            }
            self.chunk_rows.setdefault(args[1], []).append(row)
            return "INSERT 0 1"
        raise AssertionError(query)


class FakePool:
    def __init__(self) -> None:
        self.connection = FakeConnection()

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


def _document_row(*args: Any) -> dict[str, Any]:
    return {
        "id": args[0], "title": args[1], "content": args[2],
        "mime_type": args[3], "language": args[4], "owner": args[5],
        "classification": args[6], "security_labels": args[7], "tags": args[8],
        "created_at": args[9], "modified_at": args[10], "provenance": args[11],
        "ingested_at": args[12], "content_hash": args[13],
        "source_system": args[14], "source_uri": args[15], "source_version": args[16],
    }


def make_document() -> KnowledgeDocument:
    return KnowledgeDocument(
        title="Durable knowledge",
        content="canonical",
        classification=Classification.CONFIDENTIAL,
        security_labels=["ops"],
        provenance=Provenance(
            source_system="test",
            connector="test",
            content_hash="sha256:test",
            ingested_at=datetime.now(timezone.utc),
        ),
    )


@pytest.mark.asyncio
async def test_postgres_repository_document_lifecycle() -> None:
    repository = PostgresKnowledgeRepository(FakePool())
    document = make_document()

    stored = await repository.upsert_document(document)
    assert stored == document
    assert await repository.get_document(document.id) == document
    assert await repository.list_documents() == [document]
    assert await repository.delete_document(document.id) is True
    assert await repository.delete_document(document.id) is False


@pytest.mark.asyncio
async def test_postgres_repository_replaces_chunks_transactionally() -> None:
    pool = FakePool()
    repository = PostgresKnowledgeRepository(pool)
    document = make_document()
    await repository.upsert_document(document)
    chunks = [
        KnowledgeChunk(document_id=document.id, sequence=1, text="second"),
        KnowledgeChunk(document_id=document.id, sequence=0, text="first"),
    ]

    replaced = await repository.replace_chunks(document.id, chunks)

    assert [chunk.sequence for chunk in replaced] == [0, 1]
    assert [chunk.text for chunk in await repository.list_chunks(document.id)] == ["first", "second"]
    assert pool.connection.in_transaction is False


@pytest.mark.asyncio
async def test_postgres_repository_validates_chunk_replacement() -> None:
    repository = PostgresKnowledgeRepository(FakePool())
    document_id = uuid4()
    with pytest.raises(KeyError):
        await repository.replace_chunks(document_id, [])
    with pytest.raises(ValueError, match="belong"):
        await repository.replace_chunks(
            document_id,
            [KnowledgeChunk(document_id=uuid4(), sequence=0, text="wrong")],
        )


def test_postgres_repository_rejects_unsafe_schema() -> None:
    with pytest.raises(ValueError, match="identifier"):
        PostgresKnowledgeRepository(FakePool(), schema="kre;drop")
