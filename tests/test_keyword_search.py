from __future__ import annotations

import pytest

from kre.models import KnowledgeChunk, KnowledgeDocument, Provenance
from kre.search import KeywordSearch
from kre.storage import InMemoryKnowledgeRepository


def make_document(title: str, content: str) -> KnowledgeDocument:
    return KnowledgeDocument(
        title=title,
        content=content,
        provenance=Provenance(
            source_system="test",
            connector="test",
            content_hash=f"sha256:{title}",
        ),
    )


@pytest.mark.asyncio
async def test_keyword_search_ranks_phrase_coverage_and_frequency() -> None:
    repository = InMemoryKnowledgeRepository()
    architecture = make_document("KRE Architecture", "")
    operations = make_document("Operations", "")
    await repository.upsert_document(architecture)
    await repository.upsert_document(operations)
    await repository.replace_chunks(
        architecture.id,
        [
            KnowledgeChunk(
                document_id=architecture.id,
                sequence=0,
                text="Governed knowledge retrieval uses a provenance-first knowledge service.",
            ),
            KnowledgeChunk(
                document_id=architecture.id,
                sequence=1,
                text="Connectors normalize enterprise sources.",
            ),
        ],
    )
    await repository.replace_chunks(
        operations.id,
        [
            KnowledgeChunk(
                document_id=operations.id,
                sequence=0,
                text="Knowledge operations and governed workflows.",
            )
        ],
    )

    results = await KeywordSearch(repository).search("governed knowledge", limit=3)

    assert len(results) == 2
    assert results[0].document_id == architecture.id
    assert results[0].sequence == 0
    assert results[0].matched_terms == ("governed", "knowledge")
    assert results[0].score > results[1].score


@pytest.mark.asyncio
async def test_keyword_search_is_case_insensitive_and_citation_ready() -> None:
    repository = InMemoryKnowledgeRepository()
    document = make_document("SignalForge", "")
    await repository.upsert_document(document)
    chunk = KnowledgeChunk(
        document_id=document.id,
        sequence=4,
        text="SignalForge governs provenance across repositories.",
    )
    await repository.replace_chunks(document.id, [chunk])

    [result] = await KeywordSearch(repository).search("PROVENANCE")

    assert result.document_id == document.id
    assert result.chunk_id == chunk.id
    assert result.document_title == "SignalForge"
    assert result.sequence == 4
    assert result.text == chunk.text


@pytest.mark.asyncio
async def test_keyword_search_returns_empty_for_no_matches() -> None:
    repository = InMemoryKnowledgeRepository()
    document = make_document("Unrelated", "")
    await repository.upsert_document(document)
    await repository.replace_chunks(
        document.id,
        [KnowledgeChunk(document_id=document.id, sequence=0, text="Alpha beta")],
    )

    assert await KeywordSearch(repository).search("gamma") == []


@pytest.mark.asyncio
async def test_keyword_search_validates_query_and_limit() -> None:
    search = KeywordSearch(InMemoryKnowledgeRepository())

    with pytest.raises(ValueError, match="searchable"):
        await search.search("---")
    with pytest.raises(ValueError, match="limit"):
        await search.search("knowledge", limit=0)
    with pytest.raises(TypeError, match="string"):
        await search.search(42)  # type: ignore[arg-type]
