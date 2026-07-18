from __future__ import annotations

from pathlib import Path

import pytest

from kre.chunking import chunk_document
from kre.connectors.markdown import MarkdownConnector
from kre.normalization import normalize_text


def test_normalization_is_deterministic() -> None:
    first = normalize_text("Title\r\n\r\nBody   text  \r\n")
    second = normalize_text("Title\n\nBody text")

    assert first == second
    assert first.content_hash.startswith("sha256:")


@pytest.mark.asyncio
async def test_markdown_connector_discovers_and_fetches(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "architecture.md").write_text(
        "# Architecture\n\nLantern   knowledge.\n",
        encoding="utf-8",
    )
    connector = MarkdownConnector(tmp_path)

    discovered = [source_id async for source_id in connector.discover()]
    document = await connector.fetch(discovered[0])

    assert discovered == ["docs/architecture.md"]
    assert document.title == "architecture"
    assert document.content == "# Architecture\n\nLantern knowledge."
    assert document.provenance.connector == "markdown"


def test_chunking_preserves_document_traceability(tmp_path: Path) -> None:
    normalized = normalize_text("Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph.")
    from kre.models import KnowledgeDocument, Provenance

    document = KnowledgeDocument(
        title="Example",
        content=normalized.text,
        provenance=Provenance(
            source_system="test",
            connector="test",
            content_hash=normalized.content_hash,
        ),
    )

    chunks = chunk_document(document, max_characters=25, overlap_characters=5)

    assert len(chunks) >= 2
    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.document_id == document.id for chunk in chunks)
    assert all(chunk.metadata["content_hash"] == normalized.content_hash for chunk in chunks)
