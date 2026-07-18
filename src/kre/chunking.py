from __future__ import annotations

from collections.abc import Iterable

from kre.models import KnowledgeChunk, KnowledgeDocument


def chunk_document(
    document: KnowledgeDocument,
    *,
    max_characters: int = 1200,
    overlap_characters: int = 120,
) -> list[KnowledgeChunk]:
    """Split normalized content into deterministic, overlapping character chunks."""
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    if overlap_characters < 0 or overlap_characters >= max_characters:
        raise ValueError("overlap_characters must be between zero and max_characters - 1")

    paragraphs = [part.strip() for part in document.content.split("\n\n") if part.strip()]
    segments = list(_pack_paragraphs(paragraphs, max_characters=max_characters))
    if not segments and document.content.strip():
        segments = [document.content.strip()]

    chunks: list[KnowledgeChunk] = []
    previous_tail = ""
    for sequence, segment in enumerate(segments):
        text = f"{previous_tail}\n\n{segment}".strip() if previous_tail else segment
        chunks.append(
            KnowledgeChunk(
                document_id=document.id,
                sequence=sequence,
                text=text,
                metadata={"content_hash": document.provenance.content_hash},
            )
        )
        previous_tail = text[-overlap_characters:] if overlap_characters else ""
    return chunks


def _pack_paragraphs(paragraphs: Iterable[str], *, max_characters: int) -> Iterable[str]:
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_characters:
            if buffer:
                yield buffer
                buffer = ""
            for start in range(0, len(paragraph), max_characters):
                yield paragraph[start : start + max_characters]
            continue

        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= max_characters:
            buffer = candidate
        else:
            yield buffer
            buffer = paragraph

    if buffer:
        yield buffer
