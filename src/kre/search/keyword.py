from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from kre.storage.base import KnowledgeRepository

_TOKEN = re.compile(r"[\w'-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked keyword-search result with citation-ready identifiers."""

    document_id: UUID
    chunk_id: UUID
    document_title: str
    sequence: int
    text: str
    score: float
    matched_terms: tuple[str, ...]


class KeywordSearch:
    """Deterministic in-memory keyword retrieval over persisted KRE chunks."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self._repository = repository

    async def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        terms = self._tokenize(query)
        if not terms:
            raise ValueError("query must contain at least one searchable term")

        phrase = " ".join(terms)
        results: list[SearchResult] = []
        documents = await self._repository.list_documents(limit=10_000)

        for document in documents:
            for chunk in await self._repository.list_chunks(document.id):
                tokens = self._tokenize(chunk.text)
                if not tokens:
                    continue

                frequencies = {term: tokens.count(term) for term in terms}
                matched = tuple(term for term in terms if frequencies[term] > 0)
                if not matched:
                    continue

                coverage = len(matched) / len(terms)
                frequency = sum(frequencies.values()) / len(tokens)
                phrase_bonus = 1.0 if phrase in " ".join(tokens) else 0.0
                title_bonus = 0.25 if any(term in document.title.casefold() for term in terms) else 0.0
                score = round((coverage * 2.0) + frequency + phrase_bonus + title_bonus, 8)

                results.append(
                    SearchResult(
                        document_id=document.id,
                        chunk_id=chunk.id,
                        document_title=document.title,
                        sequence=chunk.sequence,
                        text=chunk.text,
                        score=score,
                        matched_terms=matched,
                    )
                )

        results.sort(
            key=lambda item: (
                -item.score,
                item.document_title.casefold(),
                str(item.document_id),
                item.sequence,
                str(item.chunk_id),
            )
        )
        return results[:limit]

    @staticmethod
    def _tokenize(value: str) -> tuple[str, ...]:
        return tuple(match.group(0).casefold() for match in _TOKEN.finditer(value))
