from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from kre.search.keyword import KeywordSearch, SearchResult
from kre.search.retrieval import SemanticRetrievalResult, SemanticRetrievalService


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    """One fused lexical and semantic result with citation-ready content."""

    document_id: UUID
    chunk_id: UUID
    document_title: str
    sequence: int
    text: str
    score: float
    keyword_rank: int | None
    semantic_rank: int | None
    keyword_score: float | None
    semantic_score: float | None
    source_uri: str | None
    source_system: str | None
    source_version: str | None
    classification: str | None
    security_labels: tuple[str, ...]


@dataclass(slots=True)
class _HybridCandidate:
    document_id: UUID
    chunk_id: UUID
    document_title: str
    sequence: int
    text: str
    score: float = 0.0
    keyword_rank: int | None = None
    semantic_rank: int | None = None
    keyword_score: float | None = None
    semantic_score: float | None = None
    source_uri: str | None = None
    source_system: str | None = None
    source_version: str | None = None
    classification: str | None = None
    security_labels: tuple[str, ...] = ()

    def result(self) -> HybridSearchResult:
        return HybridSearchResult(
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            document_title=self.document_title,
            sequence=self.sequence,
            text=self.text,
            score=self.score,
            keyword_rank=self.keyword_rank,
            semantic_rank=self.semantic_rank,
            keyword_score=self.keyword_score,
            semantic_score=self.semantic_score,
            source_uri=self.source_uri,
            source_system=self.source_system,
            source_version=self.source_version,
            classification=self.classification,
            security_labels=self.security_labels,
        )


class HybridSearchService:
    """Fuse keyword and semantic retrieval with reciprocal-rank fusion."""

    def __init__(
        self,
        keyword: KeywordSearch,
        semantic: SemanticRetrievalService,
        *,
        rank_constant: int = 60,
    ) -> None:
        if rank_constant < 1:
            raise ValueError("rank_constant must be at least 1")
        self._keyword = keyword
        self._semantic = semantic
        self._rank_constant = rank_constant

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        candidate_limit: int | None = None,
        minimum_semantic_score: float = -1.0,
    ) -> list[HybridSearchResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        candidates = candidate_limit if candidate_limit is not None else max(limit * 3, 20)
        if candidates < limit:
            raise ValueError("candidate_limit must be at least limit")

        keyword_results = await self._keyword.search(query, limit=candidates)
        semantic_results = await self._semantic.search(
            query,
            limit=candidates,
            minimum_score=minimum_semantic_score,
        )
        return self._fuse(keyword_results, semantic_results)[:limit]

    def _fuse(
        self,
        keyword_results: list[SearchResult],
        semantic_results: list[SemanticRetrievalResult],
    ) -> list[HybridSearchResult]:
        fused: dict[tuple[UUID, UUID], _HybridCandidate] = {}

        for rank, item in enumerate(keyword_results, start=1):
            key = (item.document_id, item.chunk_id)
            fused[key] = _HybridCandidate(
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                document_title=item.document_title,
                sequence=item.sequence,
                text=item.text,
                score=1.0 / (self._rank_constant + rank),
                keyword_rank=rank,
                keyword_score=item.score,
            )

        for rank, item in enumerate(semantic_results, start=1):
            key = (item.document_id, item.chunk_id)
            candidate = fused.get(key)
            if candidate is None:
                candidate = _HybridCandidate(
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    document_title=item.document_title,
                    sequence=item.sequence,
                    text=item.text,
                )
                fused[key] = candidate

            candidate.score += 1.0 / (self._rank_constant + rank)
            candidate.semantic_rank = rank
            candidate.semantic_score = item.score
            candidate.source_uri = item.source_uri
            candidate.source_system = item.source_system
            candidate.source_version = item.source_version
            candidate.classification = item.classification
            candidate.security_labels = item.security_labels

        results = [candidate.result() for candidate in fused.values()]
        results.sort(
            key=lambda item: (
                -item.score,
                item.document_title.casefold(),
                str(item.document_id),
                item.sequence,
                str(item.chunk_id),
            )
        )
        return results
