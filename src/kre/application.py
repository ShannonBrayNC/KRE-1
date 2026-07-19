from __future__ import annotations

from typing import Any

from kre.schemas import SearchChannelScore, SearchHit, SearchMode, SearchRequest, SearchResponse
from kre.search import HybridSearchService, KeywordSearch, SemanticRetrievalService
from kre.storage import KnowledgeRepository


class SearchApplicationBackend:
    """Execute search modes and project canonical, authorization-ready responses."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        keyword: KeywordSearch,
        semantic: SemanticRetrievalService,
        hybrid: HybridSearchService,
    ) -> None:
        self._repository = repository
        self._keyword = keyword
        self._semantic = semantic
        self._hybrid = hybrid

    async def execute(self, request: SearchRequest) -> SearchResponse:
        if request.mode is SearchMode.KEYWORD:
            results = await self._keyword.search(request.query, limit=request.limit)
            hits = []
            for rank, result in enumerate(results, start=1):
                context = await self._document_context(result.document_id)
                hits.append(
                    SearchHit(
                        document_id=result.document_id,
                        chunk_id=result.chunk_id,
                        document_title=result.document_title,
                        sequence=result.sequence,
                        text=result.text,
                        score=result.score,
                        keyword=SearchChannelScore(rank=rank, score=result.score),
                        metadata={"matched_terms": list(result.matched_terms)},
                        **context,
                    )
                )
        elif request.mode is SearchMode.SEMANTIC:
            results = await self._semantic.search(
                request.query,
                limit=request.limit,
                minimum_score=request.minimum_semantic_score,
            )
            hits = [
                SearchHit(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    document_title=result.document_title,
                    sequence=result.sequence,
                    text=result.text,
                    score=result.score,
                    semantic=SearchChannelScore(rank=rank, score=result.score),
                    source_uri=result.source_uri,
                    source_system=result.source_system,
                    source_version=result.source_version,
                    classification=result.classification,
                    security_labels=result.security_labels,
                    metadata=result.metadata,
                )
                for rank, result in enumerate(results, start=1)
            ]
        else:
            results = await self._hybrid.search(
                request.query,
                limit=request.limit,
                candidate_limit=request.candidate_limit,
                minimum_semantic_score=request.minimum_semantic_score,
            )
            hits = []
            for result in results:
                context = await self._document_context(result.document_id)
                hits.append(
                    SearchHit(
                        document_id=result.document_id,
                        chunk_id=result.chunk_id,
                        document_title=result.document_title,
                        sequence=result.sequence,
                        text=result.text,
                        score=result.score,
                        keyword=(
                            SearchChannelScore(
                                rank=result.keyword_rank,
                                score=result.keyword_score,
                            )
                            if result.keyword_rank is not None
                            and result.keyword_score is not None
                            else None
                        ),
                        semantic=(
                            SearchChannelScore(
                                rank=result.semantic_rank,
                                score=result.semantic_score,
                            )
                            if result.semantic_rank is not None
                            and result.semantic_score is not None
                            else None
                        ),
                        **context,
                    )
                )

        return SearchResponse(
            query=request.query,
            mode=request.mode,
            count=len(hits),
            results=tuple(hits),
        )

    async def _document_context(self, document_id: Any) -> dict[str, Any]:
        document = await self._repository.get_document(document_id)
        if document is None:
            return {
                "source_uri": None,
                "source_system": None,
                "source_version": None,
                "classification": None,
                "security_labels": (),
            }
        return {
            "source_uri": (
                str(document.provenance.source_uri)
                if document.provenance.source_uri is not None
                else None
            ),
            "source_system": document.provenance.source_system,
            "source_version": document.provenance.source_version,
            "classification": document.classification.value,
            "security_labels": tuple(document.security_labels),
        }
