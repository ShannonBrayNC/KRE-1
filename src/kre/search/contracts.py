from __future__ import annotations

from collections.abc import Sequence

from kre.schemas import SearchChannelScore, SearchHit, SearchMode, SearchResponse
from kre.search.hybrid import HybridSearchResult


def hybrid_response(query: str, results: Sequence[HybridSearchResult]) -> SearchResponse:
    """Project hybrid-domain results into the stable API response envelope."""

    hits = tuple(
        SearchHit(
            document_id=result.document_id,
            chunk_id=result.chunk_id,
            document_title=result.document_title,
            sequence=result.sequence,
            text=result.text,
            score=result.score,
            keyword=(
                SearchChannelScore(rank=result.keyword_rank, score=result.keyword_score)
                if result.keyword_rank is not None and result.keyword_score is not None
                else None
            ),
            semantic=(
                SearchChannelScore(rank=result.semantic_rank, score=result.semantic_score)
                if result.semantic_rank is not None and result.semantic_score is not None
                else None
            ),
            source_uri=result.source_uri,
            source_system=result.source_system,
            source_version=result.source_version,
            classification=result.classification,
            security_labels=result.security_labels,
        )
        for result in results
    )
    return SearchResponse(
        query=query.strip(),
        mode=SearchMode.HYBRID,
        count=len(hits),
        results=hits,
    )
