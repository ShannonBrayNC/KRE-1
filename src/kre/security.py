from __future__ import annotations

from dataclasses import dataclass

from kre.models import Classification
from kre.schemas import SearchHit, SearchResponse

_CLASSIFICATION_RANK = {
    Classification.PUBLIC.value: 0,
    Classification.INTERNAL.value: 1,
    Classification.CONFIDENTIAL.value: 2,
    Classification.RESTRICTED.value: 3,
}


@dataclass(frozen=True, slots=True)
class SearchAuthorizationContext:
    """Caller clearance and security labels used to trim retrieval results."""

    clearance: str = Classification.PUBLIC.value
    security_labels: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.clearance not in _CLASSIFICATION_RANK:
            raise ValueError("clearance must be a known classification")


class SecurityTrimmer:
    """Fail-closed search-result authorization policy."""

    def is_allowed(self, hit: SearchHit, context: SearchAuthorizationContext) -> bool:
        if hit.classification is None:
            return False

        rank = _CLASSIFICATION_RANK.get(hit.classification)
        if rank is None or rank > _CLASSIFICATION_RANK[context.clearance]:
            return False

        required_labels = frozenset(hit.security_labels)
        return required_labels.issubset(context.security_labels)

    def trim(
        self,
        response: SearchResponse,
        context: SearchAuthorizationContext,
    ) -> SearchResponse:
        results = tuple(hit for hit in response.results if self.is_allowed(hit, context))
        return SearchResponse(
            query=response.query,
            mode=response.mode,
            count=len(results),
            results=results,
        )
