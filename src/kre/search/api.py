from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter, Header, HTTPException, Request, status

from kre.schemas import SearchRequest, SearchResponse
from kre.security import SearchAuthorizationContext, SecurityTrimmer

router = APIRouter(prefix="/search", tags=["search"])


class SearchBackend(Protocol):
    """Application search execution boundary used by the HTTP route."""

    async def execute(self, request: SearchRequest) -> SearchResponse:
        """Execute one validated search request."""


def _labels(value: str | None) -> frozenset[str]:
    if value is None:
        return frozenset()
    return frozenset(label.strip() for label in value.split(",") if label.strip())


@router.post("", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    request: Request,
    clearance: str = Header(default="public", alias="X-KRE-Clearance"),
    security_labels: str | None = Header(default=None, alias="X-KRE-Security-Labels"),
) -> SearchResponse:
    backend: SearchBackend | None = getattr(request.app.state, "search_backend", None)
    if backend is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="search backend is not configured",
        )

    try:
        context = SearchAuthorizationContext(
            clearance=clearance.casefold(),
            security_labels=_labels(security_labels),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    response = await backend.execute(payload)
    return SecurityTrimmer().trim(response, context)
