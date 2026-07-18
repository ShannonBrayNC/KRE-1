from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kre.schemas import SearchHit, SearchRequest, SearchResponse
from kre.search.api import router


class StubSearchBackend:
    async def execute(self, request: SearchRequest) -> SearchResponse:
        public = SearchHit(
            document_id=uuid4(),
            chunk_id=uuid4(),
            document_title="Public",
            sequence=0,
            text="Public knowledge",
            score=1.0,
            classification="public",
        )
        restricted = SearchHit(
            document_id=uuid4(),
            chunk_id=uuid4(),
            document_title="Restricted",
            sequence=0,
            text="Restricted knowledge",
            score=0.9,
            classification="restricted",
            security_labels=("ops",),
        )
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            count=2,
            results=(public, restricted),
        )


def app_with_backend() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.search_backend = StubSearchBackend()
    return app


def test_search_route_trims_results_from_headers() -> None:
    client = TestClient(app_with_backend())

    public_response = client.post("/search", json={"query": "knowledge"})
    assert public_response.status_code == 200
    assert public_response.json()["count"] == 1

    privileged_response = client.post(
        "/search",
        json={"query": "knowledge"},
        headers={
            "X-KRE-Clearance": "restricted",
            "X-KRE-Security-Labels": "ops",
        },
    )
    assert privileged_response.status_code == 200
    assert privileged_response.json()["count"] == 2


def test_search_route_rejects_unknown_clearance_and_missing_backend() -> None:
    client = TestClient(app_with_backend())
    invalid = client.post(
        "/search",
        json={"query": "knowledge"},
        headers={"X-KRE-Clearance": "top-secret"},
    )
    assert invalid.status_code == 400

    app = FastAPI()
    app.include_router(router)
    unavailable = TestClient(app).post("/search", json={"query": "knowledge"})
    assert unavailable.status_code == 503
