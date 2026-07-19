from __future__ import annotations

from uuid import uuid4

import pytest

from kre.composition import build_components
from kre.config import KRESettings
from kre.schemas import SearchHit, SearchMode, SearchRequest, SearchResponse
from kre.telemetry import InMemoryRetrievalTelemetry, TelemetrySearchBackend


class SuccessfulBackend:
    async def execute(self, request: SearchRequest) -> SearchResponse:
        hit = SearchHit(
            document_id=uuid4(),
            chunk_id=uuid4(),
            document_title="Telemetry result",
            sequence=0,
            text="retrieval result",
            score=1.0,
            classification="public",
        )
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            count=1,
            results=(hit,),
        )


class FailingBackend:
    async def execute(self, request: SearchRequest) -> SearchResponse:
        raise RuntimeError("provider unavailable")


class Clock:
    def __init__(self, *values: int) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


@pytest.mark.asyncio
async def test_telemetry_records_success_without_sensitive_payloads() -> None:
    sink = InMemoryRetrievalTelemetry()
    backend = TelemetrySearchBackend(
        SuccessfulBackend(),
        sink,
        clock_ns=Clock(1_000_000, 3_500_000),
    )
    request = SearchRequest(
        query="confidential customer roadmap",
        mode=SearchMode.HYBRID,
        limit=4,
        candidate_limit=12,
    )

    response = await backend.execute(request)

    assert response.count == 1
    events = sink.snapshot()
    assert len(events) == 1
    event = events[0]
    assert event.mode is SearchMode.HYBRID
    assert event.requested_limit == 4
    assert event.candidate_limit == 12
    assert event.returned_count == 1
    assert event.duration_ms == 2.5
    assert event.outcome == "success"
    assert event.error_type is None
    assert "confidential" not in repr(event)


@pytest.mark.asyncio
async def test_telemetry_records_failure_and_reraises() -> None:
    sink = InMemoryRetrievalTelemetry()
    backend = TelemetrySearchBackend(
        FailingBackend(),
        sink,
        clock_ns=Clock(10_000_000, 11_250_000),
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await backend.execute(SearchRequest(query="health", mode=SearchMode.SEMANTIC))

    event = sink.snapshot()[0]
    assert event.outcome == "failure"
    assert event.error_type == "RuntimeError"
    assert event.returned_count == 0
    assert event.duration_ms == 1.25


@pytest.mark.asyncio
async def test_composition_root_exposes_process_telemetry() -> None:
    components = build_components(KRESettings())

    response = await components.search_backend.execute(
        SearchRequest(query="no indexed records", mode=SearchMode.KEYWORD)
    )

    assert response.count == 0
    event = components.telemetry.snapshot()[0]
    assert event.mode is SearchMode.KEYWORD
    assert event.outcome == "success"
