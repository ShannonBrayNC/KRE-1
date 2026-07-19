from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import monotonic_ns
from typing import Protocol

from kre.schemas import SearchMode, SearchRequest, SearchResponse


class SearchBackend(Protocol):
    async def execute(self, request: SearchRequest) -> SearchResponse: ...


@dataclass(frozen=True, slots=True)
class RetrievalTelemetryEvent:
    """Privacy-preserving operational facts for one retrieval execution."""

    mode: SearchMode
    requested_limit: int
    candidate_limit: int | None
    returned_count: int
    duration_ms: float
    outcome: str
    error_type: str | None = None

    def __post_init__(self) -> None:
        if self.requested_limit < 1:
            raise ValueError("requested_limit must be at least 1")
        if self.candidate_limit is not None and self.candidate_limit < self.requested_limit:
            raise ValueError("candidate_limit must be at least requested_limit")
        if self.returned_count < 0:
            raise ValueError("returned_count cannot be negative")
        if self.duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")
        if self.outcome not in {"success", "failure"}:
            raise ValueError("outcome must be success or failure")
        if self.outcome == "success" and self.error_type is not None:
            raise ValueError("successful events cannot include error_type")
        if self.outcome == "failure" and not self.error_type:
            raise ValueError("failed events must include error_type")


class RetrievalTelemetrySink(Protocol):
    async def record(self, event: RetrievalTelemetryEvent) -> None: ...


class InMemoryRetrievalTelemetry:
    """Bounded process-local telemetry sink for tests and local operations."""

    def __init__(self, *, max_events: int = 1_000) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self._max_events = max_events
        self._events: list[RetrievalTelemetryEvent] = []

    async def record(self, event: RetrievalTelemetryEvent) -> None:
        self._events.append(event)
        overflow = len(self._events) - self._max_events
        if overflow > 0:
            del self._events[:overflow]

    def snapshot(self) -> Sequence[RetrievalTelemetryEvent]:
        return tuple(self._events)


class TelemetrySearchBackend:
    """Decorate search while ensuring telemetry cannot disrupt retrieval."""

    def __init__(
        self,
        backend: SearchBackend,
        sink: RetrievalTelemetrySink,
        *,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        self._backend = backend
        self._sink = sink
        self._clock_ns = clock_ns

    async def execute(self, request: SearchRequest) -> SearchResponse:
        started = self._clock_ns()
        try:
            response = await self._backend.execute(request)
        except Exception as exc:
            await self._record_best_effort(
                request,
                started=started,
                returned_count=0,
                outcome="failure",
                error_type=type(exc).__name__,
            )
            raise

        await self._record_best_effort(
            request,
            started=started,
            returned_count=response.count,
            outcome="success",
            error_type=None,
        )
        return response

    async def _record_best_effort(
        self,
        request: SearchRequest,
        *,
        started: int,
        returned_count: int,
        outcome: str,
        error_type: str | None,
    ) -> None:
        try:
            elapsed_ns = max(0, self._clock_ns() - started)
            event = RetrievalTelemetryEvent(
                mode=request.mode,
                requested_limit=request.limit,
                candidate_limit=request.candidate_limit,
                returned_count=returned_count,
                duration_ms=round(elapsed_ns / 1_000_000, 3),
                outcome=outcome,
                error_type=error_type,
            )
            await self._sink.record(event)
        except Exception:
            # Observability must never alter search success or mask the original failure.
            return
