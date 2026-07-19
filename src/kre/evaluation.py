from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from kre.schemas import SearchMode, SearchRequest, SearchResponse


class SearchEvaluatorBackend(Protocol):
    async def execute(self, request: SearchRequest) -> SearchResponse: ...


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    """One governed retrieval expectation."""

    name: str
    query: str
    mode: SearchMode
    expected_document_ids: tuple[UUID, ...]
    forbidden_document_ids: tuple[UUID, ...] = ()
    limit: int = 10

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("golden query name must not be empty")
        if not self.query.strip():
            raise ValueError("golden query text must not be empty")
        if not self.expected_document_ids:
            raise ValueError("golden query must define at least one expected document")
        if len(self.expected_document_ids) != len(set(self.expected_document_ids)):
            raise ValueError("expected document identifiers must be unique")
        if len(self.forbidden_document_ids) != len(set(self.forbidden_document_ids)):
            raise ValueError("forbidden document identifiers must be unique")
        if self.limit < 1 or self.limit > 100:
            raise ValueError("golden query limit must be between 1 and 100")
        overlap = set(self.expected_document_ids) & set(self.forbidden_document_ids)
        if overlap:
            raise ValueError("documents cannot be both expected and forbidden")


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    name: str
    recall_at_k: float
    reciprocal_rank: float
    forbidden_hits: tuple[UUID, ...]
    returned_document_ids: tuple[UUID, ...]

    @property
    def passed(self) -> bool:
        return self.recall_at_k == 1.0 and not self.forbidden_hits


@dataclass(frozen=True, slots=True)
class CorpusEvaluation:
    queries: tuple[QueryEvaluation, ...]

    def __post_init__(self) -> None:
        if not self.queries:
            raise ValueError("corpus evaluation must contain at least one query")

    @property
    def mean_recall_at_k(self) -> float:
        return round(sum(item.recall_at_k for item in self.queries) / len(self.queries), 8)

    @property
    def mean_reciprocal_rank(self) -> float:
        return round(sum(item.reciprocal_rank for item in self.queries) / len(self.queries), 8)

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.queries)


async def evaluate_corpus(
    backend: SearchEvaluatorBackend,
    queries: tuple[GoldenQuery, ...],
) -> CorpusEvaluation:
    """Execute a golden corpus and calculate deterministic retrieval metrics."""

    if not queries:
        raise ValueError("golden corpus must contain at least one query")

    evaluations: list[QueryEvaluation] = []
    for golden in queries:
        response = await backend.execute(
            SearchRequest(query=golden.query, mode=golden.mode, limit=golden.limit)
        )
        if response.query != golden.query or response.mode is not golden.mode:
            raise ValueError(f"backend response contract mismatch for golden query: {golden.name}")

        returned = tuple(hit.document_id for hit in response.results[: golden.limit])
        returned_set = set(returned)
        expected_set = set(golden.expected_document_ids)
        forbidden_set = set(golden.forbidden_document_ids)
        recall = len(returned_set & expected_set) / len(expected_set)
        first_rank = next(
            (rank for rank, document_id in enumerate(returned, start=1) if document_id in expected_set),
            None,
        )
        evaluations.append(
            QueryEvaluation(
                name=golden.name,
                recall_at_k=round(recall, 8),
                reciprocal_rank=round(1.0 / first_rank, 8) if first_rank else 0.0,
                forbidden_hits=tuple(dict.fromkeys(
                    document_id for document_id in returned if document_id in forbidden_set
                )),
                returned_document_ids=returned,
            )
        )
    return CorpusEvaluation(queries=tuple(evaluations))


def load_golden_corpus(path: str | Path) -> tuple[GoldenQuery, ...]:
    """Load and validate a JSON golden retrieval corpus."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("golden corpus root must be a JSON list")
    if not raw:
        raise ValueError("golden corpus must contain at least one query")

    queries: list[GoldenQuery] = []
    names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("every golden corpus item must be an object")
        query = _parse_query(item, index=index)
        normalized_name = query.name.casefold()
        if normalized_name in names:
            raise ValueError(f"golden query names must be unique: {query.name}")
        names.add(normalized_name)
        queries.append(query)
    return tuple(queries)


def _parse_query(item: dict[str, Any], *, index: int) -> GoldenQuery:
    allowed = {
        "name",
        "query",
        "mode",
        "expected_document_ids",
        "forbidden_document_ids",
        "limit",
    }
    unknown = set(item) - allowed
    if unknown:
        raise ValueError(f"unknown golden corpus fields at index {index}: {sorted(unknown)}")

    name = item.get("name")
    query_text = item.get("query")
    mode = item.get("mode", SearchMode.HYBRID.value)
    limit = item.get("limit", 10)
    if not isinstance(name, str) or not isinstance(query_text, str):
        raise ValueError(f"golden corpus name and query must be strings at index {index}")
    if not isinstance(mode, str):
        raise ValueError(f"golden corpus mode must be a string at index {index}")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError(f"golden corpus limit must be an integer at index {index}")

    return GoldenQuery(
        name=name,
        query=query_text,
        mode=SearchMode(mode),
        expected_document_ids=_parse_uuid_list(
            item.get("expected_document_ids", []),
            field="expected_document_ids",
            index=index,
        ),
        forbidden_document_ids=_parse_uuid_list(
            item.get("forbidden_document_ids", []),
            field="forbidden_document_ids",
            index=index,
        ),
        limit=limit,
    )


def _parse_uuid_list(value: Any, *, field: str, index: int) -> tuple[UUID, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of UUID strings at index {index}")
    try:
        return tuple(UUID(item) for item in value)
    except ValueError as exc:
        raise ValueError(f"{field} contains an invalid UUID at index {index}") from exc
