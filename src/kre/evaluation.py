from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
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

    @property
    def mean_recall_at_k(self) -> float:
        return sum(item.recall_at_k for item in self.queries) / len(self.queries)

    @property
    def mean_reciprocal_rank(self) -> float:
        return sum(item.reciprocal_rank for item in self.queries) / len(self.queries)

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
        returned = tuple(hit.document_id for hit in response.results)
        returned_set = set(returned)
        expected_set = set(golden.expected_document_ids)
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
                forbidden_hits=tuple(
                    document_id
                    for document_id in returned
                    if document_id in set(golden.forbidden_document_ids)
                ),
                returned_document_ids=returned,
            )
        )
    return CorpusEvaluation(queries=tuple(evaluations))


def load_golden_corpus(path: str | Path) -> tuple[GoldenQuery, ...]:
    """Load and validate a JSON golden retrieval corpus."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("golden corpus root must be a JSON list")

    queries: list[GoldenQuery] = []
    names: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("every golden corpus item must be an object")
        query = GoldenQuery(
            name=str(item.get("name", "")),
            query=str(item.get("query", "")),
            mode=SearchMode(item.get("mode", SearchMode.HYBRID.value)),
            expected_document_ids=tuple(UUID(value) for value in item.get("expected_document_ids", [])),
            forbidden_document_ids=tuple(UUID(value) for value in item.get("forbidden_document_ids", [])),
            limit=int(item.get("limit", 10)),
        )
        if query.name in names:
            raise ValueError(f"golden query names must be unique: {query.name}")
        names.add(query.name)
        queries.append(query)
    return tuple(queries)
