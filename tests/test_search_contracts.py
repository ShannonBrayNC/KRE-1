from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from kre.schemas import SearchMode, SearchRequest, SearchResponse
from kre.search import HybridSearchResult, hybrid_response


def test_search_request_defaults_and_validation() -> None:
    request = SearchRequest(query="  governed knowledge  ")

    assert request.query == "governed knowledge"
    assert request.mode is SearchMode.HYBRID
    assert request.limit == 10
    assert request.minimum_semantic_score == -1.0

    with pytest.raises(ValidationError, match="candidate_limit"):
        SearchRequest(query="knowledge", limit=10, candidate_limit=5)
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")
    with pytest.raises(ValidationError):
        SearchRequest(query="knowledge", unexpected=True)


def test_hybrid_response_projects_channel_observability() -> None:
    document_id = uuid4()
    chunk_id = uuid4()
    result = HybridSearchResult(
        document_id=document_id,
        chunk_id=chunk_id,
        document_title="Lantern Governance",
        sequence=2,
        text="SignalForge governs registry decisions.",
        score=0.031,
        keyword_rank=1,
        semantic_rank=3,
        keyword_score=3.5,
        semantic_score=0.92,
        source_uri="https://github.com/ShannonBrayNC/KRE-1",
        source_system="github",
        source_version="abc123",
        classification="confidential",
        security_labels=("lantern-engineering",),
    )

    response = hybrid_response("  registry  ", [result])

    assert response.query == "registry"
    assert response.mode is SearchMode.HYBRID
    assert response.count == 1
    assert response.results[0].document_id == document_id
    assert response.results[0].chunk_id == chunk_id
    assert response.results[0].keyword is not None
    assert response.results[0].keyword.rank == 1
    assert response.results[0].semantic is not None
    assert response.results[0].semantic.score == pytest.approx(0.92)
    assert response.results[0].security_labels == ("lantern-engineering",)


def test_search_response_enforces_count_integrity() -> None:
    with pytest.raises(ValidationError, match="count"):
        SearchResponse(
            query="knowledge",
            mode=SearchMode.KEYWORD,
            count=1,
            results=(),
        )


def test_search_contract_serializes_stable_json_shape() -> None:
    response = hybrid_response("knowledge", [])
    payload = response.model_dump(mode="json")

    assert payload == {
        "query": "knowledge",
        "mode": "hybrid",
        "count": 0,
        "results": [],
    }
