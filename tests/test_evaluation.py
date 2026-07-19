from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from kre.evaluation import CorpusEvaluation, GoldenQuery, evaluate_corpus, load_golden_corpus
from kre.schemas import SearchHit, SearchMode, SearchResponse


EXPECTED = UUID("11111111-1111-1111-1111-111111111111")
SECOND = UUID("22222222-2222-2222-2222-222222222222")
FORBIDDEN = UUID("33333333-3333-3333-3333-333333333333")
CHUNK = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


class FakeBackend:
    async def execute(self, request):
        document_ids = (
            (SECOND, EXPECTED)
            if request.query == "Lantern registry governance"
            else (FORBIDDEN,)
        )
        results = tuple(
            SearchHit(
                document_id=document_id,
                chunk_id=CHUNK,
                document_title=f"Document {rank}",
                sequence=rank - 1,
                text="retrieval result",
                score=1.0 / rank,
                classification="public",
            )
            for rank, document_id in enumerate(document_ids, start=1)
        )
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            count=len(results),
            results=results,
        )


class MismatchedBackend(FakeBackend):
    async def execute(self, request):
        response = await super().execute(request)
        return response.model_copy(update={"query": "different query"})


def write_corpus(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "name": "lantern-registry",
                    "query": "Lantern registry governance",
                    "mode": "hybrid",
                    "limit": 3,
                    "expected_document_ids": [str(EXPECTED)],
                    "forbidden_document_ids": [str(FORBIDDEN)],
                },
                {
                    "name": "signalforge-controls",
                    "query": "SignalForge deployment controls",
                    "mode": "keyword",
                    "limit": 2,
                    "expected_document_ids": [str(SECOND)],
                    "forbidden_document_ids": [str(FORBIDDEN)],
                },
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_evaluator_calculates_recall_rank_and_forbidden_hits(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    write_corpus(path)
    corpus = load_golden_corpus(path)

    evaluation = await evaluate_corpus(FakeBackend(), corpus)

    assert evaluation.mean_recall_at_k == 0.5
    assert evaluation.mean_reciprocal_rank == 0.25
    assert evaluation.queries[0].recall_at_k == 1.0
    assert evaluation.queries[0].reciprocal_rank == 0.5
    assert evaluation.queries[0].forbidden_hits == ()
    assert evaluation.queries[1].forbidden_hits == (FORBIDDEN,)
    assert evaluation.passed is False


def test_loader_rejects_duplicate_names_and_invalid_corpora(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    write_corpus(path)
    items = json.loads(path.read_text(encoding="utf-8"))
    items[1]["name"] = items[0]["name"].upper()
    path.write_text(json.dumps(items), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_golden_corpus(path)

    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        load_golden_corpus(path)

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        load_golden_corpus(path)


def test_loader_rejects_coercion_and_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    base = {
        "name": "strict-contract",
        "query": "governed retrieval",
        "mode": "hybrid",
        "expected_document_ids": [str(EXPECTED)],
    }

    path.write_text(json.dumps([{**base, "limit": True}]), encoding="utf-8")
    with pytest.raises(ValueError, match="integer"):
        load_golden_corpus(path)

    path.write_text(json.dumps([{**base, "unexpected": "value"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_golden_corpus(path)

    path.write_text(json.dumps([{**base, "expected_document_ids": [1]}]), encoding="utf-8")
    with pytest.raises(ValueError, match="UUID strings"):
        load_golden_corpus(path)


def test_golden_query_rejects_duplicate_document_identifiers() -> None:
    with pytest.raises(ValueError, match="unique"):
        GoldenQuery(
            name="duplicates",
            query="duplicate expected identities",
            mode=SearchMode.HYBRID,
            expected_document_ids=(EXPECTED, EXPECTED),
        )


@pytest.mark.asyncio
async def test_evaluator_rejects_empty_and_mismatched_corpora() -> None:
    with pytest.raises(ValueError, match="at least one"):
        await evaluate_corpus(FakeBackend(), ())
    with pytest.raises(ValueError, match="at least one"):
        CorpusEvaluation(queries=())

    query = GoldenQuery(
        name="mismatch",
        query="Lantern registry governance",
        mode=SearchMode.HYBRID,
        expected_document_ids=(EXPECTED,),
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        await evaluate_corpus(MismatchedBackend(), (query,))
