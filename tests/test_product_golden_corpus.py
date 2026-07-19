from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from kre.composition import build_components
from kre.config import KRESettings
from kre.evaluation import evaluate_corpus, load_golden_corpus
from kre.models import Classification, KnowledgeChunk, KnowledgeDocument, Provenance
from kre.schemas import SearchMode


CORPUS_PATH = Path("evaluation/golden/lantern_products.json")
INGESTED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
PRODUCTS = (
    (
        UUID("10000000-0000-0000-0000-000000000001"),
        "OpsHelm",
        "OpsHelm provides ticket intake, support case analysis, log parsing, and operational follow-up.",
    ),
    (
        UUID("10000000-0000-0000-0000-000000000002"),
        "ETS",
        "ETS preserves evidence provenance, traceability, source lineage, and governed audit records.",
    ),
    (
        UUID("10000000-0000-0000-0000-000000000003"),
        "SignalForge",
        "SignalForge is the reusable Azure deployment wrapper for governed Lantern services.",
    ),
    (
        UUID("10000000-0000-0000-0000-000000000004"),
        "EchoChamber",
        "EchoChamber supports multilingual voice processing, transcription, and translation.",
    ),
    (
        UUID("10000000-0000-0000-0000-000000000005"),
        "Christina",
        "Christina provides orchestration, sprint runner control, and workflow coordination.",
    ),
)


@pytest.mark.asyncio
async def test_lantern_product_golden_corpus_passes_real_keyword_retrieval() -> None:
    components = build_components(KRESettings())
    repository = components.repository

    try:
        for index, (document_id, title, content) in enumerate(PRODUCTS, start=1):
            document = KnowledgeDocument(
                id=document_id,
                title=title,
                content=content,
                classification=Classification.INTERNAL,
                provenance=Provenance(
                    source_system="lantern-product-registry",
                    connector="golden-corpus",
                    content_hash=f"sha256:lantern-product-{index}",
                    ingested_at=INGESTED_AT,
                    source_version="1",
                ),
            )
            chunk = KnowledgeChunk(
                id=UUID(f"20000000-0000-0000-0000-{index:012d}"),
                document_id=document_id,
                sequence=0,
                text=content,
                token_count=len(content.split()),
                section="product-overview",
            )
            await repository.upsert_document(document)
            await repository.replace_chunks(document.id, (chunk,))

        corpus = load_golden_corpus(CORPUS_PATH)
        product_ids = {document_id for document_id, _, _ in PRODUCTS}
        expected_ids = {
            document_id
            for query in corpus
            for document_id in query.expected_document_ids
        }

        assert len(corpus) == len(PRODUCTS)
        assert expected_ids == product_ids
        assert all(query.mode is SearchMode.KEYWORD for query in corpus)
        assert all(query.limit == 1 for query in corpus)

        evaluation = await evaluate_corpus(components.search_backend, corpus)

        assert evaluation.passed is True
        assert evaluation.mean_recall_at_k == 1.0
        assert evaluation.mean_reciprocal_rank == 1.0
        assert all(not query.forbidden_hits for query in evaluation.queries)
    finally:
        await components.close()
