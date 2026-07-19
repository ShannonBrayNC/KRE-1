from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from kre.api import create_app, get_components
from kre.config import KRESettings
from kre.embeddings import DeterministicEmbeddingProvider, OpenAICompatibleEmbeddingProvider
from kre.models import KnowledgeChunk, KnowledgeDocument, Provenance


def test_settings_validate_provider_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KRE_EMBEDDING_PROVIDER", "openai-compatible")
    monkeypatch.setenv("KRE_EMBEDDING_ENDPOINT", "https://api.test/embeddings")
    monkeypatch.setenv("KRE_EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv("KRE_EMBEDDING_MODEL", "embed")
    monkeypatch.setenv("KRE_EMBEDDING_DIMENSIONS", "8")

    settings = KRESettings.from_env()

    assert settings.embedding_provider == "openai-compatible"
    assert settings.embedding_dimensions == 8

    with pytest.raises(ValueError, match="embedding_endpoint"):
        KRESettings(embedding_provider="openai-compatible", embedding_api_key="secret")


def test_composition_selects_embedding_provider() -> None:
    deterministic_app = create_app(KRESettings())
    assert isinstance(get_components(deterministic_app).embeddings, DeterministicEmbeddingProvider)

    openai_app = create_app(
        KRESettings(
            embedding_provider="openai-compatible",
            embedding_endpoint="https://api.test/embeddings",
            embedding_api_key="secret",
            embedding_model="embed",
            embedding_dimensions=8,
        )
    )
    assert isinstance(get_components(openai_app).embeddings, OpenAICompatibleEmbeddingProvider)


@pytest.mark.asyncio
async def test_configured_app_executes_security_trimmed_keyword_search() -> None:
    application = create_app(KRESettings())
    components = get_components(application)
    document = KnowledgeDocument(
        title="Lantern Public Knowledge",
        content="",
        provenance=Provenance(
            source_system="test",
            connector="test",
            content_hash="sha256:composition",
        ),
    )
    chunk = KnowledgeChunk(
        document_id=document.id,
        sequence=0,
        text="SignalForge governs the Lantern registry.",
    )
    await components.repository.upsert_document(document)
    await components.repository.replace_chunks(document.id, [chunk])

    response = TestClient(application).post(
        "/search",
        json={"query": "Lantern registry", "mode": "keyword"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["results"][0]["classification"] == "public"
    assert payload["results"][0]["keyword"]["rank"] == 1


def test_health_reports_selected_provider() -> None:
    response = TestClient(create_app(KRESettings())).get("/health")

    assert response.status_code == 200
    assert response.json()["embedding_provider"] == "deterministic"
