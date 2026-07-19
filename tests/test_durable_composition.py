from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

from kre.composition import build_components, build_components_async
from kre.config import KRESettings
from kre.search import PgVectorSemanticIndex
from kre.storage import PostgresKnowledgeRepository


class FakeConnection:
    async def execute(self, query: str, *args: Any) -> str:
        raise AssertionError("database access is not expected during composition")

    async def fetchrow(self, query: str, *args: Any):
        raise AssertionError("database access is not expected during composition")

    async def fetch(self, query: str, *args: Any):
        raise AssertionError("database access is not expected during composition")

    @asynccontextmanager
    async def transaction(self):
        yield


class FakePool:
    def __init__(self) -> None:
        self.closed = False

    @asynccontextmanager
    async def acquire(self):
        yield FakeConnection()

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_async_composition_builds_postgres_adapters() -> None:
    captured: list[str] = []
    pool = FakePool()

    async def pool_factory(dsn: str) -> FakePool:
        captured.append(dsn)
        return pool

    settings = KRESettings(
        storage_provider="postgres",
        postgres_dsn="postgresql://kre@db/knowledge",
        postgres_schema="lantern_kre",
        embedding_dimensions=8,
    )

    components = await build_components_async(
        settings,
        postgres_pool_factory=pool_factory,
    )

    assert captured == ["postgresql://kre@db/knowledge"]
    assert isinstance(components.repository, PostgresKnowledgeRepository)
    assert isinstance(components.semantic_index, PgVectorSemanticIndex)
    await components.close()
    assert pool.closed is True


@pytest.mark.asyncio
async def test_async_composition_preserves_memory_default() -> None:
    components = await build_components_async(KRESettings())

    assert components.settings.storage_provider == "memory"
    assert components.telemetry.snapshot() == ()
    await components.close()


def test_postgres_configuration_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="postgres_dsn"):
        KRESettings(storage_provider="postgres")
    with pytest.raises(ValueError, match="storage_provider"):
        KRESettings(storage_provider="sqlite")
    with pytest.raises(ValueError, match="identifier"):
        KRESettings(postgres_schema="kre;drop")

    settings = KRESettings(
        storage_provider="postgres",
        postgres_dsn="postgresql://db/kre",
    )
    with pytest.raises(RuntimeError, match="build_components_async"):
        build_components(settings)


@pytest.mark.asyncio
async def test_postgres_composition_requires_explicit_pool_factory() -> None:
    settings = KRESettings(
        storage_provider="postgres",
        postgres_dsn="postgresql://db/kre",
    )

    with pytest.raises(RuntimeError, match="postgres_pool_factory"):
        await build_components_async(settings)


def test_settings_load_durable_storage_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("KRE_STORAGE_PROVIDER", "postgres")
    monkeypatch.setenv("KRE_POSTGRES_DSN", "postgresql://db/kre")
    monkeypatch.setenv("KRE_POSTGRES_SCHEMA", "lantern_kre")
    monkeypatch.setenv("KRE_EMBEDDING_DIMENSIONS", "8")

    settings = KRESettings.from_env()

    assert settings.storage_provider == "postgres"
    assert settings.postgres_dsn == "postgresql://db/kre"
    assert settings.postgres_schema == "lantern_kre"
    assert settings.embedding_dimensions == 8
