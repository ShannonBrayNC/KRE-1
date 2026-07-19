from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

from kre.composition import build_components, build_components_async
from kre.config import KRESettings
from kre.search import PgVectorSemanticIndex
from kre.storage import PostgresKnowledgeRepository


class FakeConnection:
    def __init__(self, *, fail_migration: bool = False) -> None:
        self.fail_migration = fail_migration
        self.executed: list[str] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append(query)
        if self.fail_migration and "schema_migrations" in query:
            raise RuntimeError("migration failed")
        return "SELECT 1" if query.lstrip().startswith("SELECT") else "OK"

    async def fetchrow(self, query: str, *args: Any):
        raise AssertionError("fetchrow is not expected during composition")

    async def fetch(self, query: str, *args: Any):
        self.executed.append(query)
        return []

    @asynccontextmanager
    async def transaction(self):
        yield


class FakePool:
    def __init__(self, *, fail_migration: bool = False) -> None:
        self.closed = False
        self.connection = FakeConnection(fail_migration=fail_migration)

    @asynccontextmanager
    async def acquire(self):
        yield self.connection

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
    assert pool.connection.executed == []
    await components.close()
    assert pool.closed is True


@pytest.mark.asyncio
async def test_async_composition_applies_governed_migrations_when_enabled() -> None:
    pool = FakePool()

    async def pool_factory(dsn: str) -> FakePool:
        return pool

    components = await build_components_async(
        KRESettings(
            storage_provider="postgres",
            postgres_dsn="postgresql://db/kre",
            postgres_apply_migrations=True,
            embedding_dimensions=8,
        ),
        postgres_pool_factory=pool_factory,
    )

    assert any("schema_migrations" in query for query in pool.connection.executed)
    assert any("knowledge_documents" in query for query in pool.connection.executed)
    await components.close()


@pytest.mark.asyncio
async def test_failed_startup_migration_closes_owned_pool() -> None:
    pool = FakePool(fail_migration=True)

    async def pool_factory(dsn: str) -> FakePool:
        return pool

    with pytest.raises(RuntimeError, match="migration failed"):
        await build_components_async(
            KRESettings(
                storage_provider="postgres",
                postgres_dsn="postgresql://db/kre",
                postgres_apply_migrations=True,
            ),
            postgres_pool_factory=pool_factory,
        )

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
    with pytest.raises(ValueError, match="boolean"):
        KRESettings(postgres_apply_migrations="yes")  # type: ignore[arg-type]

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
    monkeypatch.setenv("KRE_POSTGRES_APPLY_MIGRATIONS", "true")

    settings = KRESettings.from_env()

    assert settings.storage_provider == "postgres"
    assert settings.postgres_dsn == "postgresql://db/kre"
    assert settings.postgres_schema == "lantern_kre"
    assert settings.embedding_dimensions == 8
    assert settings.postgres_apply_migrations is True


def test_settings_reject_invalid_migration_environment_flag(monkeypatch) -> None:
    monkeypatch.setenv("KRE_POSTGRES_APPLY_MIGRATIONS", "sometimes")

    with pytest.raises(ValueError, match="KRE_POSTGRES_APPLY_MIGRATIONS"):
        KRESettings.from_env()
