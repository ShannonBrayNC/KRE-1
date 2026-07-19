from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

from kre.api import create_app, get_components
from kre.config import KRESettings
from kre.runtime import create_asyncpg_pool
from kre.storage import PostgresKnowledgeRepository


class FakeConnection:
    async def execute(self, query: str, *args: Any) -> str:
        raise AssertionError("database access is not expected during startup")

    async def fetchrow(self, query: str, *args: Any):
        raise AssertionError("database access is not expected during startup")

    async def fetch(self, query: str, *args: Any):
        raise AssertionError("database access is not expected during startup")

    @asynccontextmanager
    async def transaction(self):
        yield


class ManagedFakePool:
    def __init__(self) -> None:
        self.closed = False

    @asynccontextmanager
    async def acquire(self):
        yield FakeConnection()

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_postgres_application_owns_pool_for_lifespan() -> None:
    pool = ManagedFakePool()
    captured: list[str] = []

    async def pool_factory(dsn: str) -> ManagedFakePool:
        captured.append(dsn)
        return pool

    settings = KRESettings(
        storage_provider="postgres",
        postgres_dsn="postgresql://db/kre",
        embedding_dimensions=8,
    )
    application = create_app(settings, postgres_pool_factory=pool_factory)

    with pytest.raises(RuntimeError, match="startup"):
        get_components(application)

    async with application.router.lifespan_context(application):
        components = get_components(application)
        assert captured == ["postgresql://db/kre"]
        assert isinstance(components.repository, PostgresKnowledgeRepository)
        assert pool.closed is False

    assert pool.closed is True


@pytest.mark.asyncio
async def test_memory_application_remains_available_before_startup() -> None:
    application = create_app(KRESettings())

    components = get_components(application)
    assert components.settings.storage_provider == "memory"

    async with application.router.lifespan_context(application):
        assert get_components(application) is components


@pytest.mark.asyncio
async def test_asyncpg_factory_reports_missing_optional_dependency(monkeypatch) -> None:
    def missing_module(name: str):
        assert name == "asyncpg"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("kre.runtime.import_module", missing_module)

    with pytest.raises(RuntimeError, match="optional 'postgres' dependency"):
        await create_asyncpg_pool("postgresql://db/kre")
