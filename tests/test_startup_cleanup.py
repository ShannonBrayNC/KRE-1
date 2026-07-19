from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

from kre.composition import build_components_async
from kre.config import KRESettings


class FailingConnection:
    async def execute(self, query: str, *args: Any) -> str:
        if "schema_migrations" in query:
            raise RuntimeError("migration failed")
        return "OK"

    async def fetchrow(self, query: str, *args: Any):
        return None

    async def fetch(self, query: str, *args: Any):
        return []

    @asynccontextmanager
    async def transaction(self):
        yield


class FailingClosePool:
    @asynccontextmanager
    async def acquire(self):
        yield FailingConnection()

    async def close(self) -> None:
        raise RuntimeError("close failed")


@pytest.mark.asyncio
async def test_startup_failure_is_not_masked_by_cleanup_failure() -> None:
    async def pool_factory(dsn: str) -> FailingClosePool:
        return FailingClosePool()

    with pytest.raises(RuntimeError, match="migration failed"):
        await build_components_async(
            KRESettings(
                storage_provider="postgres",
                postgres_dsn="postgresql://db/kre",
                postgres_apply_migrations=True,
            ),
            postgres_pool_factory=pool_factory,
        )
