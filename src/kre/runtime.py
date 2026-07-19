from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from kre.composition import ManagedPostgresPool


async def create_asyncpg_pool(dsn: str) -> ManagedPostgresPool:
    """Create the deployment PostgreSQL pool without a mandatory base dependency."""

    try:
        asyncpg: Any = import_module("asyncpg")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "postgres storage requires the optional 'postgres' dependency"
        ) from exc

    pool = await asyncpg.create_pool(dsn=dsn)
    return cast(ManagedPostgresPool, pool)
