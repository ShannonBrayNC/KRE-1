from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

from kre.storage import (
    PostgresMigration,
    PostgresMigrationRunner,
    PostgresSchemaConfig,
    initial_postgres_migrations,
)


class FakeConnection:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = rows or []
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.transaction_entries = 0

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        return "SELECT 1" if query.lstrip().startswith("SELECT") else "OK"

    async def fetchrow(self, query: str, *args: Any):
        raise AssertionError("fetchrow is not expected")

    async def fetch(self, query: str, *args: Any):
        self.executed.append((query, args))
        return self.rows

    @asynccontextmanager
    async def transaction(self):
        self.transaction_entries += 1
        yield


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


@pytest.mark.asyncio
async def test_runner_applies_pending_migrations_in_one_transaction() -> None:
    connection = FakeConnection()
    migrations = (
        PostgresMigration(version=1, name="one", sql="CREATE TABLE one(id integer);"),
        PostgresMigration(version=2, name="two", sql="CREATE TABLE two(id integer);"),
    )
    runner = PostgresMigrationRunner(FakePool(connection), migrations=migrations)

    applied = await runner.apply()

    assert applied == (1, 2)
    assert connection.transaction_entries == 1
    statements = [query for query, _ in connection.executed]
    assert any("schema_migrations" in query for query in statements)
    assert any("pg_advisory_xact_lock" in query for query in statements)
    assert migrations[0].sql in statements
    assert migrations[1].sql in statements


@pytest.mark.asyncio
async def test_runner_skips_verified_applied_migrations() -> None:
    migration = PostgresMigration(version=1, name="one", sql="SELECT 1;")
    connection = FakeConnection(
        rows=[{"version": 1, "name": migration.name, "checksum": migration.checksum}]
    )
    runner = PostgresMigrationRunner(FakePool(connection), migrations=(migration,))

    assert await runner.apply() == ()
    assert [query for query, _ in connection.executed].count(migration.sql) == 0


@pytest.mark.asyncio
async def test_runner_fails_closed_on_drift_or_unknown_versions() -> None:
    migration = PostgresMigration(version=1, name="one", sql="SELECT 1;")

    drifted = FakeConnection(rows=[{"version": 1, "name": "one", "checksum": "wrong"}])
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        await PostgresMigrationRunner(FakePool(drifted), migrations=(migration,)).apply()

    unknown = FakeConnection(rows=[{"version": 2, "name": "two", "checksum": "x"}])
    with pytest.raises(RuntimeError, match="unknown migration"):
        await PostgresMigrationRunner(FakePool(unknown), migrations=(migration,)).apply()


@pytest.mark.asyncio
async def test_runner_rejects_noncontiguous_database_history() -> None:
    migrations = (
        PostgresMigration(version=1, name="one", sql="SELECT 1;"),
        PostgresMigration(version=2, name="two", sql="SELECT 2;"),
    )
    connection = FakeConnection(
        rows=[
            {
                "version": 2,
                "name": migrations[1].name,
                "checksum": migrations[1].checksum,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="contiguous prefix"):
        await PostgresMigrationRunner(FakePool(connection), migrations=migrations).apply()


def test_migration_plan_and_runner_validation() -> None:
    plan = initial_postgres_migrations(PostgresSchemaConfig(vector_dimensions=8))
    assert len(plan) == 1
    assert plan[0].version == 1
    assert "vector(8)" in plan[0].sql
    assert len(plan[0].checksum) == 64

    with pytest.raises(ValueError, match="contiguous"):
        PostgresMigrationRunner(
            FakePool(FakeConnection()),
            migrations=(PostgresMigration(version=2, name="two", sql="SELECT 2"),),
        )
    with pytest.raises(ValueError, match="unique"):
        PostgresMigrationRunner(
            FakePool(FakeConnection()),
            migrations=(
                PostgresMigration(version=1, name="one", sql="SELECT 1"),
                PostgresMigration(version=1, name="again", sql="SELECT 2"),
            ),
        )
