from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

from kre.storage.postgres import PostgresPool
from kre.storage.postgres_schema import PostgresSchemaConfig, render_postgres_schema

_MIGRATION_LOCK_ID = 4_932_001


@dataclass(frozen=True, slots=True)
class PostgresMigration:
    """One immutable, ordered PostgreSQL schema migration."""

    version: int
    name: str
    sql: str

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("migration version must be at least 1")
        if not self.name.strip():
            raise ValueError("migration name must not be empty")
        if not self.sql.strip():
            raise ValueError("migration SQL must not be empty")

    @property
    def checksum(self) -> str:
        return sha256(self.sql.encode("utf-8")).hexdigest()


def initial_postgres_migrations(config: PostgresSchemaConfig) -> tuple[PostgresMigration, ...]:
    """Return the ordered built-in migration plan for one KRE schema."""

    return (
        PostgresMigration(
            version=1,
            name="initial_knowledge_and_vector_schema",
            sql=render_postgres_schema(config),
        ),
    )


class PostgresMigrationRunner:
    """Apply immutable migrations exactly once under a transaction-scoped lock."""

    def __init__(
        self,
        pool: PostgresPool,
        *,
        schema: str = "kre",
        migrations: Sequence[PostgresMigration],
    ) -> None:
        PostgresSchemaConfig(schema=schema)
        ordered = tuple(sorted(migrations, key=lambda item: item.version))
        versions = [item.version for item in ordered]
        if len(versions) != len(set(versions)):
            raise ValueError("migration versions must be unique")
        if versions != list(range(1, len(versions) + 1)):
            raise ValueError("migration versions must be contiguous starting at 1")
        self._pool = pool
        self._schema = schema
        self._migrations = ordered

    async def apply(self) -> tuple[int, ...]:
        """Apply pending migrations and return the versions applied in this run."""

        bootstrap = f"""
        CREATE SCHEMA IF NOT EXISTS {self._schema};
        CREATE TABLE IF NOT EXISTS {self._schema}.schema_migrations (
            version integer PRIMARY KEY,
            name text NOT NULL,
            checksum text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT now()
        );
        """
        select_applied = (
            f"SELECT version, name, checksum FROM {self._schema}.schema_migrations "
            "ORDER BY version"
        )
        insert_applied = f"""
        INSERT INTO {self._schema}.schema_migrations (version, name, checksum)
        VALUES ($1, $2, $3)
        """

        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(bootstrap)
                await connection.execute("SELECT pg_advisory_xact_lock($1)", _MIGRATION_LOCK_ID)
                rows = await connection.fetch(select_applied)
                applied = self._validate_applied(rows)

                applied_now: list[int] = []
                for migration in self._migrations:
                    if migration.version in applied:
                        continue
                    await connection.execute(migration.sql)
                    await connection.execute(
                        insert_applied,
                        migration.version,
                        migration.name,
                        migration.checksum,
                    )
                    applied_now.append(migration.version)
                return tuple(applied_now)

    def _validate_applied(self, rows: Sequence[Mapping[str, object]]) -> set[int]:
        known = {migration.version: migration for migration in self._migrations}
        applied: set[int] = set()
        for row in rows:
            version = int(row["version"])
            migration = known.get(version)
            if migration is None:
                raise RuntimeError(f"database contains unknown migration version: {version}")
            if row["name"] != migration.name:
                raise RuntimeError(f"migration name mismatch for version {version}")
            if row["checksum"] != migration.checksum:
                raise RuntimeError(f"migration checksum mismatch for version {version}")
            applied.add(version)
        return applied
