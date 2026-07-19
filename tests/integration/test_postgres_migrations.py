from __future__ import annotations

from os import environ

import pytest

from kre.storage import (
    PostgresMigrationRunner,
    PostgresSchemaConfig,
    initial_postgres_migrations,
)

asyncpg = pytest.importorskip("asyncpg")


@pytest.mark.asyncio
async def test_postgres_migrations_are_repeatable_and_recorded_once() -> None:
    dsn = environ.get("KRE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KRE_TEST_POSTGRES_DSN is not configured")

    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2)
    schema = "kre_migration_integration"
    config = PostgresSchemaConfig(schema=schema, vector_dimensions=3)
    runner = PostgresMigrationRunner(
        pool,
        schema=schema,
        migrations=initial_postgres_migrations(config),
    )

    try:
        async with pool.acquire() as connection:
            await connection.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")

        assert await runner.apply() == (1,)
        assert await runner.apply() == ()

        async with pool.acquire() as connection:
            rows = await connection.fetch(
                f"SELECT version, name, checksum FROM {schema}.schema_migrations "
                "ORDER BY version"
            )
            document_table = await connection.fetchval(
                "SELECT to_regclass($1)", f"{schema}.knowledge_documents"
            )
            embedding_table = await connection.fetchval(
                "SELECT to_regclass($1)", f"{schema}.semantic_embeddings"
            )

        assert len(rows) == 1
        assert rows[0]["version"] == 1
        assert rows[0]["name"] == "initial_knowledge_and_vector_schema"
        assert len(rows[0]["checksum"]) == 64
        assert document_table == f"{schema}.knowledge_documents"
        assert embedding_table == f"{schema}.semantic_embeddings"
    finally:
        async with pool.acquire() as connection:
            await connection.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()
