from __future__ import annotations

from dataclasses import dataclass
from os import environ

from kre.storage.postgres_schema import PostgresSchemaConfig


@dataclass(frozen=True, slots=True)
class KRESettings:
    """Validated process configuration for the KRE composition root."""

    embedding_provider: str = "deterministic"
    embedding_endpoint: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str = "sha256-v1"
    embedding_dimensions: int = 16
    embedding_api_version: str | None = None
    embedding_api_key_header: str = "Authorization"
    storage_provider: str = "memory"
    postgres_dsn: str | None = None
    postgres_schema: str = "kre"

    def __post_init__(self) -> None:
        provider = self.embedding_provider.casefold()
        if provider not in {"deterministic", "openai-compatible"}:
            raise ValueError("embedding_provider must be deterministic or openai-compatible")
        if self.embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be at least 1")
        if provider == "openai-compatible":
            if not self.embedding_endpoint or not self.embedding_endpoint.strip():
                raise ValueError("embedding_endpoint is required for openai-compatible")
            if not self.embedding_api_key or not self.embedding_api_key.strip():
                raise ValueError("embedding_api_key is required for openai-compatible")

        storage = self.storage_provider.casefold()
        if storage not in {"memory", "postgres"}:
            raise ValueError("storage_provider must be memory or postgres")
        PostgresSchemaConfig(
            schema=self.postgres_schema,
            vector_dimensions=self.embedding_dimensions,
        )
        if storage == "postgres" and (
            not self.postgres_dsn or not self.postgres_dsn.strip()
        ):
            raise ValueError("postgres_dsn is required for postgres storage")

    @classmethod
    def from_env(cls) -> KRESettings:
        """Load settings from KRE-prefixed environment variables."""

        dimensions = environ.get("KRE_EMBEDDING_DIMENSIONS", "16")
        try:
            parsed_dimensions = int(dimensions)
        except ValueError as exc:
            raise ValueError("KRE_EMBEDDING_DIMENSIONS must be an integer") from exc

        return cls(
            embedding_provider=environ.get("KRE_EMBEDDING_PROVIDER", "deterministic"),
            embedding_endpoint=environ.get("KRE_EMBEDDING_ENDPOINT"),
            embedding_api_key=environ.get("KRE_EMBEDDING_API_KEY"),
            embedding_model=environ.get("KRE_EMBEDDING_MODEL", "sha256-v1"),
            embedding_dimensions=parsed_dimensions,
            embedding_api_version=environ.get("KRE_EMBEDDING_API_VERSION"),
            embedding_api_key_header=environ.get(
                "KRE_EMBEDDING_API_KEY_HEADER", "Authorization"
            ),
            storage_provider=environ.get("KRE_STORAGE_PROVIDER", "memory"),
            postgres_dsn=environ.get("KRE_POSTGRES_DSN"),
            postgres_schema=environ.get("KRE_POSTGRES_SCHEMA", "kre"),
        )
