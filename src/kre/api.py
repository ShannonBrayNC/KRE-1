from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from kre import __version__
from kre.composition import (
    KREComponents,
    PostgresPoolFactory,
    build_components,
    build_components_async,
)
from kre.config import KRESettings
from kre.runtime import create_asyncpg_pool
from kre.search.api import router as search_router


def _attach_components(application: FastAPI, components: KREComponents) -> None:
    application.state.kre = components
    application.state.search_backend = components.search_backend


def create_app(
    settings: KRESettings | None = None,
    *,
    postgres_pool_factory: PostgresPoolFactory | None = None,
) -> FastAPI:
    """Create a configured KRE HTTP application with governed resource lifetime."""

    resolved_settings = settings or KRESettings.from_env()
    memory_components = (
        build_components(resolved_settings)
        if resolved_settings.storage_provider.casefold() == "memory"
        else None
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        components = memory_components
        if components is None:
            components = await build_components_async(
                resolved_settings,
                postgres_pool_factory=postgres_pool_factory or create_asyncpg_pool,
            )
            _attach_components(application, components)
        try:
            yield
        finally:
            await components.close()

    application = FastAPI(
        title="Lantern Knowledge Research Engine",
        version=__version__,
        description="Governed knowledge ingestion and retrieval for the Lantern Platform.",
        lifespan=lifespan,
    )
    if memory_components is not None:
        _attach_components(application, memory_components)
    application.include_router(search_router)

    @application.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        components = get_components(application)
        return {
            "service": "kre-1",
            "status": "ok",
            "version": __version__,
            "embedding_provider": components.embeddings.name,
            "storage_provider": components.settings.storage_provider,
        }

    return application


def get_components(application: FastAPI) -> KREComponents:
    """Return the process-scoped dependency graph attached to an application."""

    components: KREComponents | None = getattr(application.state, "kre", None)
    if components is None:
        raise RuntimeError("KRE application startup has not completed")
    return components


app = create_app()
