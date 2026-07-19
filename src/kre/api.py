from __future__ import annotations

from fastapi import FastAPI

from kre import __version__
from kre.composition import KREComponents, build_components
from kre.config import KRESettings
from kre.search.api import router as search_router


def create_app(settings: KRESettings | None = None) -> FastAPI:
    """Create a configured KRE HTTP application and attach process services."""

    resolved_settings = settings or KRESettings.from_env()
    components = build_components(resolved_settings)
    application = FastAPI(
        title="Lantern Knowledge Research Engine",
        version=__version__,
        description="Governed knowledge ingestion and retrieval for the Lantern Platform.",
    )
    application.state.kre = components
    application.state.search_backend = components.search_backend
    application.include_router(search_router)

    @application.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        return {
            "service": "kre-1",
            "status": "ok",
            "version": __version__,
            "embedding_provider": components.embeddings.name,
        }

    return application


def get_components(application: FastAPI) -> KREComponents:
    """Return the process-scoped dependency graph attached to an application."""

    return application.state.kre


app = create_app()
