from fastapi import FastAPI

from kre import __version__

app = FastAPI(
    title="Lantern Knowledge Research Engine",
    version=__version__,
    description="Governed knowledge ingestion and retrieval for the Lantern Platform.",
)


@app.get("/health", tags=["operations"])
async def health() -> dict[str, str]:
    return {"service": "kre-1", "status": "ok", "version": __version__}
