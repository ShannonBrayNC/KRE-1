from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from pathlib import PurePosixPath

import httpx

from kre.connectors.base import Connector
from kre.models import KnowledgeDocument, Provenance
from kre.processing import content_hash, normalize_text


class GitHubConnector(Connector):
    """Read text knowledge from one GitHub repository through the REST API."""

    name = "github"
    supported_suffixes = {".md", ".markdown", ".txt", ".rst"}

    def __init__(
        self,
        repository: str,
        *,
        ref: str = "main",
        token: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if repository.count("/") != 1:
            raise ValueError("repository must use owner/name format")

        self.repository = repository
        self.ref = ref
        self._owns_client = client is None
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = client or httpx.AsyncClient(
            base_url="https://api.github.com",
            headers=headers,
            timeout=30.0,
        )

    async def __aenter__(self) -> GitHubConnector:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def discover(self) -> AsyncIterator[str]:
        response = await self._client.get(
            f"/repos/{self.repository}/git/trees/{self.ref}",
            params={"recursive": "1"},
        )
        response.raise_for_status()

        for item in response.json().get("tree", []):
            path = item.get("path")
            if (
                item.get("type") == "blob"
                and isinstance(path, str)
                and PurePosixPath(path).suffix.lower() in self.supported_suffixes
            ):
                yield path

    async def fetch(self, source_id: str) -> KnowledgeDocument:
        path = PurePosixPath(source_id)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source_id must be a safe repository-relative path")
        if path.suffix.lower() not in self.supported_suffixes:
            raise ValueError(f"unsupported knowledge file: {source_id}")

        response = await self._client.get(
            f"/repos/{self.repository}/contents/{source_id}",
            params={"ref": self.ref},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("type") != "file":
            raise ValueError(f"source is not a file: {source_id}")

        encoded = payload.get("content")
        if not isinstance(encoded, str):
            raise ValueError("GitHub response did not include file content")

        raw = base64.b64decode(encoded).decode("utf-8")
        normalized = normalize_text(raw)
        html_url = payload.get("html_url")

        return KnowledgeDocument(
            title=path.stem.replace("-", " ").replace("_", " ").strip() or path.name,
            content=normalized,
            mime_type="text/markdown" if path.suffix.lower() in {".md", ".markdown"} else "text/plain",
            provenance=Provenance(
                source_system="github",
                source_uri=html_url,
                connector=self.name,
                content_hash=content_hash(normalized),
                source_version=payload.get("sha"),
                attributes={
                    "repository": self.repository,
                    "ref": self.ref,
                    "path": source_id,
                },
            ),
        )
