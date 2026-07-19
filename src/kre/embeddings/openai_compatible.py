from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from kre.embeddings.base import EmbeddingProvider, EmbeddingVector


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Embedding adapter for OpenAI and Azure OpenAI compatible endpoints."""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        dimensions: int,
        api_version: str | None = None,
        api_key_header: str = "Authorization",
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("endpoint must not be empty")
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if dimensions < 1:
            raise ValueError("dimensions must be at least 1")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.api_version = api_version
        self.api_key_header = api_key_header
        self.timeout = timeout
        self._client = client

    async def embed(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")

        headers = self._headers()
        params = {"api-version": self.api_version} if self.api_version else None
        payload: dict[str, Any] = {"input": list(texts), "model": self.model}
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        owns_client = self._client is None

        try:
            response = await client.post(self.endpoint, headers=headers, params=params, json=payload)
            response.raise_for_status()
            body = response.json()
        finally:
            if owns_client:
                await client.aclose()

        return self._parse_response(body, expected=len(texts))

    def _headers(self) -> Mapping[str, str]:
        if self.api_key_header.casefold() == "authorization":
            return {self.api_key_header: f"Bearer {self.api_key}"}
        return {self.api_key_header: self.api_key}

    def _parse_response(self, body: Any, *, expected: int) -> list[EmbeddingVector]:
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise RuntimeError("embedding response must contain a data list")

        ordered = sorted(body["data"], key=lambda item: item.get("index", -1))
        if len(ordered) != expected:
            raise RuntimeError("embedding response cardinality does not match input")

        vectors: list[EmbeddingVector] = []
        for index, item in enumerate(ordered):
            if not isinstance(item, dict) or item.get("index") != index:
                raise RuntimeError("embedding response indices must be contiguous and ordered")
            values = item.get("embedding")
            if not isinstance(values, list):
                raise RuntimeError("embedding response item must contain an embedding list")
            try:
                vector_values = tuple(float(value) for value in values)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("embedding values must be numeric") from exc
            vectors.append(
                EmbeddingVector(
                    values=vector_values,
                    model=self.model,
                    dimensions=self.dimensions,
                )
            )
        return vectors
