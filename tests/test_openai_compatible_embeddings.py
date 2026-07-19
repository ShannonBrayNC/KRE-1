from __future__ import annotations

import httpx
import pytest

from kre.embeddings import OpenAICompatibleEmbeddingProvider


@pytest.mark.asyncio
async def test_openai_provider_preserves_input_order_and_auth() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        assert request.url.params.get("api-version") is None
        payload = request.read().decode()
        assert '"model":"text-embedding-3-small"' in payload
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleEmbeddingProvider(
            endpoint="https://api.openai.test/v1/embeddings",
            api_key="secret",
            model="text-embedding-3-small",
            dimensions=2,
            client=client,
        )
        vectors = await provider.embed(["first", "second"])

    assert [vector.values for vector in vectors] == [(1.0, 0.0), (0.0, 1.0)]
    assert all(vector.model == "text-embedding-3-small" for vector in vectors)


@pytest.mark.asyncio
async def test_azure_provider_uses_api_key_header_and_version() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["api-key"] == "azure-secret"
        assert request.url.params["api-version"] == "2024-02-01"
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5]}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleEmbeddingProvider(
            endpoint="https://example.openai.azure.com/openai/deployments/embed/embeddings",
            api_key="azure-secret",
            api_key_header="api-key",
            api_version="2024-02-01",
            model="embed-deployment",
            dimensions=1,
            client=client,
        )
        [vector] = await provider.embed(["knowledge"])

    assert vector.values == (0.5,)


@pytest.mark.asyncio
async def test_provider_validates_inputs_and_response_shape() -> None:
    provider = OpenAICompatibleEmbeddingProvider(
        endpoint="https://api.test/embeddings",
        api_key="secret",
        model="embed",
        dimensions=2,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"data": [{"index": 0, "embedding": [1.0]}]},
                )
            )
        ),
    )
    try:
        assert await provider.embed([]) == []
        with pytest.raises(ValueError, match="non-empty"):
            await provider.embed([" "])
        with pytest.raises(ValueError, match="length"):
            await provider.embed(["knowledge"])
    finally:
        await provider._client.aclose()  # type: ignore[union-attr]


def test_provider_validates_configuration() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        OpenAICompatibleEmbeddingProvider(
            endpoint=" ", api_key="secret", model="embed", dimensions=2
        )
    with pytest.raises(ValueError, match="dimensions"):
        OpenAICompatibleEmbeddingProvider(
            endpoint="https://api.test", api_key="secret", model="embed", dimensions=0
        )
