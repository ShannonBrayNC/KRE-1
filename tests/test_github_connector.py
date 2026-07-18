from __future__ import annotations

import base64

import httpx
import pytest

from kre.connectors.github import GitHubConnector
from kre.normalization import normalize_text


@pytest.mark.asyncio
async def test_github_connector_discovers_supported_text_files() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/Lantern/KRE/git/trees/main"
        return httpx.Response(
            200,
            json={
                "tree": [
                    {"path": "README.md", "type": "blob"},
                    {"path": "docs/architecture.rst", "type": "blob"},
                    {"path": "assets/logo.png", "type": "blob"},
                    {"path": "docs", "type": "tree"},
                ]
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.github.com",
    )
    connector = GitHubConnector("Lantern/KRE", client=client)

    discovered = [path async for path in connector.discover()]

    assert discovered == ["README.md", "docs/architecture.rst"]
    await client.aclose()


@pytest.mark.asyncio
async def test_github_connector_fetches_normalized_document_with_provenance() -> None:
    source = "# KRE\r\n\r\nGoverned knowledge.  \r\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/Lantern/KRE/contents/docs/overview.md"
        assert request.url.params["ref"] == "release"
        return httpx.Response(
            200,
            json={
                "type": "file",
                "sha": "abc123",
                "html_url": "https://github.com/Lantern/KRE/blob/release/docs/overview.md",
                "content": base64.b64encode(source.encode()).decode(),
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.github.com",
    )
    connector = GitHubConnector("Lantern/KRE", ref="release", client=client)

    document = await connector.fetch("docs/overview.md")
    expected = normalize_text(source)

    assert document.title == "overview"
    assert document.content == expected.text
    assert document.provenance.connector == "github"
    assert document.provenance.source_version == "abc123"
    assert document.provenance.content_hash == expected.content_hash
    assert document.provenance.attributes["repository"] == "Lantern/KRE"
    await client.aclose()


@pytest.mark.asyncio
async def test_github_connector_rejects_unsafe_or_unsupported_paths() -> None:
    client = httpx.AsyncClient(base_url="https://api.github.com")
    connector = GitHubConnector("Lantern/KRE", client=client)

    with pytest.raises(ValueError, match="safe repository-relative"):
        await connector.fetch("../secret.md")

    with pytest.raises(ValueError, match="unsupported"):
        await connector.fetch("logo.png")

    await client.aclose()


def test_github_connector_requires_owner_name() -> None:
    with pytest.raises(ValueError, match="owner/name"):
        GitHubConnector("invalid")