from __future__ import annotations

import math

import pytest

from kre.embeddings import DeterministicEmbeddingProvider, EmbeddingVector


@pytest.mark.asyncio
async def test_deterministic_provider_preserves_order_and_is_repeatable() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=8)

    first = await provider.embed(["alpha", "beta", "alpha"])
    second = await provider.embed(["alpha", "beta", "alpha"])

    assert first == second
    assert first[0] == first[2]
    assert first[0] != first[1]
    assert all(vector.dimensions == 8 for vector in first)
    assert all(vector.model == "sha256-v1" for vector in first)


@pytest.mark.asyncio
async def test_deterministic_vectors_are_unit_normalized() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=16)
    [vector] = await provider.embed(["governed knowledge"])

    magnitude = math.sqrt(sum(value * value for value in vector.values))

    assert magnitude == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_provider_supports_empty_batch_and_rejects_non_strings() -> None:
    provider = DeterministicEmbeddingProvider()

    assert await provider.embed([]) == []

    with pytest.raises(TypeError, match="string"):
        await provider.embed(["valid", 42])  # type: ignore[list-item]


def test_embedding_vector_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="match"):
        EmbeddingVector(values=(0.1,), model="test", dimensions=2)

    with pytest.raises(ValueError, match="at least"):
        EmbeddingVector(values=(), model="test", dimensions=0)


def test_deterministic_provider_validates_dimension_range() -> None:
    with pytest.raises(ValueError, match="between"):
        DeterministicEmbeddingProvider(dimensions=0)

    with pytest.raises(ValueError, match="between"):
        DeterministicEmbeddingProvider(dimensions=33)
