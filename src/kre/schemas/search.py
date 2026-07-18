from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SearchMode(StrEnum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    """Provider-neutral request contract for KRE retrieval endpoints."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(min_length=1, max_length=2_000)
    mode: SearchMode = SearchMode.HYBRID
    limit: int = Field(default=10, ge=1, le=100)
    candidate_limit: int | None = Field(default=None, ge=1, le=500)
    minimum_semantic_score: float = Field(default=-1.0, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_candidate_limit(self) -> SearchRequest:
        if self.candidate_limit is not None and self.candidate_limit < self.limit:
            raise ValueError("candidate_limit must be at least limit")
        return self


class SearchChannelScore(BaseModel):
    """Observability metadata for one retrieval channel."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1)
    score: float


class SearchHit(BaseModel):
    """Citation-ready API projection of a ranked KRE result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: UUID
    chunk_id: UUID
    document_title: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    text: str = Field(min_length=1)
    score: float
    keyword: SearchChannelScore | None = None
    semantic: SearchChannelScore | None = None
    source_uri: str | None = None
    source_system: str | None = None
    source_version: str | None = None
    classification: str | None = None
    security_labels: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Stable response envelope for keyword, semantic, and hybrid retrieval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1)
    mode: SearchMode
    count: int = Field(ge=0)
    results: tuple[SearchHit, ...]

    @model_validator(mode="after")
    def validate_count(self) -> SearchResponse:
        if self.count != len(self.results):
            raise ValueError("count must match the number of results")
        return self
