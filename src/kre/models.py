from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


class Classification(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class Provenance(BaseModel):
    source_system: str
    source_uri: HttpUrl | None = None
    connector: str
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str
    source_version: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class KnowledgeDocument(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1)
    content: str
    mime_type: str = "text/plain"
    language: str = "en"
    owner: str | None = None
    classification: Classification = Classification.INTERNAL
    security_labels: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    modified_at: datetime | None = None
    provenance: Provenance


class KnowledgeChunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    sequence: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_count: int | None = Field(default=None, ge=0)
    section: str | None = None
    page: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
