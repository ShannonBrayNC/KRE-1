from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from kre.connectors.base import Connector
from kre.models import Classification, KnowledgeDocument, Provenance
from kre.normalization import normalize_text


class MarkdownConnector(Connector):
    """Ingest Markdown files from a bounded local directory."""

    name = "markdown"

    def __init__(
        self,
        root: Path,
        *,
        classification: Classification = Classification.INTERNAL,
    ) -> None:
        self.root = root.resolve()
        self.classification = classification

    async def discover(self) -> AsyncIterator[str]:
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob("*.md")):
            if path.is_file():
                yield path.relative_to(self.root).as_posix()

    async def fetch(self, source_id: str) -> KnowledgeDocument:
        path = (self.root / source_id).resolve()
        if self.root not in path.parents or path.suffix.lower() != ".md":
            raise ValueError("source_id must identify a Markdown file within the connector root")
        if not path.is_file():
            raise FileNotFoundError(source_id)

        normalized = normalize_text(path.read_text(encoding="utf-8-sig"))
        stat = path.stat()
        return KnowledgeDocument(
            title=path.stem.replace("-", " ").replace("_", " ").strip() or path.name,
            content=normalized.text,
            mime_type="text/markdown",
            classification=self.classification,
            modified_at=None,
            provenance=Provenance(
                source_system="filesystem",
                connector=self.name,
                content_hash=normalized.content_hash,
                source_version=str(stat.st_mtime_ns),
                attributes={"relative_path": source_id, "size_bytes": stat.st_size},
            ),
        )
