from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from kre.models import KnowledgeDocument


class Connector(ABC):
    """Source-neutral contract for discovering and ingesting knowledge."""

    name: str

    @abstractmethod
    async def discover(self) -> AsyncIterator[str]:
        """Yield stable source identifiers available to the connector."""

    @abstractmethod
    async def fetch(self, source_id: str) -> KnowledgeDocument:
        """Normalize one source object into a canonical knowledge document."""

    async def health(self) -> dict[str, str]:
        return {"connector": self.name, "status": "ok"}
