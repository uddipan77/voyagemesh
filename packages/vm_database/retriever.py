"""A self-contained RAG retriever for the Destination MCP server.

Bundles a :class:`Database` and an :class:`Embedder` behind a single ``retrieve`` call, so
the MCP server holds one object and does not manage sessions. Construction is lazy and
failure-tolerant: if the database is disabled or unreachable, ``from_settings`` returns
``None`` and the knowledge tool degrades to an honest empty result.
"""

from __future__ import annotations

import logging

from vm_config.settings import Settings
from vm_database.engine import Database
from vm_database.rag import HashingEmbedder, RetrievedChunk, retrieve

__all__ = ["RagRetriever"]

logger = logging.getLogger(__name__)


class RagRetriever:
    """Retrieves curated destination knowledge over pgvector."""

    def __init__(self, database: Database, *, embedding_dim: int) -> None:
        self._db = database
        self._embedder = HashingEmbedder(dim=embedding_dim)

    @classmethod
    def from_settings(cls, settings: Settings) -> RagRetriever | None:
        """Build a retriever, or ``None`` when the database is not configured."""
        if not settings.database.enabled:
            return None
        try:
            database = Database(settings.database)
        except Exception:  # pragma: no cover - construction rarely fails
            logger.warning("rag_retriever_unavailable")
            return None
        return cls(database, embedding_dim=settings.database.embedding_dimensions)

    async def retrieve(
        self, *, destination: str, query: str, limit: int = 4
    ) -> list[RetrievedChunk]:
        async with self._db.session() as session:
            return await retrieve(
                session,
                destination=destination,
                query=query,
                embedder=self._embedder,
                limit=limit,
            )

    async def aclose(self) -> None:
        await self._db.aclose()
