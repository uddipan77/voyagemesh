"""Seed the pgvector RAG corpus into a running database.

The corpus (``data/rag/corpus.json``) is curated, slow-changing destination knowledge, so it is
loaded once at stack startup rather than at request time. ``ingest_corpus`` is idempotent — a
re-run refreshes the documents instead of duplicating them — which is what lets Compose run this
on every ``up`` without accumulating chunks.

    python -m vm_database.seed
"""

from __future__ import annotations

import asyncio
import sys

from vm_config.settings import Settings
from vm_database import Database, HashingEmbedder
from vm_database.corpus import ingest_corpus


async def _seed() -> int:
    settings = Settings()
    if not settings.database.enabled:
        print("seed: database disabled, nothing to do")
        return 0

    database = Database(settings.database)
    try:
        if not await database.health_check():
            print("seed: database unreachable", file=sys.stderr)
            return 1
        async with database.session() as session:
            chunks = await ingest_corpus(session, HashingEmbedder())
        print(f"seed: ingested {chunks} chunk(s) from the RAG corpus")
    finally:
        await database.aclose()
    return 0


def main() -> int:
    return asyncio.run(_seed())


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    raise SystemExit(main())
