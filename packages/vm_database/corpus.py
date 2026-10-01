"""Loading and ingesting the curated RAG corpus.

The corpus lives in ``data/rag/corpus.json`` — author-written, slow-changing destination
knowledge (ADR-011). This module reads it and ingests it into pgvector, idempotently, so a
re-run refreshes rather than duplicates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from vm_database.rag import Embedder, ingest_document

__all__ = ["CORPUS_PATH", "ingest_corpus", "load_corpus"]

CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "rag" / "corpus.json"


def load_corpus(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the corpus documents. Returns an empty list if the file is missing."""
    source = path or CORPUS_PATH
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    documents = raw.get("documents", [])
    return documents if isinstance(documents, list) else []


async def ingest_corpus(
    session: AsyncSession, embedder: Embedder, *, path: Path | None = None
) -> int:
    """Ingest every corpus document. Returns the total number of chunks stored."""
    total = 0
    for doc in load_corpus(path):
        total += await ingest_document(
            session,
            document_id=str(doc["document_id"]),
            title=str(doc["title"]),
            destination=str(doc["destination"]),
            category=str(doc.get("category", "general")),
            text=str(doc["text"]),
            embedder=embedder,
            source_note=str(doc.get("source_note", "")),
        )
    return total
