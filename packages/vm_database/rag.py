"""RAG: chunking, embedding, ingestion, and retrieval over pgvector.

The embedding model is a **deterministic local hasher**, not a hosted API. It maps text to a
fixed-dimension unit vector via hashed token n-grams. This is a clear, honest trade-off
(ADR-010 family): a real sentence-transformer would give better semantic recall, but it would
add a heavy dependency or a paid API, whereas the hasher is free, offline, deterministic, and
enough to demonstrate the pgvector pipeline end to end. It is documented as a stand-in
wherever it appears, and swapping in a real embedder is one class behind the
:class:`Embedder` protocol.

Retrieval treats returned text as **untrusted** — the caller wraps it before it enters any
prompt (threat T-1). Live prices and schedules are never embedded here (ADR-011); the corpus
is curated slow-changing knowledge only.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vm_database.models import RagChunkRow, RagDocumentRow

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "RetrievedChunk",
    "chunk_text",
    "ingest_document",
    "retrieve",
]

DEFAULT_DIM = 384
_TOKEN = re.compile(r"[a-z0-9]+")

# Refuse to embed anything price-like — a guard against a curator accidentally putting a live
# figure into the RAG corpus (ADR-011).
_PRICE_SHAPED = re.compile(r"[€£$]\s?\d|\b\d+\s?(eur|gbp|usd|per night|/night)\b", re.IGNORECASE)


class Embedder(Protocol):
    """Maps text to a fixed-dimension embedding vector."""

    dim: int

    def embed(self, text: str) -> list[float]:
        """Return a unit-length embedding of ``text``."""
        ...


class HashingEmbedder:
    """A deterministic, offline embedder using hashed token bigrams.

    Not semantically strong, but reproducible and free. Same text → same vector, on any
    machine, which is what keeps the RAG tests and the evaluation suite deterministic.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            # A zero vector is invalid for cosine distance; return a stable unit vector.
            vector[0] = 1.0
            return vector

        # Unigrams and bigrams, hashed into buckets with a signed contribution.
        grams = tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]
        for gram in grams:
            digest = hashlib.sha256(gram.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk retrieved from the corpus, with its similarity and citation."""

    chunk_id: str
    document_id: str
    title: str
    destination: str
    citation_id: str
    content: str
    similarity: float


def chunk_text(text: str, *, max_chars: int = 600, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks on sentence-ish boundaries.

    Overlap keeps a fact that straddles a boundary retrievable from either side. Chunks are
    bounded so a single embedding covers a coherent span rather than a whole document.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return [cleaned] if cleaned else []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = current[-overlap:] + " " + sentence if overlap else sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


async def ingest_document(
    session: AsyncSession,
    *,
    document_id: str,
    title: str,
    destination: str,
    category: str,
    text: str,
    embedder: Embedder,
    source_note: str = "",
) -> int:
    """Chunk, embed, and store a document. Returns the number of chunks stored.

    Raises ``ValueError`` if the text contains price-shaped content — live figures must never
    enter the corpus (ADR-011).
    """
    if _PRICE_SHAPED.search(text):
        raise ValueError(
            f"document '{document_id}' contains price-shaped text; live prices must not be "
            f"stored in RAG (ADR-011)"
        )

    # Replace any existing version of this document idempotently.
    existing = await session.get(RagDocumentRow, document_id)
    if existing is not None:
        await session.delete(existing)
        await session.flush()

    document = RagDocumentRow(
        document_id=document_id,
        title=title,
        destination=destination.strip().casefold(),
        category=category,
        source_note=source_note,
    )
    session.add(document)

    chunks = chunk_text(text)
    for index, chunk in enumerate(chunks):
        session.add(
            RagChunkRow(
                chunk_id=f"{document_id}#{index}",
                document_id=document_id,
                citation_id=f"[{document_id}:{index}]",
                content=chunk,
                embedding=embedder.embed(chunk),
            )
        )
    await session.flush()
    return len(chunks)


async def retrieve(
    session: AsyncSession,
    *,
    destination: str,
    query: str,
    embedder: Embedder,
    limit: int = 4,
) -> list[RetrievedChunk]:
    """Retrieve the most similar chunks for ``query`` within ``destination``.

    Uses pgvector cosine distance. Returns an empty list — never fabricated guidance — when
    the corpus has nothing for the destination.
    """
    query_vector = embedder.embed(query)
    distance = RagChunkRow.embedding.cosine_distance(query_vector)

    stmt = (
        select(
            RagChunkRow.chunk_id,
            RagChunkRow.document_id,
            RagDocumentRow.title,
            RagDocumentRow.destination,
            RagChunkRow.citation_id,
            RagChunkRow.content,
            distance.label("distance"),
        )
        .join(RagDocumentRow, RagChunkRow.document_id == RagDocumentRow.document_id)
        .where(RagDocumentRow.destination == destination.strip().casefold())
        .order_by(distance)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        RetrievedChunk(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            title=row.title,
            destination=row.destination,
            citation_id=row.citation_id,
            content=row.content,
            similarity=round(1.0 - float(row.distance), 4),
        )
        for row in rows
    ]
