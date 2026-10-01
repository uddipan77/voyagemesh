"""Persistence: async SQLAlchemy models, repositories, and pgvector RAG.

Business logic depends on repositories, never on sessions (brief §18), so the persistence
layer can change without the domain layer noticing. The RAG corpus holds curated, slow-
changing destination knowledge only — never live prices or schedules (ADR-011).
"""

from vm_database.engine import Database
from vm_database.models import Base
from vm_database.rag import (
    Embedder,
    HashingEmbedder,
    RetrievedChunk,
    chunk_text,
    ingest_document,
    retrieve,
)
from vm_database.repositories import FeedbackRepository, TripRepository

__all__ = [
    "Base",
    "Database",
    "Embedder",
    "FeedbackRepository",
    "HashingEmbedder",
    "RetrievedChunk",
    "TripRepository",
    "chunk_text",
    "ingest_document",
    "retrieve",
]
