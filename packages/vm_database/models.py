"""SQLAlchemy models.

The persistent record of the system: trip requests and their plans, agent executions, A2A
tasks, tool-call audits, feedback, evaluation cases and runs, prompt versions, and the RAG
corpus (brief §18).

Design notes:

* Timestamps are timezone-aware and server-defaulted, so a row's creation time does not
  depend on the application clock.
* Structured payloads (the trip request, the plan, tool arguments) are stored as ``JSONB``
  — queryable, and a faithful record without a table per shape.
* The RAG embedding column is a pgvector ``Vector``; its dimension comes from configuration
  so the mock and a future real embedding model can differ without a schema edit.
* No user PII beyond a privacy-safe reference is stored (threat model, §T-2 residual).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime

__all__ = [
    "AgentExecutionRow",
    "Base",
    "EvaluationCaseRow",
    "EvaluationRunRow",
    "FeedbackRow",
    "PromptVersionRow",
    "RagChunkRow",
    "RagDocumentRow",
    "ToolAuditRow",
    "TripRow",
    "make_embedding_column",
]

# JSONB on Postgres, JSON elsewhere (SQLite in the offline repository tests).
_JSON = JSON().with_variant(JSONB(), "postgresql")

# Default embedding width. Overridable per deployment via DB_EMBEDDING_DIMENSIONS; the
# migration reads the same setting so the column and the embedder always agree.
DEFAULT_EMBEDDING_DIM = 384


def make_embedding_column(dim: int = DEFAULT_EMBEDDING_DIM) -> Mapped[Any]:
    return mapped_column(Vector(dim), nullable=False)


class Base(DeclarativeBase):
    """Declarative base with common timestamp columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class TripRow(Base):
    """A trip request and, once planned, its result."""

    __tablename__ = "trips"

    trip_id: Mapped[str] = mapped_column(String(60), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(60), index=True, nullable=False)
    user_reference: Mapped[str | None] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(20), index=True, nullable=False)

    origin: Mapped[str] = mapped_column(String(80), nullable=False)
    destination: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    cache_key: Mapped[str | None] = mapped_column(String(80), index=True)

    request_payload: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False)
    plan_payload: Mapped[dict[str, Any] | None] = mapped_column(_JSON)

    replans: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer)

    executions: Mapped[list[AgentExecutionRow]] = relationship(
        back_populates="trip", cascade="all, delete-orphan"
    )
    feedback: Mapped[list[FeedbackRow]] = relationship(
        back_populates="trip", cascade="all, delete-orphan"
    )


class AgentExecutionRow(Base):
    """One A2A delegation to a specialist agent, with its honest accounting."""

    __tablename__ = "agent_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trip_id: Mapped[str] = mapped_column(
        ForeignKey("trips.trip_id", ondelete="CASCADE"), index=True, nullable=False
    )
    task_id: Mapped[str] = mapped_column(String(60), index=True, nullable=False)
    agent_name: Mapped[str] = mapped_column(String(80), nullable=False)
    skill: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)

    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tool_call_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    llm_call_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    used_mock_data: Mapped[bool] = mapped_column(default=False, nullable=False)
    action_summaries: Mapped[list[str]] = mapped_column(_JSON, default=list, nullable=False)

    trip: Mapped[TripRow] = relationship(back_populates="executions")


class ToolAuditRow(Base):
    """An MCP tool call, recorded for audit (brief §11, §18)."""

    __tablename__ = "tool_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trip_id: Mapped[str | None] = mapped_column(String(60), index=True)
    agent_name: Mapped[str] = mapped_column(String(80), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    arguments_redacted: Mapped[dict[str, Any]] = mapped_column(_JSON, default=dict, nullable=False)


class FeedbackRow(Base):
    """User feedback on a plan."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trip_id: Mapped[str] = mapped_column(
        ForeignKey("trips.trip_id", ondelete="CASCADE"), index=True, nullable=False
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)
    trip: Mapped[TripRow] = relationship(back_populates="feedback")


class PromptVersionRow(Base):
    """A versioned prompt, for LLMOps reproducibility (brief §22)."""

    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (Index("uq_prompt_name_version", "name", "version", unique=True),)


class EvaluationCaseRow(Base):
    """One scenario in the evaluation dataset (brief §22)."""

    __tablename__ = "evaluation_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_key: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    suite: Mapped[str] = mapped_column(String(40), index=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    request_payload: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False)
    expectations: Mapped[dict[str, Any]] = mapped_column(_JSON, default=dict, nullable=False)


class EvaluationRunRow(Base):
    """The result of running the evaluation suite (brief §22)."""

    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    suite: Mapped[str] = mapped_column(String(40), index=True, nullable=False)
    total: Mapped[int] = mapped_column(Integer, nullable=False)
    passed: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(_JSON, default=dict, nullable=False)
    git_ref: Mapped[str | None] = mapped_column(String(60))


class RagDocumentRow(Base):
    """A source document in the destination-knowledge corpus (brief §18 pgvector)."""

    __tablename__ = "rag_documents"

    document_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    destination: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    category: Mapped[str] = mapped_column(String(40), nullable=False)
    source_note: Mapped[str] = mapped_column(String(200), default="", nullable=False)

    chunks: Mapped[list[RagChunkRow]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class RagChunkRow(Base):
    """A retrievable chunk with its embedding.

    The embedding column is added dynamically by the migration/metadata setup so its width
    tracks configuration. See :func:`vm_database.rag.configure_embedding_dimension`.
    """

    __tablename__ = "rag_chunks"

    chunk_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("rag_documents.document_id", ondelete="CASCADE"), index=True, nullable=False
    )
    citation_id: Mapped[str] = mapped_column(String(40), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = make_embedding_column()

    document: Mapped[RagDocumentRow] = relationship(back_populates="chunks")
