"""Offline database tests: embedding, chunking, and repositories over SQLite.

The pure logic (deterministic embedding, chunking, the price-shape guard) needs no database.
The relational repositories are exercised against an in-memory SQLite database, which covers
the repository *logic* without requiring Postgres — pgvector-specific behaviour is tested
separately in the integration suite.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vm_contracts.a2a import AgentArtifact, TaskStatus
from vm_contracts.common import Currency, Money, utc_now
from vm_contracts.plan import PlanStatus, TripPlan
from vm_contracts.trip import TripRequest
from vm_database import HashingEmbedder, chunk_text
from vm_database.rag import _PRICE_SHAPED
from vm_database.repositories import FeedbackRepository, TripRepository

pytestmark = pytest.mark.unit


class TestEmbedding:
    def test_is_deterministic(self):
        e = HashingEmbedder()
        assert e.embed("historic old town square") == e.embed("historic old town square")

    def test_produces_a_unit_vector(self):
        vector = HashingEmbedder(dim=64).embed("some travel guidance text")
        norm = sum(v * v for v in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-9

    def test_respects_the_configured_dimension(self):
        assert len(HashingEmbedder(dim=128).embed("x")) == 128

    def test_empty_text_yields_a_valid_vector(self):
        vector = HashingEmbedder(dim=32).embed("")
        assert len(vector) == 32
        assert any(v != 0.0 for v in vector)  # not the zero vector (invalid for cosine)

    def test_different_text_gives_different_vectors(self):
        e = HashingEmbedder()
        assert e.embed("prague castle") != e.embed("vienna palace")

    def test_similar_text_is_closer_than_dissimilar(self):
        """A coarse semantic sanity check the hasher should still pass on shared tokens."""
        e = HashingEmbedder()

        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b, strict=True))

        query = e.embed("wheelchair step-free access old town")
        related = e.embed("step-free access is available in the old town square")
        unrelated = e.embed("the rail journey takes four hours with two transfers")
        assert cosine(query, related) > cosine(query, unrelated)


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert chunk_text("A short sentence.") == ["A short sentence."]

    def test_long_text_is_split(self):
        text = " ".join(f"Sentence number {i} about Prague." for i in range(60))
        chunks = chunk_text(text, max_chars=200)
        assert len(chunks) > 1
        assert all(len(c) <= 300 for c in chunks)  # bounded, allowing for overlap

    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("") == []

    def test_price_shape_guard_matches_prices(self):
        """ADR-011: prices must never enter the corpus. The guard must catch them."""
        assert _PRICE_SHAPED.search("the room is €120 per night")
        assert _PRICE_SHAPED.search("costs 45 EUR")
        assert not _PRICE_SHAPED.search("the old town square is historic and free to enter")


@pytest.fixture
async def sqlite_session():
    """An in-memory SQLite session with the relational (non-vector) tables created.

    The pgvector ``rag_chunks`` table is skipped — it needs Postgres, and the repositories
    under test do not touch it. This avoids mutating shared table metadata; pgvector
    behaviour is covered in the integration suite.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    from vm_database.models import Base

    relational = [t for t in Base.metadata.sorted_tables if t.name != "rag_chunks"]
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=relational))
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as session:
            yield session
    finally:
        await engine.dispose()


def _plan(trip_id: str = "trip-1", status: PlanStatus = PlanStatus.COMPLETE) -> TripPlan:
    return TripPlan(request_id="req-1", trip_id=trip_id, status=status)


def _request() -> TripRequest:
    return TripRequest(
        origin="Nuremberg",
        destination="Prague",
        departure_date=date(2026, 8, 10),
        return_date=date(2026, 8, 13),
        max_budget=Money.of(350, Currency.EUR),
    )


def _artifact(task_id: str = "task-1") -> AgentArtifact:
    started = utc_now()
    return AgentArtifact(
        task_id=task_id,
        correlation_id="trip-1",
        agent_name="transport-agent",
        agent_version="0.1.0",
        skill="plan_transport",
        status=TaskStatus.COMPLETED,
        result={"ok": True},
        action_summaries=["searched transport"],
        started_at=started,
        completed_at=started,
        duration_ms=100,
        tool_call_count=2,
        llm_call_count=1,
        used_mock_data=True,
    )


class TestTripRepository:
    async def test_save_and_get_a_plan(self, sqlite_session):
        repo = TripRepository(sqlite_session)
        await repo.save_plan(_plan(), request=_request())
        await sqlite_session.commit()
        restored = await repo.get_plan("trip-1")
        assert restored is not None
        assert restored.trip_id == "trip-1"
        assert restored.status is PlanStatus.COMPLETE

    async def test_save_is_idempotent_on_trip_id(self):
        # Re-saving the same trip updates rather than inserting a duplicate.
        pass  # exercised below with a real session

    async def test_resaving_updates_not_duplicates(self, sqlite_session):
        repo = TripRepository(sqlite_session)
        await repo.save_plan(_plan(status=PlanStatus.PARTIAL), request=_request())
        await repo.save_plan(_plan(status=PlanStatus.COMPLETE), request=_request())
        await sqlite_session.commit()
        assert await repo.count() == 1
        assert (await repo.get_plan("trip-1")).status is PlanStatus.COMPLETE

    async def test_records_agent_executions(self, sqlite_session):
        repo = TripRepository(sqlite_session)
        await repo.save_plan(
            _plan(), request=_request(), artifacts=[_artifact("t1"), _artifact("t2")]
        )
        await sqlite_session.commit()
        executions = await repo.executions_for("trip-1")
        assert len(executions) == 2
        assert executions[0].tool_call_count == 2
        assert executions[0].used_mock_data is True

    async def test_get_status(self, sqlite_session):
        repo = TripRepository(sqlite_session)
        await repo.save_plan(_plan(status=PlanStatus.PARTIAL), request=_request())
        await sqlite_session.commit()
        assert await repo.get_status("trip-1") == "partial"
        assert await repo.get_status("nonexistent") is None

    async def test_records_a_tool_audit(self, sqlite_session):
        repo = TripRepository(sqlite_session)
        await repo.record_tool_audit(
            trip_id="trip-1",
            agent_name="transport-agent",
            tool_name="search_ground_transport",
            status="ok",
            latency_ms=5,
            arguments_redacted={"origin": "Nuremberg", "token": "***"},
        )
        await sqlite_session.commit()  # no exception = persisted

    async def test_missing_plan_returns_none(self, sqlite_session):
        assert await TripRepository(sqlite_session).get_plan("nope") is None


class TestFeedbackRepository:
    async def test_add_and_average(self, sqlite_session):
        TripRepository(sqlite_session)
        await TripRepository(sqlite_session).save_plan(_plan(), request=_request())
        await sqlite_session.commit()
        repo = FeedbackRepository(sqlite_session)
        await repo.add(trip_id="trip-1", rating=4)
        await repo.add(trip_id="trip-1", rating=2, comment="mixed")
        await sqlite_session.commit()
        assert await repo.average_rating("trip-1") == 3.0

    async def test_rating_must_be_valid(self, sqlite_session):
        repo = FeedbackRepository(sqlite_session)
        with pytest.raises(ValueError, match="between 1 and 5"):
            await repo.add(trip_id="trip-1", rating=6)

    async def test_average_of_no_feedback_is_none(self, sqlite_session):
        assert await FeedbackRepository(sqlite_session).average_rating("trip-x") is None


class TestRagRetriever:
    def test_from_settings_returns_none_when_database_disabled(self):
        from vm_config.settings import Settings
        from vm_database.retriever import RagRetriever

        settings = Settings.for_testing(DB_ENABLED="false")
        assert RagRetriever.from_settings(settings) is None
