"""RAG integration tests against real PostgreSQL + pgvector.

Skipped unless a pgvector database is reachable (``VM_TEST_DB_URL`` or the default test port).
These cover what SQLite cannot: cosine-distance retrieval, the vector column, and the
end-to-end ingest → retrieve pipeline.

    docker run -d --name vm-pg -e POSTGRES_PASSWORD=voyagemesh -e POSTGRES_USER=voyagemesh \
        -e POSTGRES_DB=voyagemesh -p 5433:5432 pgvector/pgvector:pg16
    uv run pytest -m requires_docker
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from vm_config.settings import DatabaseSettings
from vm_database import Database, HashingEmbedder, ingest_document, retrieve
from vm_database.corpus import ingest_corpus
from vm_database.models import Base

TEST_DB_URL = os.environ.get(
    "VM_TEST_DB_URL",
    "postgresql+asyncpg://voyagemesh:voyagemesh@localhost:5433/voyagemesh",
)


pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]


@pytest.fixture
async def database():
    db = Database(DatabaseSettings(_env_file=None, url=TEST_DB_URL))
    if not await db.health_check():
        await db.aclose()
        pytest.skip("no pgvector database reachable")
    async with db.engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield db
    finally:
        async with db.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await db.aclose()


class TestRagPipeline:
    async def test_ingest_and_retrieve(self, database):
        embedder = HashingEmbedder()
        async with database.session() as session:
            await ingest_document(
                session,
                document_id="prague-oldtown",
                title="Prague Old Town",
                destination="Prague",
                category="neighbourhood",
                embedder=embedder,
                text=(
                    "The Old Town Square is the historic heart of Prague. It is largely "
                    "pedestrianised and step-free. The astronomical clock draws crowds hourly."
                ),
            )
        async with database.session() as session:
            results = await retrieve(
                session,
                destination="Prague",
                query="step-free access in the old town",
                embedder=embedder,
            )
        assert results
        assert results[0].similarity > 0
        assert results[0].citation_id.startswith("[prague-oldtown")

    async def test_retrieval_is_scoped_to_the_destination(self, database):
        embedder = HashingEmbedder()
        async with database.session() as session:
            await ingest_document(
                session,
                document_id="prague-x",
                title="Prague",
                destination="Prague",
                category="general",
                embedder=embedder,
                text="Prague has a famous castle.",
            )
            await ingest_document(
                session,
                document_id="vienna-x",
                title="Vienna",
                destination="Vienna",
                category="general",
                embedder=embedder,
                text="Vienna has a famous palace.",
            )
        async with database.session() as session:
            results = await retrieve(
                session, destination="Vienna", query="famous", embedder=embedder
            )
        assert results
        assert all(r.destination == "vienna" for r in results)

    async def test_unknown_destination_returns_empty_not_fabricated(self, database):
        async with database.session() as session:
            results = await retrieve(
                session, destination="Atlantis", query="anything", embedder=HashingEmbedder()
            )
        assert results == []

    async def test_price_shaped_text_is_refused(self, database):
        async with database.session() as session:
            with pytest.raises(ValueError, match="live prices"):
                await ingest_document(
                    session,
                    document_id="bad",
                    title="Bad",
                    destination="Prague",
                    category="general",
                    embedder=HashingEmbedder(),
                    text="A great hostel costs €18 per night in the centre.",
                )

    async def test_ingest_is_idempotent(self, database):
        embedder = HashingEmbedder()
        for _ in range(3):
            async with database.session() as session:
                await ingest_document(
                    session,
                    document_id="doc",
                    title="Doc",
                    destination="Prague",
                    category="general",
                    embedder=embedder,
                    text="Prague is historic.",
                )
        from sqlalchemy import func

        from vm_database.models import RagDocumentRow

        async with database.session() as session:
            count = (
                await session.execute(func.count(RagDocumentRow.document_id).select())
            ).scalar_one()
        assert count == 1  # re-ingest replaced, did not duplicate

    async def test_ingest_the_curated_corpus(self, database):
        async with database.session() as session:
            total = await ingest_corpus(session, HashingEmbedder())
        assert total >= 5  # the corpus has several documents
        async with database.session() as session:
            results = await retrieve(
                session,
                destination="Prague",
                query="quiet neighbourhood to stay with cafes",
                embedder=HashingEmbedder(),
            )
        assert results


class TestDatabaseHealth:
    async def test_health_check_true_when_reachable(self, database):
        assert await database.health_check() is True

    async def test_repositories_persist_a_real_plan(self, database):
        from datetime import date

        from vm_contracts.common import Currency, Money
        from vm_contracts.plan import PlanStatus, TripPlan
        from vm_contracts.trip import TripRequest
        from vm_database.repositories import TripRepository

        request = TripRequest(
            origin="Nuremberg",
            destination="Prague",
            departure_date=date(2026, 8, 10),
            return_date=date(2026, 8, 13),
            max_budget=Money.of(350, Currency.EUR),
        )
        plan = TripPlan(request_id="req-1", trip_id="trip-real-1", status=PlanStatus.COMPLETE)
        async with database.session() as session:
            await TripRepository(session).save_plan(plan, request=request)
        async with database.session() as session:
            restored = await TripRepository(session).get_plan("trip-real-1")
        assert restored is not None
        assert restored.status is PlanStatus.COMPLETE
