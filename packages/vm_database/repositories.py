"""Repositories.

Business logic depends on these, never on a SQLAlchemy session (brief §18). Each repository
takes a session and exposes intention-revealing methods; the orchestrator persists a trip by
calling ``TripRepository.save_plan``, not by writing SQL.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vm_contracts.a2a import AgentArtifact
from vm_contracts.plan import TripPlan
from vm_contracts.trip import NormalizedTripRequest, TripRequest
from vm_database.models import (
    AgentExecutionRow,
    FeedbackRow,
    ToolAuditRow,
    TripRow,
)

__all__ = ["FeedbackRepository", "TripRepository"]


class TripRepository:
    """Persists trip requests, plans, and their agent executions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_plan(
        self,
        plan: TripPlan,
        *,
        request: TripRequest,
        normalized: NormalizedTripRequest | None = None,
        artifacts: list[AgentArtifact] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Upsert the trip row and record each agent execution.

        Idempotent on ``trip_id``: re-saving the same plan updates rather than duplicating.
        """
        row = await self._session.get(TripRow, plan.trip_id)
        payload_plan = plan.model_dump(mode="json")
        if row is None:
            row = TripRow(trip_id=plan.trip_id)
            self._session.add(row)

        row.request_id = plan.request_id
        row.status = plan.status.value
        row.origin = request.origin
        row.destination = request.destination
        row.cache_key = (
            "-".join(str(v) for v in normalized.cache_fingerprint().values())[:80]
            if normalized
            else None
        )
        row.request_payload = request.model_dump(mode="json")
        row.plan_payload = payload_plan
        row.replans = plan.replans
        row.duration_ms = duration_ms

        for artifact in artifacts or []:
            self._session.add(
                AgentExecutionRow(
                    trip_id=plan.trip_id,
                    task_id=artifact.task_id,
                    agent_name=artifact.agent_name,
                    skill=artifact.skill,
                    status=artifact.status.value,
                    duration_ms=artifact.duration_ms,
                    tool_call_count=artifact.tool_call_count,
                    llm_call_count=artifact.llm_call_count,
                    used_mock_data=artifact.used_mock_data,
                    action_summaries=list(artifact.action_summaries),
                )
            )
        await self._session.flush()

    async def get_plan(self, trip_id: str) -> TripPlan | None:
        row = await self._session.get(TripRow, trip_id)
        if row is None or row.plan_payload is None:
            return None
        return TripPlan.model_validate(row.plan_payload)

    async def get_status(self, trip_id: str) -> str | None:
        row = await self._session.get(TripRow, trip_id)
        return row.status if row is not None else None

    async def executions_for(self, trip_id: str) -> list[AgentExecutionRow]:
        stmt = (
            select(AgentExecutionRow)
            .where(AgentExecutionRow.trip_id == trip_id)
            .order_by(AgentExecutionRow.id)
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def record_tool_audit(
        self,
        *,
        trip_id: str | None,
        agent_name: str,
        tool_name: str,
        status: str,
        latency_ms: int,
        arguments_redacted: dict[str, Any],
    ) -> None:
        self._session.add(
            ToolAuditRow(
                trip_id=trip_id,
                agent_name=agent_name,
                tool_name=tool_name,
                status=status,
                latency_ms=latency_ms,
                arguments_redacted=arguments_redacted,
            )
        )
        await self._session.flush()

    async def count(self) -> int:
        return int((await self._session.execute(select(func.count(TripRow.trip_id)))).scalar_one())


class FeedbackRepository:
    """Persists user feedback on plans."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, *, trip_id: str, rating: int, comment: str | None = None) -> None:
        if not 1 <= rating <= 5:
            raise ValueError("rating must be between 1 and 5")
        self._session.add(FeedbackRow(trip_id=trip_id, rating=rating, comment=comment))
        await self._session.flush()

    async def average_rating(self, trip_id: str) -> float | None:
        stmt = select(func.avg(FeedbackRow.rating)).where(FeedbackRow.trip_id == trip_id)
        result = (await self._session.execute(stmt)).scalar_one_or_none()
        return float(result) if result is not None else None
