"""Persists an evaluation run to PostgreSQL (brief §22). Best-effort and fail-open — a storage
outage must never fail the evaluation itself."""

from __future__ import annotations

import logging

from evaluations.models import SuiteReport
from vm_config.settings import Settings
from vm_database import Database
from vm_database.models import EvaluationRunRow

__all__ = ["store_report"]

logger = logging.getLogger(__name__)


async def store_report(report: SuiteReport, *, settings: Settings) -> bool:
    """Insert one ``evaluation_runs`` row. Returns True on success, False if the DB is
    unavailable. Never raises."""
    if not settings.database.enabled:
        return False
    database = Database(settings.database)
    try:
        async with database.session() as session:
            session.add(
                EvaluationRunRow(
                    suite=report.suite,
                    total=report.total,
                    passed=report.passed,
                    metrics={**report.metrics, **report.versions},
                    git_ref=report.versions.get("git_ref"),
                )
            )
        return True
    except Exception:
        logger.warning("evaluation_store_failed", exc_info=False)
        return False
    finally:
        await database.aclose()
