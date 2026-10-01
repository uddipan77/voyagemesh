"""Runs the evaluation suite: every case through the real pipeline, then the metrics.

Fully offline and deterministic — the mock LLM and in-process agents mean a run on any machine
produces the same numbers, which is what makes a regression threshold meaningful.
"""

from __future__ import annotations

import logging

from evaluations.dataset import load_dataset
from evaluations.harness import build_orchestrator_for_fault, settings_for_fault
from evaluations.metrics import aggregate, check_case
from evaluations.models import CaseResult, EvaluationCase, SuiteReport
from evaluations.versioning import capture_versions
from vm_config.settings import Settings
from vm_contracts.plan import TripPlan
from vm_contracts.trip import TripRequest

__all__ = ["run_suite"]

logger = logging.getLogger(__name__)


def _base_settings() -> Settings:
    return Settings.for_testing(PROVIDER_MODE="mock", ENVIRONMENT="test", OTEL_ENABLED="false")


async def _run_case(case: EvaluationCase) -> tuple[TripPlan | None, str | None]:
    """Return ``(plan, error)`` for one case, applying its fault. Never raises."""
    try:
        request = TripRequest.model_validate(case.request)
    except Exception as exc:  # invalid-by-design request (e.g. inverted dates)
        return None, f"{type(exc).__name__}: {exc}"

    settings = settings_for_fault(_base_settings(), case.fault)
    try:
        orchestrator = build_orchestrator_for_fault(settings, case.fault)
        plan = await orchestrator.plan_trip(request)
        return plan, None
    except Exception as exc:  # the pipeline should never raise; record it as a hard failure
        logger.exception("evaluation_case_crashed", extra={"case": case.case_key})
        return None, f"{type(exc).__name__}: {exc}"


async def _measure_cache_hit_ratio() -> float:
    """A repeated identical request must hit the plan cache. Returns 1.0 on a hit, else 0.0."""
    _version, cases = load_dataset()
    normal = next((c for c in cases if c.case_key == "normal_budget_trip"), cases[0])
    request = TripRequest.model_validate(normal.request)

    from fakeredis.aioredis import FakeRedis

    from vm_caching import RedisClient, RedisPlanCache
    from vm_orchestrator import OrchestratorDependencies, TravelOrchestrator

    settings = _base_settings()
    cache = RedisPlanCache(RedisClient(settings.redis, redis=FakeRedis()), settings.cache_ttl)
    base = build_orchestrator_for_fault(settings, None)
    # Rebuild deps so both runs share the one cache.
    deps = OrchestratorDependencies(
        settings=settings,
        transport_client=base._deps.agent_client("transport"),
        stay_client=base._deps.agent_client("stay"),
        itinerary_client=base._deps.agent_client("itinerary"),
        cache=cache,
    )
    orchestrator = TravelOrchestrator(deps)
    await orchestrator.plan_trip(request)  # populate
    second = await orchestrator.plan_trip(request)  # should hit
    return 1.0 if second.cache_status == "hit" else 0.0


async def run_suite(suite: str = "regression") -> SuiteReport:
    dataset_version, all_cases = load_dataset()
    cases = [c for c in all_cases if c.suite == suite]

    results: list[CaseResult] = []
    plans: list[TripPlan | None] = []
    checks_per_case = []

    for case in cases:
        plan, error = await _run_case(case)
        checks = check_case(case, plan, error)
        plans.append(plan)
        checks_per_case.append(checks)
        results.append(
            CaseResult(
                case_key=case.case_key,
                suite=case.suite,
                passed=bool(checks) and all(c.passed for c in checks),
                checks=checks,
                status=plan.status.value if plan else "",
                replans=plan.replans if plan else 0,
                cache_status=plan.cache_status if plan else "miss",
                error=error,
            )
        )

    cache_hit_ratio = await _measure_cache_hit_ratio()
    metrics = aggregate(cases, plans, checks_per_case, cache_hit_ratio=cache_hit_ratio)

    return SuiteReport(
        suite=suite,
        dataset_version=dataset_version,
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        metrics=metrics,
        results=results,
        versions=capture_versions(dataset_version, provider="mock"),
    )
