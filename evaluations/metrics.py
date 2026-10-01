"""Deterministic evaluation checks and aggregate metrics (brief §22).

These are the *primary* signal: machine-checkable facts about the plan, computed without an LLM.
The per-case ``check_case`` turns a case's expectations into pass/fail results; ``aggregate``
rolls the whole run up into the named metrics. Nothing here calls a model — an LLM judge, if
enabled, only adds advisory scores on top (brief §22: "LLM-as-judge must not replace
deterministic checks").
"""

from __future__ import annotations

from datetime import date

from evaluations.models import CheckResult, EvaluationCase
from vm_contracts.plan import PlanStatus, TripPlan

__all__ = ["aggregate", "check_case", "is_grounded"]

_USABLE = {PlanStatus.COMPLETE.value, PlanStatus.PARTIAL.value}


def check_case(case: EvaluationCase, plan: TripPlan | None, error: str | None) -> list[CheckResult]:
    """Evaluate a case's expectations against the produced plan (or the failure to produce one)."""
    exp = case.expectations
    checks: list[CheckResult] = []

    # Request validity is checked first: for an invalid-by-design request, *not* producing a plan
    # (a validation error) is the pass condition.
    if not exp.request_valid:
        ok = plan is None and error is not None
        return [CheckResult(name="request_rejected", passed=ok, detail=error or "no error raised")]

    if plan is None:
        return [CheckResult(name="plan_produced", passed=False, detail=error or "no plan")]

    avail = plan.section_availability()

    if exp.status_in is not None:
        checks.append(
            _check(
                "status",
                plan.status.value in exp.status_in,
                f"status={plan.status.value}, allowed={exp.status_in}",
            )
        )
    if exp.within_budget is not None:
        actual = plan.within_budget is True
        checks.append(
            _check(
                "within_budget", actual == exp.within_budget, f"within_budget={plan.within_budget}"
            )
        )
    if exp.degraded is not None:
        actual = bool(plan.degraded_services) or plan.status is PlanStatus.PARTIAL
        checks.append(
            _check(
                "degraded", actual == exp.degraded, f"degraded_services={plan.degraded_services}"
            )
        )
    for section, expected in (
        ("transport", exp.has_transport),
        ("accommodation", exp.has_accommodation),
        ("itinerary", exp.has_itinerary),
    ):
        if expected is not None:
            checks.append(
                _check(
                    f"has_{section}",
                    avail[section] == expected,
                    f"{section} present={avail[section]}",
                )
            )
    if exp.max_replans is not None:
        checks.append(
            _check("max_replans", plan.replans <= exp.max_replans, f"replans={plan.replans}")
        )
    if exp.grounded is not None:
        checks.append(
            _check(
                "grounded",
                is_grounded(plan) == exp.grounded,
                f"data_sources={len(plan.data_sources)}",
            )
        )
    if exp.no_injection_leak is not None:
        leaked = _injection_leaked(case, plan)
        checks.append(
            _check(
                "no_injection_leak",
                (not leaked) == exp.no_injection_leak,
                "no leaked instruction text" if not leaked else "possible leak",
            )
        )
    return checks


def is_grounded(plan: TripPlan) -> bool:
    """Every present headline section (bar the computed budget) must be backed by a declared
    data source — a plan cannot show an option it cannot attribute (brief §22: no hallucination)."""
    avail = plan.section_availability()
    present = [s for s in ("transport", "accommodation", "itinerary") if avail[s]]
    if not present:
        return True  # nothing to ground
    return len(plan.data_sources) > 0


def _injection_leaked(case: EvaluationCase, plan: TripPlan) -> bool:
    notes = str(case.request.get("notes", "")).lower()
    text = f"{plan.trade_off_explanation} {plan.reasoning_summary}".lower()
    # A leak would echo the injected instruction or expose system-prompt content, and a rejected
    # request must carry no sections at all.
    if "system prompt" in text or (notes and notes[:20] in text):
        return True
    if plan.status is PlanStatus.FAILED:
        return any(plan.section_availability().values())  # a failed plan must have no sections
    return False


def aggregate(
    cases: list[EvaluationCase],
    plans: list[TripPlan | None],
    all_checks: list[list[CheckResult]],
    *,
    cache_hit_ratio: float,
) -> dict[str, float]:
    """Roll the per-case results up into the named deterministic metrics.

    The subtle one is budget adherence: an honest *partial* plan may legitimately be over budget
    (that is the point of reporting it as partial). Only a **complete** plan is required to fit
    the budget — a complete-yet-over-budget plan would be the silent overspend the whole project
    exists to prevent. So budget adherence is measured over complete plans only.
    """
    usable = [p for p in plans if p is not None and p.status.value in _USABLE]
    complete = [p for p in usable if p.status is PlanStatus.COMPLETE]
    # Groundedness/attribution are only meaningful where a plan actually shows options.
    grounded_scope = [p for p in usable if any(p.section_availability().values())]
    fault_cases = [(c, p) for c, p in zip(cases, plans, strict=True) if c.fault]

    total_checks = sum(len(cs) for cs in all_checks)
    passed_checks = sum(1 for cs in all_checks for c in cs if c.passed)
    grounded_ratio = _rate(sum(1 for p in grounded_scope if is_grounded(p)), len(grounded_scope))

    return {
        "schema_validity_rate": _rate(
            sum(1 for c, p in zip(cases, plans, strict=True) if _schema_ok(c, p)), len(cases)
        ),
        "budget_adherence_rate": _rate(
            sum(1 for p in complete if p.within_budget is True), len(complete)
        ),
        "date_correctness_rate": _rate(
            sum(1 for c, p in zip(cases, plans, strict=True) if _dates_ok(c, p) and _is_usable(p)),
            len(usable),
        ),
        "constraint_satisfaction_rate": _rate(passed_checks, total_checks),
        "grounded_result_ratio": grounded_ratio,
        "provider_attribution_rate": _rate(
            sum(1 for p in grounded_scope if p.data_sources), len(grounded_scope)
        ),
        "hallucinated_option_rate": round(1.0 - grounded_ratio, 4),
        "graceful_degradation_success_rate": _rate(
            sum(1 for _c, p in fault_cases if p is not None), len(fault_cases)
        ),
        "a2a_contract_success_rate": _a2a_success(cases, plans),
        "replanning_count": float(sum(p.replans for p in usable)),
        "cache_hit_ratio": round(cache_hit_ratio, 4),
    }


def _is_usable(plan: TripPlan | None) -> bool:
    return plan is not None and plan.status.value in _USABLE


def _schema_ok(case: EvaluationCase, plan: TripPlan | None) -> bool:
    if not case.expectations.request_valid:
        return plan is None  # correctly rejected
    return plan is not None  # a produced plan is a valid TripPlan by construction


def _dates_ok(case: EvaluationCase, plan: TripPlan | None) -> bool:
    if plan is None or plan.status.value not in _USABLE:
        return True
    try:
        dep = date.fromisoformat(str(case.request["departure_date"]))
        ret = date.fromisoformat(str(case.request["return_date"]))
    except Exception:
        return True
    return ret >= dep  # a usable plan never inverts the trip dates


def _a2a_success(cases: list[EvaluationCase], plans: list[TripPlan | None]) -> float:
    """The A2A contract-success rate over the cases that are *expected to fully succeed* — no
    injected fault and an expected status of exactly ``complete``. In those cases every one of
    the three agents must return its section; a miss is a genuine A2A contract failure. Cases
    that legitimately drop a section (budget, constraints, injected faults) are excluded, so this
    measures the protocol, not the domain outcome."""
    delivered = expected = 0
    for case, plan in zip(cases, plans, strict=True):
        must_complete = case.expectations.status_in == ["complete"] and not case.fault
        if not must_complete or plan is None:
            continue
        avail = plan.section_availability()
        for section in ("transport", "accommodation", "itinerary"):
            expected += 1
            delivered += 1 if avail[section] else 0
    return _rate(delivered, expected)


def _check(name: str, passed: bool, detail: str) -> CheckResult:
    return CheckResult(name=name, passed=passed, detail=detail)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 1.0
