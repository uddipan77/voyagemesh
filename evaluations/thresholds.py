"""Regression thresholds — the contract CI enforces (brief §22).

A run fails if the pass rate drops or any critical metric regresses below its floor. These
encode the project's non-negotiable guarantees: every case passes, no usable plan exceeds its
budget, every option is grounded, and every fault degrades gracefully.
"""

from __future__ import annotations

from evaluations.models import SuiteReport

__all__ = ["THRESHOLDS", "check_thresholds"]

# metric name -> minimum acceptable value
THRESHOLDS: dict[str, float] = {
    "pass_rate": 1.0,  # nosec B105 — a metric threshold, not a password ("pass_rate" ≠ secret)
    "schema_validity_rate": 1.0,
    "budget_adherence_rate": 1.0,
    "grounded_result_ratio": 1.0,
    "provider_attribution_rate": 1.0,
    "graceful_degradation_success_rate": 1.0,
    "a2a_contract_success_rate": 1.0,
    "constraint_satisfaction_rate": 1.0,
}


def check_thresholds(report: SuiteReport) -> list[str]:
    """Return a list of human-readable regression messages; empty means the gate passes."""
    values = {"pass_rate": report.pass_rate, **report.metrics}
    failures: list[str] = []
    for metric, floor in THRESHOLDS.items():
        actual = values.get(metric)
        if actual is None:
            failures.append(f"{metric}: missing from report")
        elif actual < floor:
            failures.append(f"{metric}: {actual} < required {floor}")
    return failures
