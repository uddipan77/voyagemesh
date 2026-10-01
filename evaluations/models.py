"""Typed results for the evaluation framework (brief §22).

An evaluation is deterministic first: every case declares *expectations* — machine-checkable
facts about the plan it should produce — and the runner records a pass/fail per expectation plus
the aggregate metrics. The optional LLM judge (see :mod:`evaluations.judge`) adds quality scores
*on top of* these checks; it never replaces them.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from vm_contracts.common import StrictModel

__all__ = [
    "CaseResult",
    "CheckResult",
    "EvaluationCase",
    "Expectation",
    "JudgeScores",
    "SuiteReport",
]


class Expectation(StrictModel):
    """Machine-checkable expectations for a case. Every field is optional — a case asserts only
    what is meaningful for the scenario it exercises."""

    request_valid: bool = True
    """Whether the request should construct at all. False for the invalid-dates scenario."""

    status_in: list[str] | None = None
    within_budget: bool | None = None
    degraded: bool | None = None
    has_transport: bool | None = None
    has_accommodation: bool | None = None
    has_itinerary: bool | None = None
    max_replans: int | None = None
    no_injection_leak: bool | None = None
    grounded: bool | None = None
    """Every recommended option must trace to a declared data source (no hallucinated option)."""


class EvaluationCase(StrictModel):
    """One scenario: a request, an optional injected fault, and its expectations."""

    case_key: Annotated[str, Field(min_length=1, max_length=80)]
    suite: Annotated[str, Field(min_length=1, max_length=40)]
    description: Annotated[str, Field(min_length=1, max_length=300)]
    request: dict[str, Any]
    fault: str | None = None
    expectations: Expectation = Field(default_factory=Expectation)


class CheckResult(StrictModel):
    name: str
    passed: bool
    detail: str = ""


class JudgeScores(StrictModel):
    """Optional LLM-as-judge quality scores (0-5). Advisory only."""

    relevance: float = 0.0
    usefulness: float = 0.0
    coherence: float = 0.0
    preference_alignment: float = 0.0
    tradeoff_quality: float = 0.0
    comment: str = ""


class CaseResult(StrictModel):
    case_key: str
    suite: str
    passed: bool
    checks: list[CheckResult] = Field(default_factory=list)
    status: str = ""
    replans: int = 0
    cache_status: str = "miss"
    error: str | None = None
    judge: JudgeScores | None = None

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


class SuiteReport(StrictModel):
    suite: str
    dataset_version: str
    total: int
    passed: int
    metrics: dict[str, float] = Field(default_factory=dict)
    results: list[CaseResult] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return round(self.passed / self.total, 4) if self.total else 0.0
