"""The evaluation suite as a CI gate (brief §22).

Runs the whole regression dataset through the real pipeline (offline, deterministic) and asserts
that every case passes and no critical metric has regressed. This is the test CI fails on when a
change quietly breaks budget adherence, groundedness, or graceful degradation.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from evaluations.metrics import check_case
from evaluations.models import EvaluationCase
from evaluations.runner import run_suite
from evaluations.thresholds import check_thresholds

pytestmark = pytest.mark.evaluation


@pytest.fixture(scope="module")
def report():
    # Run the whole suite once for the module. A plain sync fixture driving asyncio.run avoids a
    # module-scoped async fixture (which would clash with the function-scoped event loop) and
    # keeps the expensive full-pipeline run to a single execution.
    logging.disable(logging.CRITICAL)
    try:
        return asyncio.run(run_suite("regression"))
    finally:
        logging.disable(logging.NOTSET)


class TestRegressionSuite:
    async def test_every_case_passes(self, report):
        failed = [r.case_key for r in report.results if not r.passed]
        assert not failed, f"failing cases: {failed}"

    async def test_all_fourteen_scenarios_are_present(self, report):
        assert report.total == 14

    async def test_no_threshold_regressions(self, report):
        regressions = check_thresholds(report)
        assert not regressions, "\n".join(regressions)

    async def test_the_pipeline_never_crashed(self, report):
        crashed = [
            r.case_key
            for r in report.results
            if r.error and "Error:" in (r.error or "") and r.case_key != "invalid_dates"
        ]
        # invalid_dates is *expected* to raise a validation error; nothing else should error.
        assert not crashed, f"unexpected pipeline errors: {crashed}"


class TestMetricInvariants:
    async def test_no_usable_plan_exceeds_budget(self, report):
        assert report.metrics["budget_adherence_rate"] == 1.0

    async def test_nothing_is_hallucinated(self, report):
        assert report.metrics["hallucinated_option_rate"] == 0.0

    async def test_faults_degrade_gracefully(self, report):
        assert report.metrics["graceful_degradation_success_rate"] == 1.0

    async def test_a_repeated_request_hits_the_cache(self, report):
        assert report.metrics["cache_hit_ratio"] == 1.0


class TestCheckLogic:
    def test_an_invalid_request_expects_rejection(self):
        case = EvaluationCase(
            case_key="x",
            suite="s",
            description="d",
            request={},
            expectations={"request_valid": False},
        )
        checks = check_case(case, plan=None, error="ValidationError: bad")
        assert checks[0].name == "request_rejected"
        assert checks[0].passed
