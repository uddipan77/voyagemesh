"""A2A protocol invariants and trace-context propagation.

The artifact validation tests matter most: they encode the rule that a failed task can
never carry a result. That is the structural barrier preventing an agent error from being
silently converted into plausible-looking data in the final plan.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from vm_contracts import (
    AgentArtifact,
    AgentCard,
    AgentSkill,
    AgentTask,
    TaskError,
    TaskStatus,
    TraceContext,
    utc_now,
)
from vm_contracts.a2a import AgentCapabilities

pytestmark = pytest.mark.unit


def _skill(name: str = "search_transport") -> AgentSkill:
    return AgentSkill(
        name=name,
        description="Search transport options between two cities.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )


def _card(**overrides) -> AgentCard:
    defaults = {
        "name": "transport-agent",
        "display_name": "Transport Agent",
        "description": "Finds and ranks transport options.",
        "version": "0.1.0",
        "url": "http://transport-agent:8010",
        "skills": [_skill()],
    }
    return AgentCard(**{**defaults, **overrides})


def _task(**overrides) -> AgentTask:
    defaults = {
        "skill": "search_transport",
        "payload": {"origin": "Nuremberg"},
        "correlation_id": "corr-1",
        "request_id": "req-1",
    }
    return AgentTask(**{**defaults, **overrides})


def _artifact(task: AgentTask, **overrides) -> AgentArtifact:
    started = utc_now()
    defaults = {
        "task_id": task.task_id,
        "correlation_id": task.correlation_id,
        "agent_name": "transport-agent",
        "agent_version": "0.1.0",
        "skill": task.skill,
        "status": TaskStatus.COMPLETED,
        "result": {"offers": []},
        "started_at": started,
        "completed_at": started + timedelta(milliseconds=250),
        "duration_ms": 250,
    }
    return AgentArtifact(**{**defaults, **overrides})


class TestAgentCard:
    def test_skill_lookup(self):
        card = _card()
        assert card.has_skill("search_transport") is True
        assert card.has_skill("book_flight") is False
        assert card.skill("search_transport").name == "search_transport"

    def test_missing_skill_error_lists_what_is_offered(self):
        """Discovery failure should say what the agent *can* do, not just what it can't."""
        with pytest.raises(KeyError, match="search_transport"):
            _card().skill("book_flight")

    def test_rejects_duplicate_skills(self):
        with pytest.raises(ValueError, match="duplicate skill"):
            _card(skills=[_skill(), _skill()])

    def test_rejects_invalid_skill_name_format(self):
        with pytest.raises(ValueError):
            _skill("Search Transport")

    def test_card_requires_at_least_one_skill(self):
        with pytest.raises(ValueError):
            _card(skills=[])

    def test_mock_mode_is_advertised_on_the_card(self):
        """The UI's MOCKED badge is driven by the agent's own declaration."""
        card = _card(capabilities=AgentCapabilities(mock_mode=True))
        assert card.capabilities.mock_mode is True


class TestAgentTask:
    def test_generates_unique_ids(self):
        assert _task().task_id != _task().task_id

    def test_carries_correlation_and_trace(self):
        task = _task(trace_id="a" * 32, traceparent="00-" + "a" * 32 + "-" + "b" * 16 + "-01")
        assert task.correlation_id == "corr-1"
        assert task.traceparent is not None

    def test_does_not_accept_a_user_access_token(self):
        """Least privilege: only a user reference and roles cross the boundary."""
        with pytest.raises(ValueError):
            _task(access_token="eyJhbGciOi...")


class TestAgentArtifact:
    def test_matches_its_task(self):
        task = _task()
        assert _artifact(task).matches(task) is True

    @pytest.mark.parametrize(
        "tamper",
        [
            {"task_id": "different-task"},
            {"correlation_id": "different-correlation"},
            {"skill": "different_skill"},
        ],
    )
    def test_rejects_artifact_for_a_different_task(self, tamper):
        """Guards threat T-4: a response must answer the request that was actually sent."""
        task = _task()
        assert _artifact(task, **tamper).matches(task) is False

    def test_failed_task_must_not_carry_a_result(self):
        """The single most important invariant here — see module docstring."""
        task = _task()
        with pytest.raises(ValueError, match="must not carry a result"):
            _artifact(
                task,
                status=TaskStatus.FAILED,
                error=TaskError(code="upstream_timeout", message="Provider timed out"),
                result={"offers": [{"fabricated": True}]},
            )

    def test_failed_task_requires_an_error(self):
        task = _task()
        with pytest.raises(ValueError, match="requires an error"):
            _artifact(task, status=TaskStatus.FAILED, result=None)

    def test_completed_task_requires_a_result(self):
        task = _task()
        with pytest.raises(ValueError, match="usable result"):
            _artifact(task, status=TaskStatus.COMPLETED, result=None)

    def test_partial_must_explain_what_is_missing(self):
        task = _task()
        with pytest.raises(ValueError, match="at least one warning"):
            _artifact(task, status=TaskStatus.PARTIAL, warnings=[])

    def test_partial_is_usable_but_flagged(self):
        task = _task()
        artifact = _artifact(
            task, status=TaskStatus.PARTIAL, warnings=["weather forecast unavailable"]
        )
        assert artifact.is_usable is True
        assert artifact.warnings

    def test_rejects_completion_before_start(self):
        task = _task()
        started = utc_now()
        with pytest.raises(ValueError, match="precedes started_at"):
            _artifact(task, started_at=started, completed_at=started - timedelta(seconds=1))

    @pytest.mark.parametrize(
        ("status", "usable", "terminal"),
        [
            (TaskStatus.PENDING, False, False),
            (TaskStatus.RUNNING, False, False),
            (TaskStatus.COMPLETED, True, True),
            (TaskStatus.PARTIAL, True, True),
            (TaskStatus.FAILED, False, True),
            (TaskStatus.REJECTED, False, True),
            (TaskStatus.TIMEOUT, False, True),
        ],
    )
    def test_status_semantics(self, status, usable, terminal):
        assert status.is_usable is usable
        assert status.is_terminal is terminal


class TestTraceContext:
    def test_round_trips_through_traceparent(self):
        original = TraceContext.new()
        restored = TraceContext.from_traceparent(original.to_traceparent())
        assert restored.trace_id == original.trace_id
        assert restored.span_id == original.span_id

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            "",
            "garbage",
            "00-tooshort-abcdef1234567890-01",
            "99-" + "a" * 32 + "-" + "b" * 16 + "-01",  # unsupported version
            "00-" + "0" * 32 + "-" + "b" * 16 + "-01",  # all-zero trace id
            "00-" + "a" * 32 + "-" + "0" * 16 + "-01",  # all-zero span id
        ],
    )
    def test_malformed_traceparent_starts_a_new_trace_rather_than_failing(self, bad):
        """Broken upstream telemetry must degrade observability, never the user's request."""
        context = TraceContext.from_traceparent(bad)
        assert len(context.trace_id) == 32
        assert context.trace_id != "0" * 32

    def test_child_keeps_trace_and_changes_span(self):
        parent = TraceContext.new()
        child = parent.child(agent_name="stay-agent", agent_task_id="task_1")
        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        assert child.agent_name == "stay-agent"

    def test_headers_include_propagation_fields(self):
        context = TraceContext.new().model_copy(
            update={"correlation_id": "corr-9", "trip_id": "trip-9"}
        )
        headers = context.to_headers()
        assert headers["traceparent"] == context.to_traceparent()
        assert headers["x-correlation-id"] == "corr-9"
        assert headers["x-trip-id"] == "trip-9"

    def test_log_fields_exclude_absent_values(self):
        fields = TraceContext.new().log_fields()
        assert set(fields) == {"request_id", "trace_id", "span_id"}

    def test_sampling_flag(self):
        assert TraceContext.new(sampled=True).is_sampled is True
        assert TraceContext.new(sampled=False).is_sampled is False

    def test_rejects_all_zero_ids(self):
        with pytest.raises(ValueError, match="all zeros"):
            TraceContext(trace_id="0" * 32, span_id="a" * 16)
