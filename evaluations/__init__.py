"""LLMOps evaluation framework (brief §22).

A repeatable, deterministic evaluation of the whole planning pipeline: a versioned dataset of
scenarios, machine-checkable expectations, aggregate quality metrics, an optional (advisory) LLM
judge, PostgreSQL storage, and a CI-gating regression command (``python -m evaluations.run``).
"""

from evaluations.models import CaseResult, EvaluationCase, Expectation, SuiteReport
from evaluations.runner import run_suite

__all__ = ["CaseResult", "EvaluationCase", "Expectation", "SuiteReport", "run_suite"]
