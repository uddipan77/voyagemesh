"""Renders a :class:`SuiteReport` for humans and for CI."""

from __future__ import annotations

import json

from evaluations.models import SuiteReport

__all__ = ["to_console", "to_json", "to_markdown"]


def to_console(report: SuiteReport) -> str:
    lines = [
        f"Evaluation suite: {report.suite}  (dataset {report.dataset_version}, "
        f"git {report.versions.get('git_ref', '?')})",
        f"Cases: {report.passed}/{report.total} passed  ({report.pass_rate:.0%})",
        "",
        "Cases:",
    ]
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        extra = f" status={r.status}" if r.status else ""
        lines.append(f"  [{mark}] {r.case_key}{extra}")
        for c in r.failed_checks:
            lines.append(f"         ✗ {c.name}: {c.detail}")
        if r.error and not r.passed:
            lines.append(f"         ! {r.error}")
    lines += ["", "Deterministic metrics:"]
    for key, value in sorted(report.metrics.items()):
        lines.append(f"  {key:<34} {value}")
    return "\n".join(lines)


def to_markdown(report: SuiteReport) -> str:
    out = [
        f"# Evaluation report — {report.suite}",
        "",
        f"- Dataset version: `{report.dataset_version}`",
        f"- Git ref: `{report.versions.get('git_ref', '?')}`",
        f"- Result: **{report.passed}/{report.total} passed** ({report.pass_rate:.0%})",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    out += [f"| {k} | {v} |" for k, v in sorted(report.metrics.items())]
    out += ["", "## Cases", "", "| Case | Result | Status | Notes |", "| --- | --- | --- | --- |"]
    for r in report.results:
        notes = "; ".join(f"{c.name}: {c.detail}" for c in r.failed_checks) or (r.error or "")
        out.append(f"| {r.case_key} | {'✅' if r.passed else '❌'} | {r.status} | {notes} |")
    return "\n".join(out) + "\n"


def to_json(report: SuiteReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2) + "\n"
