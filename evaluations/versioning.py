"""Captures the versions that define a reproducible evaluation run (brief §22).

An evaluation number is only meaningful next to *what produced it*: the prompts, the workflow,
the provider adapter, the dataset, and the code revision. These are recorded on every run so a
regression can be attributed to a specific change.
"""

from __future__ import annotations

import subprocess  # nosec B404 — used only for `git rev-parse`, no user input

from vm_caching import WORKFLOW_VERSION

__all__ = ["capture_versions", "git_ref"]


def git_ref() -> str:
    try:
        out = subprocess.run(  # nosec B603 B607 — fixed args, no shell, no user input
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def capture_versions(dataset_version: str, *, provider: str) -> dict[str, str]:
    return {
        "workflow_version": WORKFLOW_VERSION,
        "dataset_version": dataset_version,
        "provider": provider,
        "git_ref": git_ref(),
    }
