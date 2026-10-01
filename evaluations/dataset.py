"""Loads the versioned evaluation dataset from ``data/evaluation/cases.json``.

The dataset lives in Git as data, not code, so it can be versioned and diffed independently of
the framework that runs it (brief §22: "Version … evaluation datasets"). ``DATASET`` is the
parsed, validated form.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluations.models import EvaluationCase

__all__ = ["DATASET_PATH", "load_dataset"]

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "cases.json"


def load_dataset(path: Path | None = None) -> tuple[str, list[EvaluationCase]]:
    """Return ``(dataset_version, cases)``."""
    raw = json.loads((path or DATASET_PATH).read_text(encoding="utf-8"))
    version = str(raw.get("dataset_version", "0"))
    cases = [EvaluationCase.model_validate(c) for c in raw["cases"]]
    return version, cases
