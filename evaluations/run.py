"""Evaluation CLI (brief §22).

    uv run python -m evaluations.run --suite regression

Runs the suite offline, prints the report, optionally stores it and/or writes it to a file, and
exits non-zero if a critical threshold regresses — which is exactly what lets CI gate on it.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from evaluations.report import to_console, to_json, to_markdown
from evaluations.runner import run_suite
from evaluations.thresholds import check_thresholds


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluations.run", description="Run the evaluation suite."
    )
    parser.add_argument("--suite", default="regression", help="Suite name to run.")
    parser.add_argument("--format", choices=["console", "json", "markdown"], default="console")
    parser.add_argument("--output", type=Path, help="Write the report to this file too.")
    parser.add_argument("--store", action="store_true", help="Persist the run to PostgreSQL.")
    parser.add_argument(
        "--no-gate", action="store_true", help="Report only; do not fail on threshold regressions."
    )
    return parser.parse_args(argv)


async def _main_async(argv: list[str] | None) -> int:
    args = _parse_args(argv)
    # The report is the output; the pipeline's INFO chatter is noise here.
    logging.getLogger().setLevel(logging.WARNING)
    report = await run_suite(args.suite)

    rendered = {"console": to_console, "json": to_json, "markdown": to_markdown}[args.format](
        report
    )
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    if args.store:
        from evaluations.storage import store_report
        from vm_config.settings import get_settings

        stored = await store_report(report, settings=get_settings())
        print(f"\n[stored to PostgreSQL: {stored}]")

    failures = [] if args.no_gate else check_thresholds(report)
    if failures:
        print("\nTHRESHOLD REGRESSIONS:", file=sys.stderr)
        for message in failures:
            print(f"  - {message}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
