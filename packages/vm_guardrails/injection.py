"""Prompt-injection detection.

This is **defence in depth, not the primary control** (threat T-1). The real protection is
that the LLM has no dangerous capability to hijack — it cannot construct a URL, emit SQL,
read a file, or set a price. Pattern detection here flags attempts so they are visible in
metrics and can be refused at the edge, but the system's safety does not depend on it.

Detection *flags*, it does not silently strip: a caught attempt increments a counter and can
reject the request, so the attempt is observable rather than quietly swallowed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["InjectionScan", "scan_for_injection"]

# Patterns that indicate an attempt to override instructions or exfiltrate. Deliberately
# conservative — these target imperative override phrasing, not ordinary travel text that
# happens to mention "ignore" or "system".
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
            r"(previous|prior|above|earlier|all)\b[^.\n]{0,20}\b(instruction|prompt|rule|context)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"\b(you are now|act as|pretend to be|from now on you)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_probe",
        re.compile(
            r"\b(system prompt|your instructions|reveal your|print your|repeat the above)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"\b(api[_\s-]?key|secret|password|token|environment variable|credentials?)\b"
            r"[^.\n]{0,30}\b(show|print|reveal|list|leak|send|output|give)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_injection",
        re.compile(
            r"(END[- ]?UNTRUSTED[- ]?DATA|```system|<\|im_start\|>|\[/?INST\])",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class InjectionScan:
    """The result of scanning a piece of text."""

    is_suspicious: bool
    categories: tuple[str, ...] = ()
    field: str | None = None

    @property
    def summary(self) -> str:
        if not self.is_suspicious:
            return "no injection patterns detected"
        where = f" in '{self.field}'" if self.field else ""
        return f"possible prompt injection{where}: {', '.join(self.categories)}"


def scan_for_injection(text: str, *, field: str | None = None) -> InjectionScan:
    """Scan ``text`` for prompt-injection patterns.

    Returns an :class:`InjectionScan`; the caller decides whether to reject (input guardrail)
    or merely flag (retrieved RAG content, which is wrapped as untrusted data anyway).
    """
    if not text or not text.strip():
        return InjectionScan(is_suspicious=False, field=field)

    hits = [name for name, pattern in _PATTERNS if pattern.search(text)]
    return InjectionScan(is_suspicious=bool(hits), categories=tuple(hits), field=field)
