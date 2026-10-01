"""Best-effort recovery of JSON from imperfect model output.

Even with JSON mode enabled, models occasionally wrap output in markdown fences, prepend
"Here is the JSON:", or truncate mid-object when they hit a token limit. Repairing those
cheaply is much better than burning a retry — but the repair is deliberately *narrow*.

Every transformation here is structural: strip surrounding prose, remove fences, close
unterminated strings and brackets. Nothing invents, renames, or reorders a value. If the
payload cannot be recovered structurally, the caller retries with the validation error fed
back to the model rather than guessing at intent.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["extract_json", "repair_json"]

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(raw: str) -> str | None:
    """Pull the most plausible JSON object or array out of ``raw``.

    Returns ``None`` when nothing JSON-shaped is present.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidate = fenced.group(1).strip()
        if candidate:
            text = candidate

    # Trim any prose surrounding the payload by locating the outermost bracket pair.
    start = _first_index(text, "{", "[")
    if start is None:
        return None

    end = _last_index(text, "}", "]")
    if end is None:
        # No closer at all: a truncated response. repair_json may legitimately close it.
        return text[start:]
    if end < start:
        # A closer *before* the opener ("}{") is structurally broken, not truncated.
        # Treating it as truncation let repair close the stray "{" into an empty object —
        # fabricating a value out of garbage, which is exactly what this module must not do.
        return None

    return text[start : end + 1]


def repair_json(raw: str) -> Any | None:
    """Parse ``raw`` as JSON, applying structural repairs if needed.

    Returns the parsed value, or ``None`` if it could not be recovered.
    """
    candidate = extract_json(raw)
    if candidate is None:
        return None

    for attempt in (candidate, _strip_trailing_commas(candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue

    closed = _close_open_structures(_strip_trailing_commas(candidate))
    if closed is not None:
        try:
            return json.loads(closed)
        except json.JSONDecodeError:
            return None
    return None


def _first_index(text: str, *chars: str) -> int | None:
    found = [text.find(char) for char in chars]
    positions = [index for index in found if index != -1]
    return min(positions) if positions else None


def _last_index(text: str, *chars: str) -> int | None:
    found = [text.rfind(char) for char in chars]
    positions = [index for index in found if index != -1]
    return max(positions) if positions else None


def _strip_trailing_commas(text: str) -> str:
    """Remove commas before a closing bracket — a very common model slip."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _close_open_structures(text: str) -> str | None:
    """Close brackets and quotes left open by a truncated response.

    Walks the text tracking string state and escapes so that a brace *inside* a string
    value is not mistaken for structure. Returns ``None`` if the text is not recoverable
    this way.
    """
    stack: list[str] = []
    in_string = False
    escaped = False

    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return None  # more closers than openers: not a truncation
            opener = stack.pop()
            if (opener, char) not in (("{", "}"), ("[", "]")):
                return None  # mismatched: structurally wrong, not merely truncated

    if not stack and not in_string:
        return text  # nothing to close; the failure was something else

    repaired = text
    if in_string:
        repaired += '"'

    # A truncation often ends mid-key ({"a": 1, "b") — drop the dangling fragment so the
    # result is valid rather than merely balanced.
    repaired = re.sub(r",\s*\"[^\"]*\"\s*:?\s*$", "", repaired)
    repaired = re.sub(r",\s*$", "", repaired)

    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"

    return repaired
