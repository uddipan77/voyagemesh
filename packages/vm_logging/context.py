"""Correlation context for structured logs.

A request's correlation fields (request_id, trace_id, trip_id, agent, …) are bound once — at
the edge, when the request arrives — and then flow into *every* log line emitted while handling
it, without any call site having to pass them. This is what makes a JSON log searchable: filter
on one ``request_id`` and you see the whole request across the gateway, the graph, and the
agents.

Implemented with :mod:`contextvars`, so it is coroutine-safe: two requests handled concurrently
never see each other's context.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

__all__ = ["bind_log_context", "clear_log_context", "get_log_context", "log_context"]

# Default is None (not a mutable dict shared across contexts); readers normalise to {}.
_CONTEXT: ContextVar[dict[str, str] | None] = ContextVar("vm_log_context", default=None)


def get_log_context() -> dict[str, str]:
    """The correlation fields currently in scope (a copy — callers must not mutate)."""
    return dict(_CONTEXT.get() or {})


def bind_log_context(**fields: str | None) -> None:
    """Merge ``fields`` into the current context. ``None`` values are ignored."""
    current = dict(_CONTEXT.get() or {})
    for key, value in fields.items():
        if value is not None:
            current[key] = str(value)
    _CONTEXT.set(current)


def clear_log_context() -> None:
    _CONTEXT.set({})


@contextlib.contextmanager
def log_context(**fields: str | None) -> Iterator[None]:
    """Bind ``fields`` for the duration of the block, restoring the prior context on exit."""
    token = _CONTEXT.set(
        {**(_CONTEXT.get() or {}), **{k: str(v) for k, v in fields.items() if v is not None}}
    )
    try:
        yield
    finally:
        _CONTEXT.reset(token)
