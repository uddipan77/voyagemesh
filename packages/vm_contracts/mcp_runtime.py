"""Shared runtime for the MCP servers.

Wraps FastMCP with the concerns every VoyageMesh tool server needs: argument validation
against a shared schema, uniform :class:`ToolResult` envelopes, error sanitisation, tool
budgets, and a health endpoint for Compose health checks.

Keeping this here rather than in each server means the three servers differ only in the
tools they expose — and a security fix applies to all of them at once.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette

from vm_contracts.mcp import ToolResult

__all__ = [
    "ManagedMCPServer",
    "ToolRegistry",
    "build_health_routes",
    "sanitise_error",
    "strip_computed_fields",
]

logger = logging.getLogger(__name__)


class ManagedMCPServer(FastMCP[Any]):
    """Own shared resources for the HTTP application's lifetime.

    FastMCP's tool-server lifespan runs once per request in stateless HTTP mode,
    including initialization and discovery. Closing a shared provider client there
    makes the first real tool call fail. The ASGI lifespan encloses all requests and
    closes resources only after the session manager has finished its tasks.
    """

    def __init__(
        self,
        *,
        app_lifespan: Callable[[Any], AbstractAsyncContextManager[dict[str, object]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._app_lifespan = app_lifespan

    def streamable_http_app(self) -> Starlette:
        app = super().streamable_http_app()
        session_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: Starlette) -> AsyncIterator[Any]:
            async with self._app_lifespan(self), session_lifespan(application) as state:
                yield state

        app.router.lifespan_context = lifespan
        return app


_MAX_ERROR_CHARS = 300

# Substrings that must never reach a tool caller. A provider error or a traceback can
# carry a connection string or an internal hostname; the tool boundary is where that stops.
_SENSITIVE_MARKERS = (
    "password",
    "secret",
    "api_key",
    "api-key",
    "authorization",
    "bearer ",
    "gsk_",
    "postgresql://",
    "redis://",
    "traceback",
)


def strip_computed_fields(payload: dict[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    """Recursively drop a model's computed-field keys from a serialised payload.

    ``model_dump()`` includes computed fields, but the contract models use
    ``extra="forbid"``, so feeding a dumped object straight back into ``model_validate``
    fails on exactly those keys. This is the round-trip an agent performs when it receives
    offers from a search tool and passes them to a scoring tool, so it must work.

    The recursion matters and was the fix for a real bug: a ``TransportOffer`` dump nests
    ``legs[].duration_minutes`` and an ``AccommodationOffer`` nests
    ``provenance.is_synthetic`` — computed fields one level down that a top-level strip
    misses, so re-validation still failed on them.

    Computed fields carry no independent information — they are re-derived on construction —
    so dropping them is lossless.
    """
    if not isinstance(payload, dict):
        return payload

    computed = set(getattr(model, "model_computed_fields", {}))
    cleaned: dict[str, Any] = {}
    fields = getattr(model, "model_fields", {})

    for key, value in payload.items():
        if key in computed:
            continue
        nested_model = _nested_model(fields.get(key))
        if nested_model is not None:
            cleaned[key] = _strip_within(value, nested_model)
        else:
            cleaned[key] = value
    return cleaned


def _strip_within(value: Any, model: type[BaseModel]) -> Any:
    if isinstance(value, dict):
        return strip_computed_fields(value, model)
    if isinstance(value, list):
        return [_strip_within(item, model) for item in value]
    return value


def _nested_model(field_info: Any) -> type[BaseModel] | None:
    """Return the BaseModel type inside a field annotation, unwrapping list/Optional.

    Returns ``None`` for scalar fields, which are left untouched.
    """
    if field_info is None:
        return None
    annotation = getattr(field_info, "annotation", None)
    for candidate in _iter_annotation_types(annotation):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _iter_annotation_types(annotation: Any) -> list[Any]:
    """Flatten ``list[X]``, ``X | None``, ``Optional[X]`` to the contained types."""
    import typing

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is None:
        return [annotation]
    flattened: list[Any] = []
    for arg in args:
        flattened.extend(_iter_annotation_types(arg))
    return flattened


def sanitise_error(exc: BaseException) -> str:
    """Render an exception as a message safe to return across the tool boundary.

    Errors whose text looks credential-bearing are replaced wholesale rather than redacted
    piecemeal — partial redaction of an unknown format is a guess, and the exception type
    alone is enough for an agent to decide what to do.
    """
    text = str(exc).strip() or type(exc).__name__
    lowered = text.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return f"{type(exc).__name__} (details withheld)"
    if len(text) > _MAX_ERROR_CHARS:
        text = text[:_MAX_ERROR_CHARS] + "…"
    return text


class ToolRegistry:
    """Registers validated, budget-limited tools on a FastMCP server."""

    def __init__(self, server_name: str, *, max_calls_per_session: int = 64) -> None:
        self.server_name = server_name
        self._max_calls = max_calls_per_session
        self._call_counts: dict[str, int] = {}
        self.tool_names: list[str] = []

    def register[A: BaseModel](
        self,
        server: Any,
        *,
        name: str,
        description: str,
        args_model: type[A],
        handler: Callable[[A], Awaitable[ToolResult]],
    ) -> None:
        """Register one read-only tool.

        The wrapper is where the guarantees live:

        * arguments are validated against ``args_model`` before the handler runs, so a
          handler never sees an out-of-range value;
        * a validation failure returns a structured ``ToolResult``, not an exception, so an
          agent's loop can correct itself;
        * unexpected exceptions are sanitised, so a provider's raw error cannot leak;
        * a per-tool call budget bounds runaway loops (brief §12).
        """
        self.tool_names.append(name)
        registry = self

        async def tool_entrypoint(**kwargs: Any) -> dict[str, Any]:
            started = time.perf_counter()

            count = registry._call_counts.get(name, 0) + 1
            registry._call_counts[name] = count
            if count > registry._max_calls:
                return ToolResult.failure(
                    code="tool_budget_exceeded",
                    message=(
                        f"'{name}' has been called {count} times, exceeding the per-session "
                        f"budget of {registry._max_calls}"
                    ),
                    source_name=registry.server_name,
                ).model_dump(mode="json")

            try:
                args = args_model.model_validate(kwargs)
            except ValidationError as exc:
                return ToolResult.failure(
                    code="invalid_arguments",
                    message=_describe_validation(exc),
                    source_name=registry.server_name,
                ).model_dump(mode="json")

            try:
                result = await handler(args)
            except Exception as exc:
                logger.warning(
                    "mcp_tool_failed",
                    extra={
                        "tool": name,
                        "server": registry.server_name,
                        "error_type": type(exc).__name__,
                    },
                )
                return ToolResult.failure(
                    code="tool_error",
                    message=sanitise_error(exc),
                    source_name=registry.server_name,
                    retryable=True,
                ).model_dump(mode="json")

            logger.info(
                "mcp_tool_called",
                extra={
                    "tool": name,
                    "server": registry.server_name,
                    "ok": result.ok,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            return result.model_dump(mode="json")

        # FastMCP builds a tool's advertised parameter schema by *inspecting the handler's
        # signature*, not its __annotations__. A `**kwargs` wrapper therefore advertises a
        # single opaque `kwargs` field, and every call fails validation before reaching us.
        #
        # Give the wrapper a real signature — one keyword-only parameter per model field —
        # so FastMCP sees the true argument schema. The parameters still funnel into the
        # same validate-then-dispatch body via the closure.
        parameters = []
        for field_name, field_info in args_model.model_fields.items():
            if field_info.is_required():
                default: Any = inspect.Parameter.empty
            elif field_info.default_factory is not None:
                # A field with a default_factory (list/dict fields) reports default=None
                # but is not required. Materialise the factory so the advertised default is
                # the real empty container, not a misleading None.
                default = field_info.default_factory()  # type: ignore[call-arg]
            else:
                default = field_info.default
            parameters.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=field_info.annotation,
                )
            )
        tool_entrypoint.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters, return_annotation=dict[str, Any]
        )
        tool_entrypoint.__name__ = name
        tool_entrypoint.__doc__ = description
        tool_entrypoint.__annotations__ = {
            field_name: field_info.annotation
            for field_name, field_info in args_model.model_fields.items()
        } | {"return": dict[str, Any]}

        server.tool(name=name, description=description)(tool_entrypoint)

    def call_count(self, name: str) -> int:
        return self._call_counts.get(name, 0)

    def reset_budgets(self) -> None:
        self._call_counts.clear()


def _describe_validation(exc: ValidationError) -> str:
    """Field locations and error types only — never the submitted values.

    An echoed value could carry user input straight into a log line, and the agent only
    needs to know *which* field was wrong to correct its next call.
    """
    problems = [
        f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['type']}"
        for error in exc.errors()[:8]
    ]
    return "invalid arguments — " + "; ".join(problems)


def build_health_routes(server: Any, *, service_name: str, version: str) -> None:
    """Attach ``/health/live`` and ``/health/ready`` to a FastMCP server.

    Compose depends on these rather than on startup order, so a dependent container waits
    for the server to actually be able to serve rather than merely to exist.
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @server.custom_route("/health/live", methods=["GET"])  # type: ignore[untyped-decorator]
    async def live(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "alive", "service": service_name, "version": version})

    @server.custom_route("/health/ready", methods=["GET"])  # type: ignore[untyped-decorator]
    async def ready(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ready", "service": service_name, "version": version})
