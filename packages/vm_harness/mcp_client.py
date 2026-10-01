"""Tool-client abstraction for reaching MCP servers.

An agent depends on the :class:`ToolClient` protocol, never on a concrete transport. Two
implementations back it:

* :class:`InProcessToolClient` drives a FastMCP server object directly. Used by tests and by
  a single-process run, it needs no network and no running container.
* :class:`HttpMCPToolClient` connects to a remote MCP server over streamable-HTTP. Used when
  the agents and MCP servers run as separate containers.

Both return the same :class:`~vm_contracts.mcp.ToolResult`, so switching between them changes
one line of wiring and nothing in the agent (mirrors the A2A ``AgentClient`` seam — a
protocol-version change touches an adapter, not the domain layer).
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable

from vm_contracts.mcp import ToolResult

__all__ = [
    "HttpMCPToolClient",
    "InProcessToolClient",
    "ToolClient",
    "ToolClientError",
]

logger = logging.getLogger(__name__)


class ToolClientError(Exception):
    """A transport-level failure reaching an MCP server.

    Distinct from a tool *result* whose ``ok`` is ``False``: that is a normal outcome the
    agent reasons about, whereas this means the server could not be reached or spoke
    nonsense. The message is sanitised — no URL, no credential.
    """


@runtime_checkable
class ToolClient(Protocol):
    """A source of MCP tool calls."""

    async def call(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        """Invoke ``tool`` and return its structured result.

        Raises:
            ToolClientError: The server could not be reached or returned an unparseable
                response. A tool that *ran* and failed returns ``ToolResult(ok=False)``.
        """
        ...

    async def list_tools(self) -> list[str]:
        """Names of the tools the server advertises."""
        ...


def _parse_tool_output(raw: Any, *, tool: str) -> ToolResult:
    """Coerce a FastMCP ``call_tool`` return value into a :class:`ToolResult`.

    FastMCP returns ``(content_blocks, structured_dict)``; we consume the structured dict,
    which is what an A2A caller would receive. Anything else is a contract violation and is
    raised rather than guessed at.
    """
    structured = raw[1] if isinstance(raw, tuple) and len(raw) == 2 else raw
    if not isinstance(structured, dict):
        raise ToolClientError(f"tool '{tool}' returned a non-object result")
    try:
        return ToolResult.model_validate(structured)
    except Exception as exc:
        raise ToolClientError(
            f"tool '{tool}' returned a result that does not match the ToolResult contract "
            f"({type(exc).__name__})"
        ) from None


class InProcessToolClient:
    """Drives a FastMCP server object directly, without a network hop."""

    def __init__(self, server: Any) -> None:
        self._server = server

    async def call(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            raw = await self._server.call_tool(tool, arguments)
        except Exception as exc:
            # A raised exception here means the dispatch itself broke — the tool wrapper is
            # meant to convert argument and handler errors into ToolResults, so this is a
            # genuine transport-level fault.
            raise ToolClientError(
                f"in-process call to '{tool}' failed ({type(exc).__name__})"
            ) from None
        return _parse_tool_output(raw, tool=tool)

    async def list_tools(self) -> list[str]:
        tools = await self._server.list_tools()
        return [t.name for t in tools]

    async def aclose(self) -> None:  # symmetry with the HTTP client; nothing to release
        return None


class HttpMCPToolClient:
    """Connects to a remote MCP server over streamable-HTTP.

    A fresh session is opened per call. That is deliberate: the specialist agents make a
    handful of tool calls per task, and a per-call session keeps the client stateless and
    removes any question of a connection bound to a stale event loop (the failure mode that
    bit the Groq provider in Phase 4). If call volume ever justified it, a pooled session
    would be a local optimisation behind this same interface.
    """

    def __init__(self, base_url: str, *, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/") + "/mcp"
        self._timeout = timeout_seconds

    async def call(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        raw: Any = await self._with_session(lambda session: session.call_tool(tool, arguments))
        # The streamable-HTTP client returns a CallToolResult; its structured content is the
        # dict our tools emit.
        structured = getattr(raw, "structuredContent", None)
        if structured is None:
            raise ToolClientError(f"tool '{tool}' returned no structured content")
        return _parse_tool_output(structured, tool=tool)

    async def list_tools(self) -> list[str]:
        result: Any = await self._with_session(lambda session: session.list_tools())
        return [t.name for t in result.tools]

    async def _with_session[T](self, operation: Any) -> T:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:  # pragma: no cover - mcp is a pinned dependency
            raise ToolClientError("the 'mcp' client is not installed") from exc

        try:
            async with (
                streamablehttp_client(self._base_url) as (read, write, _get_session_id),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                return await operation(session)  # type: ignore[no-any-return]
        except ToolClientError:
            raise
        except Exception as exc:
            raise ToolClientError(f"could not reach MCP server ({type(exc).__name__})") from None

    async def aclose(self) -> None:  # sessions are per-call; nothing persistent to close
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
