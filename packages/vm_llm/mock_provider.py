"""Deterministic mock LLM provider.

Used by the entire test suite, by ``LLM_PROVIDER=mock``, and as the controlled degradation
path when Groq is unavailable in development. It is genuinely deterministic — unlike the
real provider, which is not reproducible even at ``temperature=0.0`` (verified against the
live API; see the Phase 4 build log).

Values are synthesised from the response model's own schema, so it satisfies any
`BaseModel` without needing a hand-written fixture per call site. Text fields are seeded
from the prompt, so different prompts give different — but stable — answers.

Failure injection
-----------------
``fail_with`` and ``fail_times`` let tests exercise retry, backoff, and degradation paths
without waiting on a real provider or burning quota.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from pydantic import BaseModel

from vm_llm.types import (
    LLMError,
    LLMResponse,
    Message,
    TokenUsage,
)

__all__ = ["MockLLMProvider"]


class MockLLMProvider:
    """Schema-driven deterministic provider."""

    name = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-model-v1",
        fixed_responses: dict[type[BaseModel], BaseModel] | None = None,
        fail_with: LLMError | None = None,
        fail_times: int = 0,
        latency_ms: int = 1,
    ) -> None:
        """Construct the mock.

        Args:
            model: Reported model name.
            fixed_responses: Exact values to return for given response models, when a test
                needs specific content rather than synthesised content.
            fail_with: Error to raise. Combined with ``fail_times`` to simulate a
                dependency that fails then recovers.
            fail_times: Number of calls to fail before succeeding. ``0`` with ``fail_with``
                set means fail every time.
            latency_ms: Reported latency.
        """
        self.model = model
        self._fixed = fixed_responses or {}
        self._fail_with = fail_with
        self._fail_times = fail_times
        self._latency_ms = latency_ms
        self.call_count = 0
        self.calls: list[list[Message]] = []

    async def generate_structured[T: BaseModel](
        self,
        messages: list[Message],
        response_model: type[T],
        *,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        max_output_tokens: int | None = None,
    ) -> LLMResponse[T]:
        self.call_count += 1
        self.calls.append(list(messages))

        if self._fail_with is not None and (
            self._fail_times == 0 or self.call_count <= self._fail_times
        ):
            raise self._fail_with

        started = time.perf_counter()
        fixed = self._fixed.get(response_model)
        value = (
            fixed
            if fixed is not None
            else response_model.model_validate(
                _synthesise(response_model, seed=_seed(messages, response_model))
            )
        )

        prompt_chars = sum(len(m.content) for m in messages)
        usage = TokenUsage(
            # A rough 4-chars-per-token heuristic. Good enough for exercising accounting
            # and dashboards; never presented as a real count.
            input_tokens=max(1, prompt_chars // 4),
            output_tokens=32,
            estimated_cost_usd=0.0,
        )

        elapsed = max(self._latency_ms, int((time.perf_counter() - started) * 1000))
        return LLMResponse(
            value=value,  # type: ignore[arg-type]
            model=self.model,
            usage=usage,
            latency_ms=elapsed,
            provider=self.name,
            attempts=1,
            repaired=False,
            finish_reason="stop",
        )

    async def health_check(self) -> bool:
        return self._fail_with is None

    def __repr__(self) -> str:
        return f"MockLLMProvider(model={self.model!r}, calls={self.call_count})"


def _seed(messages: list[Message], response_model: type[BaseModel]) -> int:
    payload = f"{response_model.__name__}|" + "|".join(m.content for m in messages)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def _synthesise(model: type[BaseModel], *, seed: int) -> dict[str, Any]:
    """Build a payload satisfying ``model``'s JSON schema."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    result = _from_schema(schema, defs, seed=seed, depth=0)
    return result if isinstance(result, dict) else {}


def _from_schema(schema: dict[str, Any], defs: dict[str, Any], *, seed: int, depth: int) -> Any:
    if depth > 6:  # guards against a self-referential schema
        return None

    if "$ref" in schema:
        ref = schema["$ref"].rsplit("/", 1)[-1]
        return _from_schema(defs.get(ref, {}), defs, seed=seed, depth=depth + 1)

    for key in ("anyOf", "oneOf"):
        if key in schema:
            # Prefer a non-null branch so optional fields get realistic content.
            options = [o for o in schema[key] if o.get("type") != "null"] or schema[key]
            return _from_schema(options[seed % len(options)], defs, seed=seed, depth=depth + 1)

    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        values = schema["enum"]
        return values[seed % len(values)] if values else None
    if "default" in schema:
        return schema["default"]

    schema_type = schema.get("type")

    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        payload: dict[str, Any] = {}
        for index, (name, subschema) in enumerate(properties.items()):
            # Populate required fields plus, deterministically, some optional ones — so
            # tests see both populated and defaulted paths.
            if name in required or (seed >> (index % 8)) & 1:
                payload[name] = _from_schema(
                    subschema, defs, seed=seed + index * 31, depth=depth + 1
                )
        return payload

    if schema_type == "array":
        minimum = int(schema.get("minItems", 1) or 1)
        maximum = int(schema.get("maxItems", minimum + 1) or minimum + 1)
        count = max(1, min(minimum if minimum > 0 else 1, maximum))
        item_schema = schema.get("items", {"type": "string"})
        return [
            _from_schema(item_schema, defs, seed=seed + i * 17, depth=depth + 1)
            for i in range(count)
        ]

    if schema_type == "integer":
        return _bounded_number(schema, seed, integer=True)
    if schema_type == "number":
        return _bounded_number(schema, seed, integer=False)
    if schema_type == "boolean":
        return bool(seed & 1)
    if schema_type == "null":
        return None

    return _string_for(schema, seed)


def _bounded_number(schema: dict[str, Any], seed: int, *, integer: bool) -> Any:
    low = schema.get("minimum", schema.get("exclusiveMinimum", 0))
    high = schema.get("maximum", schema.get("exclusiveMaximum", low + 100))
    if schema.get("exclusiveMinimum") is not None:
        low = schema["exclusiveMinimum"] + (1 if integer else 0.01)
    if schema.get("exclusiveMaximum") is not None:
        high = schema["exclusiveMaximum"] - (1 if integer else 0.01)
    if high < low:
        high = low
    span = high - low
    value = low + (seed % max(1, int(span) + 1) if span else 0)
    value = min(value, high)
    return int(value) if integer else round(float(value), 2)


_LOREM = (
    "The selected option balances cost against journey time",
    "This choice reflects the traveller stated preferences",
    "Deterministic scoring placed this option first",
    "A cheaper alternative existed but required more transfers",
)


def _string_for(schema: dict[str, Any], seed: int) -> str:
    """Produce a string satisfying length and format constraints."""
    fmt = schema.get("format")
    if fmt == "date":
        return "2026-08-10"
    if fmt == "date-time":
        return "2026-08-10T09:00:00+00:00"
    if fmt == "uuid":
        return f"{seed:08x}-0000-4000-8000-000000000000"

    minimum = int(schema.get("minLength", 0) or 0)
    maximum = int(schema.get("maxLength", 200) or 200)

    text = _LOREM[seed % len(_LOREM)]
    if len(text) < minimum:
        # Pad by repeating rather than with filler characters, so the result still reads
        # as text if it surfaces in a snapshot.
        while len(text) < minimum:
            text = f"{text}. {_LOREM[(seed + len(text)) % len(_LOREM)]}"
    if len(text) > maximum:
        text = text[:maximum]
    if len(text) < minimum:  # maximum < minimum would be a broken schema
        text = text.ljust(minimum, "x")
    return text
