"""A metrics/tracing decorator for any LLMProvider.

Wrapping a provider here means every LLM call — whichever provider is selected — emits the same
Prometheus metrics (requests, failures, latency, input/output tokens, structured-output
failures) and a span, without each provider re-implementing instrumentation. The wrapper is
transparent: it forwards ``name``, ``model``, and ``health_check`` and returns the inner
provider's response unchanged.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

from vm_llm.base import LLMProvider
from vm_llm.types import LLMResponse, Message, StructuredOutputError
from vm_telemetry import metrics, observe_llm, traced

if TYPE_CHECKING:
    pass

__all__ = ["InstrumentedLLMProvider"]


class InstrumentedLLMProvider:
    """Wraps an ``LLMProvider`` to record metrics and a span around each call."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.name = inner.name
        self.model = inner.model

    async def generate_structured[T: BaseModel](
        self,
        messages: list[Message],
        response_model: type[T],
        *,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        max_output_tokens: int | None = None,
    ) -> LLMResponse[T]:
        started = time.monotonic()
        with traced("llm.generate", **{"vm.llm.provider": self.name, "vm.llm.model": self.model}):
            try:
                response = await self._inner.generate_structured(
                    messages,
                    response_model,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    max_output_tokens=max_output_tokens,
                )
            except StructuredOutputError:
                metrics.structured_output_failures_total.inc()
                observe_llm(
                    provider=self.name,
                    model=self.model,
                    duration_s=time.monotonic() - started,
                    input_tokens=0,
                    output_tokens=0,
                    failed=True,
                )
                raise
            except Exception:
                observe_llm(
                    provider=self.name,
                    model=self.model,
                    duration_s=time.monotonic() - started,
                    input_tokens=0,
                    output_tokens=0,
                    failed=True,
                )
                raise

        observe_llm(
            provider=self.name,
            model=response.model or self.model,
            duration_s=time.monotonic() - started,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response

    async def health_check(self) -> bool:
        return await self._inner.health_check()
