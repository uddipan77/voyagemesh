"""Request and response envelopes specific to the HTTP API.

The domain contracts (:class:`TripRequest`, :class:`TripPlan`) are the payloads; these small
wrappers add only what the transport needs — a health report and a liveness report — and keep
the OpenAPI schema honest about what each endpoint returns.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from vm_contracts.common import StrictModel

__all__ = ["HealthResponse", "LivenessResponse", "ReadinessResponse"]


class LivenessResponse(StrictModel):
    """Answers only "is the process up?" — never touches a dependency."""

    status: Literal["alive"] = "alive"
    service: Annotated[str, Field(max_length=60)]
    version: Annotated[str, Field(max_length=30)]


class ReadinessResponse(StrictModel):
    """Reports each fail-open dependency's state. The gateway is ``ready`` even when a
    dependency is ``degraded`` — that is the whole point of failing open (brief §17)."""

    status: Literal["ready"] = "ready"
    components: dict[str, str] = Field(default_factory=dict)


# A liveness alias kept for symmetry with the readiness naming used across services.
HealthResponse = LivenessResponse
