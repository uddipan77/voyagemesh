"""Reusable agent harness: tool clients, run context, and the base specialist agent.

Every specialist agent shares this machinery — bounded tool loops, stopping conditions,
auditable action traces, token accounting, structured narration, and A2A envelope handling
— so those guarantees are implemented once rather than per agent (brief §13).
"""

from vm_harness.a2a_client import (
    A2AClientError,
    A2ADiscoveryError,
    A2AResponseError,
    AgentClient,
    HttpAgentClient,
    InProcessAgentClient,
    submit_verified,
)
from vm_harness.a2a_server import build_a2a_app
from vm_harness.agent import AgentOutcome, SpecialistAgent
from vm_harness.harness import AgentHarness
from vm_harness.mcp_client import (
    HttpMCPToolClient,
    InProcessToolClient,
    ToolClient,
    ToolClientError,
)
from vm_harness.service_factory import (
    build_agent_service,
    build_service_token_provider,
    select_service_authenticator,
)
from vm_harness.trace import (
    ActionRecord,
    ActionStatus,
    AgentRunContext,
    StoppingConditionReached,
    StopReason,
)

__all__ = [
    "A2AClientError",
    "A2ADiscoveryError",
    "A2AResponseError",
    "ActionRecord",
    "ActionStatus",
    "AgentClient",
    "AgentHarness",
    "AgentOutcome",
    "AgentRunContext",
    "HttpAgentClient",
    "HttpMCPToolClient",
    "InProcessAgentClient",
    "InProcessToolClient",
    "SpecialistAgent",
    "StopReason",
    "StoppingConditionReached",
    "ToolClient",
    "ToolClientError",
    "build_a2a_app",
    "build_agent_service",
    "build_service_token_provider",
    "select_service_authenticator",
    "submit_verified",
]
