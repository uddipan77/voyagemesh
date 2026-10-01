"""Stay Agent A2A service entrypoint.

Run standalone:
    uv run uvicorn vm_stay_agent.app:app --port 8020
"""

from __future__ import annotations

from vm_config.settings import get_settings
from vm_harness.service_factory import build_agent_service
from vm_stay_agent.agent import build_agent

_settings = get_settings()
app = build_agent_service(
    build_agent,
    settings=_settings,
    url=_settings.a2a.stay_agent_url,
    mcp_url=_settings.mcp.lodging_url,
)
