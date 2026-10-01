"""VoyageMesh API gateway — the authenticated, rate-limited HTTP front door.

Owns the request lifecycle (auth → rate limit → size guard → idempotency → plan → persist),
exposes versioned ``/api/v1`` routes plus health probes, and returns only contract-shaped,
sanitised error responses. See :func:`vm_api_gateway.app.create_app`.
"""

from vm_api_gateway.app import create_app
from vm_api_gateway.services import GatewayServices

__all__ = ["GatewayServices", "create_app"]
