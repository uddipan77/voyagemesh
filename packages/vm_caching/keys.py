"""Deterministic cache keys.

A trip's cache identity is a SHA-256 over its *normalised* fingerprint plus the workflow
version (brief §17). Normalising before hashing is what makes "Prague" and "  prague " the
same entry; including the workflow version is what invalidates the whole namespace on a
deploy, so a changed planning algorithm cannot serve stale results from the old one.
"""

from __future__ import annotations

import hashlib
import json

from vm_contracts.trip import NormalizedTripRequest

__all__ = ["WORKFLOW_VERSION", "plan_cache_key"]

# Bump this when the planning workflow changes in a way that should invalidate cached plans.
WORKFLOW_VERSION = "v1"


def plan_cache_key(normalized: NormalizedTripRequest) -> str:
    """The cache key for a normalised trip request.

    Content-addressed: a different request cannot collide onto an existing entry (threat
    T-8), and an identical request always maps to the same key.
    """
    fingerprint = normalized.cache_fingerprint()
    # sort_keys makes the serialisation canonical regardless of dict insertion order.
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{WORKFLOW_VERSION}|{payload}".encode()).hexdigest()
    return f"trip:{WORKFLOW_VERSION}:{digest}"
