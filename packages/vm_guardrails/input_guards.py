"""Input guardrails.

Most structural validation is already done by ``TripRequest``'s own validators — dates,
budget, currency, place-name shape. This layer adds what a schema cannot express: scanning
the free-text fields for prompt injection, and enforcing configurable request limits.

An input guardrail *rejects* rather than sanitises. Silently editing a user's request to
make it "safe" hides the attempt and can produce a plan they did not ask for; refusing with
a clear reason is both safer and more honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vm_contracts.trip import TripRequest
from vm_guardrails.injection import scan_for_injection

__all__ = ["InputGuardResult", "check_trip_request"]


@dataclass(frozen=True, slots=True)
class InputGuardResult:
    """The outcome of the input guardrail check."""

    passed: bool
    errors: tuple[str, ...] = ()
    injection_categories: tuple[str, ...] = field(default=())

    @property
    def rejected_for_injection(self) -> bool:
        return bool(self.injection_categories)


def check_trip_request(
    request: TripRequest,
    *,
    detect_injection: bool = True,
    max_interests: int = 12,
) -> InputGuardResult:
    """Run input guardrails over a (schema-valid) trip request.

    Args:
        request: An already schema-validated request.
        detect_injection: Whether to scan free-text fields for injection patterns.
        max_interests: Configurable ceiling on the interest count.

    Returns:
        An :class:`InputGuardResult`. When ``passed`` is ``False`` the request must be
        refused with the given errors.
    """
    errors: list[str] = []
    injection_categories: list[str] = []

    if len(request.interests) > max_interests:
        errors.append(f"too many interests ({len(request.interests)}, max {max_interests})")

    if detect_injection:
        # Scan every field a user controls that could reach a prompt. Place names are already
        # shape-restricted by TripRequest, but notes and interests are freer.
        fields_to_scan = {
            "notes": request.notes or "",
            "origin": request.origin,
            "destination": request.destination,
        }
        for index, interest in enumerate(request.interests):
            fields_to_scan[f"interest[{index}]"] = interest

        for name, value in fields_to_scan.items():
            scan = scan_for_injection(value, field=name)
            if scan.is_suspicious:
                injection_categories.extend(scan.categories)
                errors.append(scan.summary)

    passed = not errors
    return InputGuardResult(
        passed=passed,
        errors=tuple(errors),
        injection_categories=tuple(dict.fromkeys(injection_categories)),
    )
