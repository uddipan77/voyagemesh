"""Guardrails: input, output, and prompt-injection detection.

Layered defence (brief §16). Structural invariants live in the contract models (a price
always has a currency, a budget re-verifies its sum, an itinerary day cannot overlap);
this package adds the cross-component and domain-specific checks a single model cannot
express, plus prompt-injection detection as defence in depth (threat T-1).
"""

from vm_guardrails.injection import InjectionScan, scan_for_injection
from vm_guardrails.input_guards import InputGuardResult, check_trip_request
from vm_guardrails.output_guards import GuardSeverity, OutputGuardResult, check_plan

__all__ = [
    "GuardSeverity",
    "InjectionScan",
    "InputGuardResult",
    "OutputGuardResult",
    "check_plan",
    "check_trip_request",
    "scan_for_injection",
]
