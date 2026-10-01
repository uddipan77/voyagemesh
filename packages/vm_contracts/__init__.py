"""Shared contracts: the vocabulary every VoyageMesh service agrees on.

Nothing in this package performs I/O or imports a service. It is the innermost layer of
the dependency graph, which is what allows the orchestrator, the agents, and the MCP
servers to exchange typed data without depending on one another.
"""

from vm_contracts.a2a import (
    A2A_PROTOCOL_VERSION,
    AGENT_CARD_PATH,
    AgentArtifact,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    AgentTask,
    AuthScheme,
    TaskError,
    TaskStatus,
    new_task_id,
)
from vm_contracts.budget import (
    BudgetCategory,
    BudgetLine,
    BudgetStatus,
    BudgetSummary,
)
from vm_contracts.common import (
    AccessibilityNeed,
    AccommodationType,
    ConfidenceLevel,
    Currency,
    DataOrigin,
    GeoPoint,
    Money,
    Provenance,
    RankingStrategy,
    StrictModel,
    TransportMode,
    utc_now,
)
from vm_contracts.destination import (
    Attraction,
    AttractionCategory,
    OpeningHours,
    RetrievedDocument,
    WeatherDay,
    WeatherSummary,
)
from vm_contracts.errors import ErrorCode, ErrorResponse, ValidationProblem
from vm_contracts.itinerary import (
    ActivitySlot,
    Itinerary,
    ItineraryDay,
    SlotKind,
    format_minute_of_day,
)
from vm_contracts.offers import (
    AccommodationOffer,
    CancellationPolicy,
    TransportLeg,
    TransportOffer,
)
from vm_contracts.plan import DataSourceInfo, PlanStatus, TripPlan
from vm_contracts.tracing import (
    HEADER_CORRELATION_ID,
    HEADER_REQUEST_ID,
    HEADER_TRACEPARENT,
    HEADER_TRIP_ID,
    TraceContext,
    new_request_id,
)
from vm_contracts.trip import NormalizedTripRequest, TripRequest

__all__ = [
    "A2A_PROTOCOL_VERSION",
    "AGENT_CARD_PATH",
    "HEADER_CORRELATION_ID",
    "HEADER_REQUEST_ID",
    "HEADER_TRACEPARENT",
    "HEADER_TRIP_ID",
    "AccessibilityNeed",
    "AccommodationOffer",
    "AccommodationType",
    "ActivitySlot",
    "AgentArtifact",
    "AgentCapabilities",
    "AgentCard",
    "AgentSkill",
    "AgentTask",
    "Attraction",
    "AttractionCategory",
    "AuthScheme",
    "BudgetCategory",
    "BudgetLine",
    "BudgetStatus",
    "BudgetSummary",
    "CancellationPolicy",
    "ConfidenceLevel",
    "Currency",
    "DataOrigin",
    "DataSourceInfo",
    "ErrorCode",
    "ErrorResponse",
    "GeoPoint",
    "Itinerary",
    "ItineraryDay",
    "Money",
    "NormalizedTripRequest",
    "OpeningHours",
    "PlanStatus",
    "Provenance",
    "RankingStrategy",
    "RetrievedDocument",
    "SlotKind",
    "StrictModel",
    "TaskError",
    "TaskStatus",
    "TraceContext",
    "TransportLeg",
    "TransportMode",
    "TransportOffer",
    "TripPlan",
    "TripRequest",
    "ValidationProblem",
    "WeatherDay",
    "WeatherSummary",
    "format_minute_of_day",
    "new_request_id",
    "new_task_id",
    "utc_now",
]
