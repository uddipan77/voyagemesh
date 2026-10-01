"""Transport MCP server — the Transport Agent's only route to transport data.

Tools (all read-only):

* ``search_ground_transport`` — rail and coach options
* ``search_flights`` — air options
* ``normalize_transport_offer`` — provider payload to canonical offer
* ``calculate_transport_score`` — deterministic ranking
* ``validate_transport_offer`` — structural and pricing checks

The scoring tool is worth noting: it exists so that an agent *can* rank without doing the
arithmetic itself. The agent asks for a score; the code computes it (ADR-009).
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from vm_config.settings import ProviderMode, Settings
from vm_contracts.mcp import (
    ScoreTransportArgs,
    ToolResult,
    TransportSearchArgs,
)
from vm_contracts.mcp_runtime import (
    ManagedMCPServer,
    ToolRegistry,
    build_health_routes,
    strip_computed_fields,
)
from vm_contracts.offers import TransportOffer
from vm_domain.providers.base import TransportProvider, TransportQuery
from vm_domain.providers.duffel import DuffelTransportProvider, UnavailableGroundTransportProvider
from vm_domain.providers.live_support import make_provider_client, provider_lifespan
from vm_domain.providers.mock_transport import MockTransportProvider
from vm_domain.scoring import TOP_N, filter_transport, rank_transport

SERVICE_NAME = "transport-mcp"
VERSION = "0.1.0"


def build_server(settings: Settings | None = None) -> FastMCP:
    """Construct the server with its tools registered."""
    config = settings or Settings()
    client = (
        make_provider_client(config.providers)
        if config.providers.mode is not ProviderMode.MOCK
        else None
    )
    server = ManagedMCPServer(
        name=SERVICE_NAME,
        instructions=(
            "Transport search and deterministic ranking for VoyageMesh. All tools are "
            "read-only. Prices and scores are computed by code, never estimated by a model."
        ),
        host=config.host,
        port=8110,
        stateless_http=True,
        app_lifespan=provider_lifespan(client),
    )
    registry = ToolRegistry(
        SERVICE_NAME, max_calls_per_session=config.mcp.max_tool_calls_per_task * 8
    )
    ground_provider: TransportProvider = (
        MockTransportProvider() if client is None else UnavailableGroundTransportProvider()
    )
    air_provider: TransportProvider = (
        MockTransportProvider()
        if client is None
        else DuffelTransportProvider(
            client,
            api_key_file=config.providers.duffel_api_key_file,
            search_timeout_seconds=config.providers.search_timeout_seconds,
        )
    )

    async def search_ground(args: TransportSearchArgs) -> ToolResult:
        return await _search(ground_provider, args, air_only=False)

    async def search_air(args: TransportSearchArgs) -> ToolResult:
        return await _search(air_provider, args, air_only=True)

    async def score(args: ScoreTransportArgs) -> ToolResult:
        offers, invalid = _parse_offers(args.offers)
        if not offers:
            return ToolResult.failure(
                code="no_valid_offers",
                message=f"none of the {len(args.offers)} supplied offers were valid",
                source_name=SERVICE_NAME,
            )
        ranked = rank_transport(offers, args.ranking_strategy)
        warnings = [f"{len(invalid)} offer(s) were rejected as malformed"] if invalid else []
        return ToolResult.success(
            origin="computed",
            source_name=SERVICE_NAME,
            data={
                "ranking_strategy": args.ranking_strategy.value,
                "offers": [o.model_dump(mode="json") for o in ranked],
            },
            warnings=warnings,
        )

    async def normalize(args: ScoreTransportArgs) -> ToolResult:
        offers, invalid = _parse_offers(args.offers)
        return ToolResult.success(
            origin="computed",
            source_name=SERVICE_NAME,
            data={
                "offers": [o.model_dump(mode="json") for o in offers],
                "rejected_count": len(invalid),
            },
            warnings=[f"{reason}" for reason in invalid[:5]],
        )

    async def validate(args: ScoreTransportArgs) -> ToolResult:
        offers, invalid = _parse_offers(args.offers)
        return ToolResult.success(
            origin="computed",
            source_name=SERVICE_NAME,
            data={
                "valid_count": len(offers),
                "invalid_count": len(invalid),
                "valid_offer_ids": [o.offer_id for o in offers],
                "problems": invalid[:10],
            },
        )

    registry.register(
        server,
        name="search_ground_transport",
        description=(
            "Search rail and coach options between two cities. Returns offers ranked by "
            "the requested strategy, each labelled with its data origin."
        ),
        args_model=TransportSearchArgs,
        handler=search_ground,
    )
    registry.register(
        server,
        name="search_flights",
        description=(
            "Search air options between two cities. Returns an empty result for short "
            "routes where flying is not sensible once airport transfers are counted."
        ),
        args_model=TransportSearchArgs,
        handler=search_air,
    )
    registry.register(
        server,
        name="normalize_transport_offer",
        description=(
            "Convert raw provider offers into the canonical schema, dropping malformed ones."
        ),
        args_model=ScoreTransportArgs,
        handler=normalize,
    )
    registry.register(
        server,
        name="calculate_transport_score",
        description=(
            "Deterministically score and rank offers. Lower is better. Returns the score "
            "breakdown so the trade-off can be explained to the traveller."
        ),
        args_model=ScoreTransportArgs,
        handler=score,
    )
    registry.register(
        server,
        name="validate_transport_offer",
        description="Check offers for structural and pricing consistency without ranking them.",
        args_model=ScoreTransportArgs,
        handler=validate,
    )

    build_health_routes(server, service_name=SERVICE_NAME, version=VERSION)
    server.voyagemesh_registry = registry  # type: ignore[attr-defined]
    return server


async def _search(
    provider: TransportProvider, args: TransportSearchArgs, *, air_only: bool
) -> ToolResult:
    from vm_contracts.common import TransportMode

    mode = args.preferred_mode
    if air_only:
        mode = TransportMode.FLIGHT

    result = await provider.search(
        TransportQuery(
            origin=args.origin,
            destination=args.destination,
            departure_date=args.departure_date,
            return_date=args.return_date,
            travellers=args.travellers,
            currency=args.currency,
            preferred_mode=mode,
            max_duration_hours=args.max_duration_hours,
            max_transfers=args.max_transfers,
            accessibility_needs=tuple(args.accessibility_needs),
        )
    )

    if not result.ok:
        return ToolResult.failure(
            code="provider_unavailable",
            message=result.error or "transport provider unavailable",
            source_name=result.source_name,
            retryable=False,
        )

    kept, reasons = filter_transport(
        result.items,
        max_duration_hours=args.max_duration_hours,
        max_transfers=args.max_transfers,
        accessibility_needs=list(args.accessibility_needs),
    )
    ranked = rank_transport(kept, args.ranking_strategy)[: min(args.limit, TOP_N * 2)]

    warnings = list(result.warnings)
    if reasons:
        warnings.append(f"{len(reasons)} option(s) excluded by your constraints")
        warnings.extend(reasons[:5])

    return ToolResult.success(
        origin=result.origin.value,
        source_name=result.source_name,
        data={
            "offers": [o.model_dump(mode="json") for o in ranked],
            "total_found": len(result.items),
            "excluded_by_constraints": len(reasons),
        },
        warnings=warnings[:20],
    )


def _parse_offers(raw: list[dict[str, Any]]) -> tuple[list[TransportOffer], list[str]]:
    """Validate raw offers, collecting reasons for the ones that fail.

    Dropping bad records rather than raising means one malformed offer from a provider
    does not cost the traveller the entire result set (threat T-5).
    """
    offers: list[TransportOffer] = []
    problems: list[str] = []
    for index, item in enumerate(raw):
        try:
            offers.append(
                TransportOffer.model_validate(strip_computed_fields(item, TransportOffer))
            )
        except ValidationError as exc:
            details = exc.errors()
            if details:
                location = ".".join(str(part) for part in details[0]["loc"]) or "<root>"
                kind = details[0]["type"]
            else:  # pragma: no cover - pydantic always populates at least one error
                location, kind = "<root>", "invalid"
            problems.append(f"offer[{index}]: {location} {kind}")
    return offers, problems


def main() -> None:  # pragma: no cover - process entrypoint
    build_server().run(transport="streamable-http")


if __name__ == "__main__":  # pragma: no cover
    main()
