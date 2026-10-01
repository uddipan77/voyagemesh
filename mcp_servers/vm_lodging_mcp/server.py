"""Lodging MCP server — the Stay Agent's only route to accommodation data.

Tools (all read-only):

* ``search_accommodation`` — ranked accommodation options
* ``normalize_accommodation_offer`` — provider payload to canonical offer
* ``calculate_total_stay_cost`` — nightly rate to full-stay total, by code
* ``calculate_distance_score`` — proximity scoring
* ``validate_accommodation_offer`` — structural and pricing checks
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from vm_config.settings import ProviderMode, Settings
from vm_contracts.common import Money
from vm_contracts.mcp import (
    AccommodationSearchArgs,
    DistanceArgs,
    ScoreTransportArgs,
    StayCostArgs,
    ToolResult,
)
from vm_contracts.mcp_runtime import (
    ManagedMCPServer,
    ToolRegistry,
    build_health_routes,
    strip_computed_fields,
)
from vm_contracts.offers import AccommodationOffer
from vm_domain.geo import distance_km
from vm_domain.providers.base import AccommodationProvider, AccommodationQuery
from vm_domain.providers.liteapi import LiteAPIAccommodationProvider
from vm_domain.providers.live_support import make_provider_client, provider_lifespan
from vm_domain.providers.mock_accommodation import MockAccommodationProvider
from vm_domain.scoring import TOP_N, filter_accommodation, rank_accommodation

SERVICE_NAME = "lodging-mcp"
VERSION = "0.1.0"


def build_server(settings: Settings | None = None) -> FastMCP:
    config = settings or Settings()
    client = (
        make_provider_client(config.providers)
        if config.providers.mode is not ProviderMode.MOCK
        else None
    )
    server = ManagedMCPServer(
        name=SERVICE_NAME,
        instructions=(
            "Accommodation search and deterministic ranking for VoyageMesh. All tools are "
            "read-only. Stay totals are computed by code, never estimated by a model."
        ),
        host=config.host,
        port=8120,
        stateless_http=True,
        app_lifespan=provider_lifespan(client),
    )
    registry = ToolRegistry(
        SERVICE_NAME, max_calls_per_session=config.mcp.max_tool_calls_per_task * 8
    )
    provider: AccommodationProvider = (
        MockAccommodationProvider()
        if client is None
        else LiteAPIAccommodationProvider(
            client,
            api_key_file=config.providers.liteapi_api_key_file,
            search_timeout_seconds=config.providers.search_timeout_seconds,
        )
    )

    async def search(args: AccommodationSearchArgs) -> ToolResult:
        result = await provider.search(
            AccommodationQuery(
                destination=args.destination,
                check_in=args.check_in,
                check_out=args.check_out,
                guests=args.guests,
                guest_nationality=args.guest_nationality,
                currency=args.currency,
                accommodation_type=args.accommodation_type,
                max_total_price=args.max_total_price,
                accessibility_needs=tuple(args.accessibility_needs),
            )
        )
        if not result.ok:
            return ToolResult.failure(
                code="provider_unavailable",
                message=result.error or "accommodation provider unavailable",
                source_name=result.source_name,
                retryable=False,
            )

        ceiling = Decimal(str(args.max_total_price)) if args.max_total_price is not None else None
        kept, reasons = filter_accommodation(
            result.items,
            accessibility_needs=list(args.accessibility_needs),
            max_total_price=ceiling,
        )
        ranked = rank_accommodation(kept, args.ranking_strategy)[: min(args.limit, TOP_N * 2)]

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

    async def total_cost(args: StayCostArgs) -> ToolResult:
        # This is the whole reason the tool exists: the total is arithmetic, so an agent
        # asks for it rather than multiplying in a prompt (ADR-009).
        nightly = Money.of(Decimal(str(args.price_per_night)), args.currency)
        per_night_all_rooms = nightly.times(args.rooms)
        total = per_night_all_rooms.times(args.nights)
        return ToolResult.success(
            origin="computed",
            source_name=SERVICE_NAME,
            data={
                "price_per_night": nightly.model_dump(mode="json"),
                "rooms": args.rooms,
                "nights": args.nights,
                "total_price": total.model_dump(mode="json"),
                "basis": f"{nightly} x {args.rooms} room(s) x {args.nights} night(s)",
            },
        )

    async def distance_score(args: DistanceArgs) -> ToolResult:
        km = distance_km(args.origin, args.destination)
        # A smooth 0..1 proximity score: 0 at the centre, approaching 1 by ~8 km out.
        proximity = min(1.0, km / 8.0)
        return ToolResult.success(
            origin="computed",
            source_name=SERVICE_NAME,
            data={
                "distance_km": round(km, 3),
                "distance_penalty": round(proximity, 4),
                "note": "0 = at the reference point, 1 = far (>= 8 km)",
            },
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
            warnings=invalid[:5],
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
        name="search_accommodation",
        description=(
            "Search accommodation for a stay. Returns options ranked by the requested "
            "strategy, each priced for the whole stay and labelled with its data origin."
        ),
        args_model=AccommodationSearchArgs,
        handler=search,
    )
    registry.register(
        server,
        name="calculate_total_stay_cost",
        description=(
            "Compute the total cost of a stay from the nightly rate, room count, and number "
            "of nights. Deterministic arithmetic — the figure an agent must not estimate."
        ),
        args_model=StayCostArgs,
        handler=total_cost,
    )
    registry.register(
        server,
        name="calculate_distance_score",
        description="Compute the distance and a 0..1 proximity penalty between two points.",
        args_model=DistanceArgs,
        handler=distance_score,
    )
    registry.register(
        server,
        name="normalize_accommodation_offer",
        description=(
            "Convert raw provider offers into the canonical schema, dropping malformed ones."
        ),
        args_model=ScoreTransportArgs,
        handler=normalize,
    )
    registry.register(
        server,
        name="validate_accommodation_offer",
        description="Check offers for structural and pricing consistency without ranking them.",
        args_model=ScoreTransportArgs,
        handler=validate,
    )

    build_health_routes(server, service_name=SERVICE_NAME, version=VERSION)
    server.voyagemesh_registry = registry  # type: ignore[attr-defined]
    return server


def _parse_offers(raw: list[dict[str, Any]]) -> tuple[list[AccommodationOffer], list[str]]:
    offers: list[AccommodationOffer] = []
    problems: list[str] = []
    for index, item in enumerate(raw):
        try:
            offers.append(
                AccommodationOffer.model_validate(strip_computed_fields(item, AccommodationOffer))
            )
        except ValidationError as exc:
            details = exc.errors()
            if details:
                location = ".".join(str(part) for part in details[0]["loc"]) or "<root>"
                kind = details[0]["type"]
            else:  # pragma: no cover
                location, kind = "<root>", "invalid"
            problems.append(f"offer[{index}]: {location} {kind}")
    return offers, problems


def main() -> None:  # pragma: no cover - process entrypoint
    build_server().run(transport="streamable-http")


if __name__ == "__main__":  # pragma: no cover
    main()
