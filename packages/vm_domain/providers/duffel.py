"""Duffel v2 flight searches; no orders, holds or payment calls.

Contract: https://duffel.com/docs/api/offer-requests
City/airport suggestions: https://duffel.com/docs/api/places
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from vm_config.secrets import try_read_secret_file
from vm_contracts.common import DataOrigin, Money, Provenance, TransportMode, utc_now
from vm_contracts.offers import TransportLeg, TransportOffer
from vm_domain.providers.base import ProviderResult, TransportQuery
from vm_domain.providers.live_support import failure_message, object_payload, records
from vm_resilience.safe_http import SafeHTTPClient


class DuffelTransportProvider:
    name = "Duffel Flights"

    def __init__(
        self, client: SafeHTTPClient, *, api_key_file: str | None, search_timeout_seconds: int = 6
    ) -> None:
        self._client = client
        self._key_file = api_key_file
        self._timeout = search_timeout_seconds

    async def search(self, query: TransportQuery) -> ProviderResult[TransportOffer]:
        key = try_read_secret_file(self._key_file, name=self.name)
        if key is None:
            return ProviderResult.unavailable(
                self.name, "Duffel: set PROVIDER_DUFFEL_API_KEY_FILE to a readable live token file."
            )
        if not key.get_secret_value().startswith("duffel_live_"):
            return ProviderResult.unavailable(
                self.name, "Duffel requires a live token; test tokens are refused."
            )
        headers = {"Authorization": f"Bearer {key.get_secret_value()}", "Duffel-Version": "v2"}
        try:
            origin, destination = await asyncio.gather(
                self._resolve(query.origin, headers), self._resolve(query.destination, headers)
            )
            slices = [
                {
                    "origin": origin,
                    "destination": destination,
                    "departure_date": query.departure_date.isoformat(),
                }
            ]
            if query.return_date is not None:
                slices.append(
                    {
                        "origin": destination,
                        "destination": origin,
                        "departure_date": query.return_date.isoformat(),
                    }
                )
            payload = object_payload(
                await self._client.post_json(
                    "https://api.duffel.com/air/offer_requests",
                    params={"return_offers": "true", "supplier_timeout": self._timeout * 1000},
                    headers=headers,
                    json={
                        "data": {
                            "slices": slices,
                            "passengers": [{"type": "adult"} for _ in range(query.travellers)],
                            "cabin_class": "economy",
                            "max_connections": query.max_transfers,
                        }
                    },
                )
            )
            data = object_payload(payload["data"])
            if data.get("live_mode") is not True:
                return ProviderResult.unavailable(
                    self.name, "Duffel returned test data; no live quotes shown."
                )
            offers: list[TransportOffer] = []
            rejected = 0
            for raw in records(data["offers"]):
                try:
                    offers.append(_offer(raw, query))
                except (ValueError, KeyError, TypeError, ArithmeticError):
                    rejected += 1
            warnings = [
                "Flight prices cover all adults and both journeys when a return date is supplied. "
                "Airport transfers and optional baggage are not included; fares can change.",
                f"Flight search resolved {query.origin} to {origin} "
                f"and {query.destination} to {destination}.",
                "Itinerary activity times are tentative; "
                "check them against flight arrival and departure.",
            ]
            if rejected:
                warnings.append(
                    f"{rejected} flight quote(s) rejected: invalid, expired, "
                    "test data or different currency."
                )
            return ProviderResult(offers, DataOrigin.LIVE, self.name, warnings=warnings)
        except Exception as exc:
            return ProviderResult.unavailable(self.name, failure_message(self.name, exc))

    async def _resolve(self, place: str, headers: dict[str, str]) -> str:
        # Explicit IATA codes bypass name suggestions, avoiding ambiguous city matches.
        if re.fullmatch(r"[A-Z]{3}", place):
            return place
        payload = object_payload(
            await self._client.get_json(
                "https://api.duffel.com/places/suggestions",
                params={"query": place},
                headers=headers,
            )
        )
        candidates = records(payload["data"])
        exact = [p for p in candidates if str(p.get("name", "")).casefold() == place.casefold()]
        candidates = exact or candidates
        if not candidates:
            raise ValueError("no matching airport or city")
        # A city code searches all its airports. Suggestions remain provider-controlled.
        selected = next((p for p in candidates if p.get("type") == "city"), candidates[0])
        code = selected["iata_code"]
        if not isinstance(code, str) or not re.fullmatch(r"[A-Z]{3}", code):
            raise ValueError("invalid IATA code")
        return code


def _time(value: str, airport: dict[str, Any]) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(airport["time_zone"]))
    # UTC arithmetic handles DST changes and connections across time zones correctly.
    return parsed.astimezone(UTC)


def _leg(segment: dict[str, Any]) -> TransportLeg:
    origin, destination = segment["origin"], segment["destination"]
    carrier = segment["operating_carrier"]
    number = segment.get("operating_carrier_flight_number")
    return TransportLeg(
        mode=TransportMode.FLIGHT,
        carrier=carrier["name"],
        from_location=f"{origin['name']} ({origin['iata_code']})",
        to_location=f"{destination['name']} ({destination['iata_code']})",
        departure_at=_time(segment["departing_at"], origin),
        arrival_at=_time(segment["arriving_at"], destination),
        service_identifier=f"{carrier.get('iata_code') or ''}{number}" if number else None,
    )


def _offer(raw: dict[str, Any], query: TransportQuery) -> TransportOffer:
    if raw.get("live_mode") is not True or raw["total_currency"] != query.currency.value:
        raise ValueError("test quote or currency mismatch")
    expiry = datetime.fromisoformat(raw["expires_at"])
    if expiry.tzinfo is None or expiry <= utc_now():
        raise ValueError("expired quote")
    slices = records(raw["slices"])
    if len(slices) != (2 if query.return_date is not None else 1):
        raise ValueError("incomplete journey")
    if len(records(raw["passengers"])) != query.travellers:
        raise ValueError("passenger count mismatch")
    legs = [_leg(s) for s in records(slices[0]["segments"])]
    returns = [_leg(s) for s in records(slices[1]["segments"])] if len(slices) == 2 else []
    total = Money.of(Decimal(str(raw["total_amount"])), query.currency)
    return TransportOffer(
        offer_id=raw["id"],
        legs=legs,
        return_legs=returns,
        expires_at=expiry,
        total_price=total,
        price_per_traveller=Money.of(total.amount / query.travellers, query.currency),
        travellers=query.travellers,
        provenance=Provenance.live(
            "Duffel Flights", source_id=raw["id"], source_url="https://duffel.com"
        ),
    )


class UnavailableGroundTransportProvider:
    """No public Omio partner contract has been supplied. Never fabricate a fare."""

    name = "Ground transport"

    async def search(self, query: TransportQuery) -> ProviderResult[TransportOffer]:
        return ProviderResult.unavailable(
            self.name,
            "Live rail and bus search is not connected. Omio partner credentials and API "
            "documentation are required to implement its TransportProvider adapter.",
        )
