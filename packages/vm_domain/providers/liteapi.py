"""LiteAPI v3 hotel search. Rates are read-only quotes, never reservations.

https://docs.liteapi.travel/reference/post_hotels-rates
https://docs.liteapi.travel/docs/hotel-rates-api-json-data-structure
"""

from __future__ import annotations

import asyncio
import hashlib
from decimal import Decimal
from typing import Any

from vm_config.secrets import try_read_secret_file
from vm_contracts.common import AccommodationType, Currency, DataOrigin, GeoPoint, Money, Provenance
from vm_contracts.offers import AccommodationOffer
from vm_domain.geo import distance_km
from vm_domain.providers.base import AccommodationQuery, ProviderResult
from vm_domain.providers.live_support import failure_message, object_payload, records
from vm_domain.providers.open_meteo import OpenMeteoGeocodingProvider
from vm_resilience.safe_http import SafeHTTPClient

_TYPE_NAMES = {
    AccommodationType.HOSTEL: {"hostel", "hostels"},
    AccommodationType.HOTEL: {"hotel", "hotels"},
    AccommodationType.BUDGET_HOTEL: {"hotel", "hotels"},
    AccommodationType.APARTMENT: {"apartment", "apartments", "aparthotel", "aparthotels"},
    AccommodationType.GUESTHOUSE: {"guest house", "guest houses", "guesthouse", "guesthouses"},
}


class LiteAPIAccommodationProvider:
    name = "LiteAPI Hotels"

    def __init__(
        self, client: SafeHTTPClient, *, api_key_file: str | None, search_timeout_seconds: int = 6
    ) -> None:
        self._client = client
        self._key_file = api_key_file
        self._timeout = search_timeout_seconds

    async def search(self, query: AccommodationQuery) -> ProviderResult[AccommodationOffer]:
        key = try_read_secret_file(self._key_file, name=self.name)
        if key is None:
            return ProviderResult.unavailable(
                self.name,
                "LiteAPI: set PROVIDER_LITEAPI_API_KEY_FILE to a readable production key file.",
            )
        if key.get_secret_value().startswith("sand_"):
            return ProviderResult.unavailable(
                self.name, "LiteAPI requires a production key; sandbox keys are refused."
            )
        if query.guest_nationality is None:
            return ProviderResult.unavailable(
                self.name,
                "Enter the guest nationality (two-letter country code) for live hotel rates.",
            )
        headers = {"X-API-Key": key.get_secret_value()}
        try:
            location_result, type_ids = await asyncio.gather(
                OpenMeteoGeocodingProvider(self._client).geocode(query.destination),
                self._type_ids(query.accommodation_type, headers),
            )
            if not location_result.ok or not location_result.items:
                return ProviderResult.unavailable(
                    self.name, "Could not resolve the hotel search destination."
                )
            centre = location_result.items[0]
            body: dict[str, Any] = {
                "checkin": query.check_in.isoformat(),
                "checkout": query.check_out.isoformat(),
                "guestNationality": query.guest_nationality,
                "currency": query.currency.value,
                "occupancies": [{"adults": query.guests}],
                "latitude": centre.latitude,
                "longitude": centre.longitude,
                "radius": 5000,
                "limit": 20,
                "maxRatesPerHotel": 1,
                "includeHotelData": True,
                "timeout": self._timeout,
            }
            if type_ids:
                body["hotelTypeIds"] = type_ids
            if query.accommodation_type is AccommodationType.BUDGET_HOTEL:
                body["starRating"] = [1, 2, 3]
            payload = object_payload(
                await self._client.post_json(
                    "https://api.liteapi.travel/v3.0/hotels/rates", headers=headers, json=body
                )
            )
            if payload.get("data") and payload.get("sandbox") is not False:
                return ProviderResult.unavailable(
                    self.name,
                    "LiteAPI response did not confirm production mode; no live quotes shown.",
                )
            if "error" in payload:
                return ProviderResult.unavailable(self.name, "LiteAPI returned a search error.")
            hotels = {h["id"]: h for h in records(payload.get("hotels", []))}
            offers: list[AccommodationOffer] = []
            rejected = 0
            for hotel in records(payload.get("data", [])):
                for room in records(hotel.get("roomTypes", [])):
                    try:
                        offers.append(_offer(room, hotels[hotel["hotelId"]], query, centre))
                    except (KeyError, TypeError, ValueError, ArithmeticError):
                        rejected += 1
            warnings = [
                "Hotel quote is for one room, all adults and the full stay. It includes disclosed "
                "mandatory charges, including pay-at-property fees. "
                "Availability and rates can change.",
                *location_result.warnings,
            ]
            if query.accommodation_type is AccommodationType.BUDGET_HOTEL:
                warnings.append(
                    "Budget hotel search uses hotels rated 1-3 stars; "
                    "your price ceiling is applied separately."
                )
            if rejected:
                warnings.append(
                    f"{rejected} hotel quote(s) rejected: incomplete metadata, "
                    "occupancy or currency/fee mismatch."
                )
            return ProviderResult(offers, DataOrigin.LIVE, self.name, warnings=warnings)
        except Exception as exc:
            return ProviderResult.unavailable(self.name, failure_message(self.name, exc))

    async def _type_ids(self, kind: AccommodationType, headers: dict[str, str]) -> list[int]:
        if kind is AccommodationType.ANY:
            return []
        payload = object_payload(
            await self._client.get_json(
                "https://api.liteapi.travel/v3.0/data/hotelTypes", headers=headers
            )
        )
        ids = [
            int(t["id"])
            for t in records(payload["data"])
            if str(t["name"]).casefold() in _TYPE_NAMES[kind]
        ]
        if not ids:
            raise ValueError("provider cannot filter this accommodation type")
        return ids


def _amount(value: dict[str, Any], currency: Currency) -> Decimal:
    if value["currency"] != currency.value:
        raise ValueError("currency mismatch; no implicit currency conversion")
    amount = Decimal(str(value["amount"]))
    if not amount.is_finite() or amount < 0:
        raise ValueError("invalid amount")
    return amount


def _offer(
    room: dict[str, Any], hotel: dict[str, Any], query: AccommodationQuery, centre: GeoPoint
) -> AccommodationOffer:
    rates = records(room["rates"])
    # We requested one room containing all adults, not one room per traveller.
    if (
        len(rates) != 1
        or rates[0]["adultCount"] != query.guests
        or rates[0].get("childCount", 0) != 0
    ):
        raise ValueError("occupancy mismatch")
    retail = object_payload(rates[0]["retailRate"])
    totals = records(retail["total"])
    if len(totals) != 1:
        raise ValueError("ambiguous rate total")
    total = _amount(totals[0], query.currency)
    for fee in records(retail.get("taxesAndFees") or []):
        if fee.get("included") is False:
            total += _amount(fee, query.currency)
        elif fee.get("included") is not True:
            raise ValueError("unknown fee inclusion")
    point = None
    if hotel.get("latitude") is not None and hotel.get("longitude") is not None:
        point = GeoPoint(latitude=hotel["latitude"], longitude=hotel["longitude"])
    # LiteAPI offer tokens can exceed 1KB. Keep a stable short identifier, not a booking token.
    offer_id = "liteapi-" + hashlib.sha256(room["offerId"].encode()).hexdigest()[:32]
    return AccommodationOffer(
        offer_id=offer_id,
        name=hotel["name"],
        accommodation_type=query.accommodation_type,
        total_price=Money.of(total, query.currency),
        price_per_night=Money.of(total / query.nights, query.currency),
        nights=query.nights,
        guests=query.guests,
        location=point,
        address=hotel.get("address"),
        rating=hotel.get("rating"),
        review_count=hotel.get("review_count"),
        distance_to_centre_km=distance_km(centre, point) if point is not None else None,
        # No verified per-feature accessibility or timezone-safe cancellation cutoff in
        # this adapter: leave unknown rather than claiming accessibility/free cancellation.
        provenance=Provenance.live(
            "LiteAPI Hotels", source_id=hotel["id"], source_url="https://liteapi.travel"
        ),
    )
