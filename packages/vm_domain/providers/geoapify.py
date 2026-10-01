"""Live place discovery; Geoapify supplies locations, not verified ticket prices.

https://apidocs.geoapify.com/docs/places/
"""

from __future__ import annotations

import hashlib
from typing import Any

from vm_config.secrets import try_read_secret_file
from vm_contracts.common import DataOrigin, GeoPoint, Provenance
from vm_contracts.destination import Attraction, AttractionCategory
from vm_domain.providers.base import POIQuery, ProviderResult
from vm_domain.providers.live_support import failure_message, object_payload, records
from vm_domain.providers.open_meteo import OpenMeteoGeocodingProvider
from vm_resilience.safe_http import SafeHTTPClient

# Provider category -> domain category and user-facing interest tags.
_CATEGORIES = {
    "entertainment.museum": (AttractionCategory.MUSEUM, ["museum", "history", "art"]),
    "entertainment.culture": (AttractionCategory.ENTERTAINMENT, ["culture", "art"]),
    "leisure.park": (AttractionCategory.PARK, ["park", "nature", "outdoors"]),
    "tourism.sights": (AttractionCategory.HISTORIC, ["history", "architecture", "sightseeing"]),
    "tourism.attraction": (AttractionCategory.OTHER, ["sightseeing"]),
    "catering.restaurant": (AttractionCategory.FOOD, ["food", "local food"]),
}


class GeoapifyPOIProvider:
    name = "Geoapify Places"

    def __init__(self, client: SafeHTTPClient, *, api_key_file: str | None) -> None:
        self._client = client
        self._key_file = api_key_file

    async def search(self, query: POIQuery) -> ProviderResult[Attraction]:
        key = try_read_secret_file(self._key_file, name=self.name)
        if key is None:
            return ProviderResult.unavailable(
                self.name, "Geoapify: set PROVIDER_GEOAPIFY_API_KEY_FILE to a readable key file."
            )
        try:
            location = query.location
            if location is None:
                result = await OpenMeteoGeocodingProvider(self._client).geocode(query.destination)
                if not result.ok or not result.items:
                    return ProviderResult.unavailable(
                        self.name, "Could not resolve destination for live place discovery."
                    )
                location = result.items[0]
            selected = [
                key
                for key, (_, tags) in _CATEGORIES.items()
                if any(i.lower() in tags for i in query.interests)
            ]
            categories = selected or [
                "tourism.sights",
                "tourism.attraction",
                "entertainment.museum",
                "leisure.park",
            ]
            payload = object_payload(
                await self._client.post_json(
                    "https://api.geoapify.com/v2/places",
                    headers={"x-api-key": key.get_secret_value()},
                    json={
                        "categories": categories,
                        "limit": min(query.limit, 100),
                        "lang": "en",
                        "filter": {
                            "type": "circle",
                            "lon": location.longitude,
                            "lat": location.latitude,
                            "radius": query.radius_metres,
                        },
                    },
                )
            )
            attractions: list[Attraction] = []
            seen: set[str] = set()
            rejected = 0
            for feature in records(payload["features"]):
                try:
                    attraction = _attraction(object_payload(feature["properties"]))
                    if attraction.attraction_id not in seen:
                        attractions.append(attraction)
                        seen.add(attraction.attraction_id)
                except (KeyError, ValueError, TypeError):
                    rejected += 1
            warnings = [
                "Places: Geoapify / OpenStreetMap contributors (ODbL). "
                "https://www.geoapify.com/ | https://www.openstreetmap.org/copyright",
                "Live place listings do not verify admission prices, "
                "opening hours or accessibility. "
                "Scheduled visit durations are planning estimates; check venue details.",
            ]
            if rejected:
                warnings.append(f"{rejected} incomplete place record(s) skipped.")
            return ProviderResult(attractions, DataOrigin.LIVE, self.name, warnings=warnings)
        except Exception as exc:
            return ProviderResult.unavailable(self.name, failure_message(self.name, exc))


def _attraction(properties: dict[str, Any]) -> Attraction:
    categories = properties["categories"]
    if not isinstance(categories, list):
        raise ValueError("invalid place categories")
    category, tags = next(
        (
            value
            for key, value in _CATEGORIES.items()
            if any(isinstance(c, str) and (c == key or c.startswith(key + ".")) for c in categories)
        ),
        (AttractionCategory.OTHER, ["sightseeing"]),
    )
    identity = "geoapify-" + hashlib.sha256(properties["place_id"].encode()).hexdigest()[:32]
    return Attraction(
        attraction_id=identity,
        name=properties["name"],
        category=category,
        location=GeoPoint(latitude=properties["lat"], longitude=properties["lon"]),
        is_outdoor=category is AttractionCategory.PARK,
        interest_tags=tags,
        # Fee information is unknown, so neither admission_price nor is_free is asserted.
        provenance=Provenance.live(
            "Geoapify Places",
            source_id=identity,
            source_url="https://www.geoapify.com/",
            license_note="Geoapify / OpenStreetMap contributors; ODbL. https://www.openstreetmap.org/copyright",
        ),
    )
