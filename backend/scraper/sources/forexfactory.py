"""
Forex Factory calendar scraper.

Source feed:
    https://nfs.faireconomy.media/ff_calendar_thisweek.json

This scraper retrieves only high-impact (red folder) macro-economic events
from the Forex Factory calendar, resolves their location using a hardcoded
currency-to-capital mapping, and prepares them for Supabase ingestion.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from ..common.models import EventCategory, NormalizedEvent

logger = logging.getLogger(__name__)

SOURCE_NAME = "forexfactory"
FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
REQUEST_TIMEOUT_S = 30

# Mappings of Forex Factory currency codes to their respective capital cities and coordinates
# for plotting on the 3D globe.
CURRENCY_TO_CAPITAL: dict[str, dict[str, Any]] = {
    "USD": {"country_iso2": "US", "location_name": "Washington, D.C., USA", "lat": 38.9072, "lng": -77.0369},
    "EUR": {"country_iso2": "BE", "location_name": "Brussels, Belgium", "lat": 50.8503, "lng": 4.3517},
    "GBP": {"country_iso2": "GB", "location_name": "London, UK", "lat": 51.5074, "lng": -0.1278},
    "JPY": {"country_iso2": "JP", "location_name": "Tokyo, Japan", "lat": 35.6762, "lng": 139.6503},
    "AUD": {"country_iso2": "AU", "location_name": "Canberra, Australia", "lat": -35.2809, "lng": 149.1300},
    "NZD": {"country_iso2": "NZ", "location_name": "Wellington, New Zealand", "lat": -41.2865, "lng": 174.7762},
    "CAD": {"country_iso2": "CA", "location_name": "Ottawa, Canada", "lat": 45.4215, "lng": -75.6972},
    "CHF": {"country_iso2": "CH", "location_name": "Bern, Switzerland", "lat": 46.9480, "lng": 7.4474},
    "CNY": {"country_iso2": "CN", "location_name": "Beijing, China", "lat": 39.9042, "lng": 116.4074},
}


def fetch_events() -> list[NormalizedEvent]:
    """Pull the Forex Factory calendar feed and return normalized high-impact events."""
    logger.info("Forex Factory: fetching %s", FEED_URL)
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        r = client.get(
            FEED_URL,
            headers={"User-Agent": "trading-agent-event-map/1.0 (https://github.com)"},
        )
        r.raise_for_status()
        data = r.json()

    if not isinstance(data, list):
        raise ValueError(f"Expected list of events from Forex Factory feed, got {type(data)}")

    logger.info("Forex Factory: feed returned %d total events", len(data))

    events: list[NormalizedEvent] = []
    skipped_non_high = 0
    skipped_unmapped_currency = 0

    for item in data:
        try:
            # We only extract high-impact events (red folder)
            impact = item.get("impact", "").strip().lower()
            if impact != "high":
                skipped_non_high += 1
                continue

            normalized = _normalize(item)
            if normalized is None:
                skipped_unmapped_currency += 1
                continue

            events.append(normalized)
        except Exception as e:
            logger.warning(
                "Forex Factory: skip event %s — %s: %s",
                item.get("title", "Unknown"), type(e).__name__, e,
            )

    logger.info(
        "Forex Factory: normalized %d / %d (skipped low-impact=%d, unmapped-currency=%d)",
        len(events), len(data), skipped_non_high, skipped_unmapped_currency,
    )
    return events


def _normalize(item: dict) -> NormalizedEvent | None:
    """Normalize a raw Forex Factory JSON event into a NormalizedEvent."""
    title = (item.get("title") or "").strip()
    if not title:
        raise ValueError("missing event title")

    currency = (item.get("country") or "").strip().upper()
    if not currency:
        raise ValueError("missing event country/currency")

    # Map currency to capital city geolocations
    geo_info = CURRENCY_TO_CAPITAL.get(currency)
    if not geo_info:
        logger.debug("Forex Factory: unmapped currency/country: %s", currency)
        return None

    # Parse occurred_at date
    date_str = item.get("date")
    if not date_str:
        raise ValueError("missing event date")

    # Parse ISO-8601 date string, e.g. 2026-07-16T08:30:00-04:00
    occurred_at = datetime.fromisoformat(date_str)
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=timezone.utc)

    # Format values for description
    forecast = (item.get("forecast") or "").strip()
    previous = (item.get("previous") or "").strip()
    actual = (item.get("actual") or "").strip()

    description = (
        f"Economic indicator for {currency}.\n"
        f"Forecast: {forecast or 'N/A'}\n"
        f"Previous: {previous or 'N/A'}\n"
        f"Actual: {actual or 'N/A'}"
    )

    # Build a stable, deterministic source_event_id using a hex hash
    unique_str = f"ff-{currency}-{title.lower()}-{occurred_at.isoformat()}"
    source_event_id = hashlib.md5(unique_str.encode("utf-8")).hexdigest()

    return NormalizedEvent(
        source=SOURCE_NAME,
        source_event_id=source_event_id,
        source_url="https://www.forexfactory.com/calendar",
        title=title[:200],
        description=description,
        category=EventCategory.economy,
        subcategory="macro_economic_indicator",
        location_name=geo_info["location_name"],
        country_iso2=geo_info["country_iso2"],
        lat=geo_info["lat"],
        lng=geo_info["lng"],
        occurred_at=occurred_at,
        severity_hint=7,  # Default high-impact macro indicator score
        raw=item,
    )
