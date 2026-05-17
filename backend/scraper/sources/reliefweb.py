"""
ReliefWeb Disasters API scraper.

ReliefWeb is OCHA's (UN Office for the Coordination of Humanitarian Affairs)
curated disaster database. Each entry is a verified, human-reviewed disaster
event with standardised fields: type, country, status, dates, and coordinates.

API docs: https://apidoc.rwlabs.org/
No API key required. Rate limit: 1 000 req/min (we use 1 req/run, paginated).

URL: https://api.reliefweb.int/v1/disasters

This is the source most likely to trigger Layer-2 cross-source dedup with USGS,
because major earthquakes, floods and tsunamis show up in both ReliefWeb
(humanitarian declaration) and USGS (seismic detection) within 50km / 6h /
same category (disaster_nat). When that happens the RPC returns 'merged'.

ZERO LLM calls — category from `enrichment.severity_from_disaster_type()`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from ..common.enrichment import severity_from_disaster_type
from ..common.models import EventCategory, NormalizedEvent

logger = logging.getLogger(__name__)

SOURCE_NAME = "reliefweb"
RELIEFWEB_API_URL = "https://api.reliefweb.int/v1/disasters"
REQUEST_TIMEOUT_S = 30

# How many days back to look.  20-min cron + 2d window = any missed run
# still caught on next run with no gaps.
DAYS_BACK = 2

# Maximum pages to fetch per run (100 items/page).
MAX_PAGES = 5

# ReliefWeb disaster types → our EventCategory
_TYPE_TO_CATEGORY: dict[str, EventCategory] = {
    "Earthquake":            EventCategory.disaster_nat,
    "Tsunami":               EventCategory.disaster_nat,
    "Volcano":               EventCategory.disaster_nat,
    "Flood":                 EventCategory.disaster_nat,
    "Flash Flood":           EventCategory.disaster_nat,
    "Tropical Cyclone":      EventCategory.disaster_nat,
    "Storm":                 EventCategory.disaster_nat,
    "Cold Wave":             EventCategory.disaster_nat,
    "Heat Wave":             EventCategory.disaster_nat,
    "Drought":               EventCategory.disaster_nat,
    "Wildfire":              EventCategory.disaster_nat,
    "Landslide":             EventCategory.disaster_nat,
    "Avalanche":             EventCategory.disaster_nat,
    "Mud Slide":             EventCategory.disaster_nat,
    "Epidemic":              EventCategory.health,
    "Plague":                EventCategory.health,
    "Insect Infestation":    EventCategory.disaster_nat,
    "Famine":                EventCategory.disaster_hum,
    "Civil Unrest":          EventCategory.conflict,
    "War":                   EventCategory.conflict,
    "Industrial Accident":   EventCategory.disaster_hum,
    "Transport Accident":    EventCategory.disaster_hum,
    "Other":                 EventCategory.other,
}

_DEFAULT_CATEGORY = EventCategory.other


def fetch_events() -> list[NormalizedEvent]:
    """Pull recent ReliefWeb disaster records and return NormalizedEvents."""
    payload = _build_payload()
    all_items: list[dict] = []

    offset = 0
    limit = 100
    for page in range(MAX_PAGES):
        payload["offset"] = offset
        payload["limit"] = limit

        logger.info("ReliefWeb: fetching page %d (offset=%d)", page + 1, offset)
        with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
            r = client.post(RELIEFWEB_API_URL, json=payload)
            r.raise_for_status()
        data = r.json()

        items = data.get("data", [])
        total = data.get("totalCount", 0)
        logger.info("ReliefWeb: page %d → %d items (total=%d)", page + 1, len(items), total)
        all_items.extend(items)

        if len(all_items) >= total or not items:
            break
        offset += limit

    events: list[NormalizedEvent] = []
    skipped = 0
    for item in all_items:
        try:
            ev = _normalize(item)
            if ev is None:
                skipped += 1
                continue
            events.append(ev)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            logger.debug("ReliefWeb: normalize failed for id=%s: %s", item.get("id"), e)

    logger.info(
        "ReliefWeb: normalized %d / %d (skipped=%d)",
        len(events), len(all_items), skipped,
    )
    return events


def _build_payload() -> dict:
    """ReliefWeb API POST body for recent disasters with coordinates."""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return {
        "filter": {
            "operator": "AND",
            "conditions": [
                {"field": "date.created", "value": cutoff, "operator": ">="},
            ],
        },
        "fields": {
            "include": [
                "id", "name", "status", "date",
                "type",       # list of {id, name}
                "country",    # list of {iso3, name, location:{lat, lon}}
                "primary_country",
                "glide",
            ],
        },
        "sort": ["date.created:desc"],
    }


def _normalize(item: dict) -> NormalizedEvent | None:
    fields = item.get("fields", {})
    dis_id = str(item.get("id", ""))
    if not dis_id:
        return None

    name = (fields.get("name") or "").strip()
    status = (fields.get("status") or "").strip()
    glide = (fields.get("glide") or "").strip()

    # Type list — take the first one
    types_raw = fields.get("type") or []
    primary_type = types_raw[0].get("name", "") if types_raw else ""
    category = _TYPE_TO_CATEGORY.get(primary_type, _DEFAULT_CATEGORY)

    # Date — try date.event first, then date.created
    date_block = fields.get("date") or {}
    date_str = date_block.get("event") or date_block.get("created") or ""
    if not date_str:
        return None
    occurred_at = _parse_date(date_str)
    if occurred_at is None:
        return None

    # Coordinates — from primary_country.location or first country.location
    lat, lng = _extract_coords(fields)
    if lat is None or lng is None:
        return None  # can't place on map without coordinates

    # Country ISO2 — reliefweb gives ISO3, try to get from country list
    country_iso2 = _extract_iso2(fields)
    location_name = _extract_location_name(fields)

    severity = severity_from_disaster_type(primary_type, status)

    # Build a clean title
    title = name or f"{primary_type} disaster"
    if location_name and location_name.lower() not in title.lower():
        title = f"{title} — {location_name}"

    description = primary_type
    if glide:
        description += f" (GLIDE: {glide})"
    if status:
        description += f". Status: {status}."

    source_url = f"https://reliefweb.int/disaster/{dis_id}"

    return NormalizedEvent(
        source=SOURCE_NAME,
        source_event_id=dis_id,
        source_url=source_url,
        title=title[:200],
        description=description,
        category=category,
        subcategory=primary_type.lower().replace(" ", "_") if primary_type else None,
        location_name=location_name,
        country_iso2=country_iso2,
        lat=lat,
        lng=lng,
        occurred_at=occurred_at,
        severity_hint=severity,
        raw={
            "reliefweb_id": dis_id,
            "name": name,
            "status": status,
            "glide": glide,
            "primary_type": primary_type,
            "all_types": [t.get("name") for t in types_raw],
        },
    )


def _extract_coords(fields: dict) -> tuple[float | None, float | None]:
    """Try primary_country.location then first country.location."""
    pc = fields.get("primary_country") or {}
    loc = pc.get("location") or {}
    if loc.get("lat") is not None and loc.get("lon") is not None:
        return float(loc["lat"]), float(loc["lon"])
    for c in fields.get("country") or []:
        loc = c.get("location") or {}
        if loc.get("lat") is not None and loc.get("lon") is not None:
            return float(loc["lat"]), float(loc["lon"])
    return None, None


def _extract_iso2(fields: dict) -> str | None:
    """ReliefWeb gives iso3 not iso2. Best we can do is grab it from the data."""
    pc = fields.get("primary_country") or {}
    iso3 = pc.get("iso3") or ""
    if iso3:
        return None  # we only store iso2; leave null for now (enricher can fill later)
    return None


def _extract_location_name(fields: dict) -> str | None:
    pc = fields.get("primary_country") or {}
    name = (pc.get("name") or "").strip()
    return name or None


def _parse_date(s: str) -> datetime | None:
    """ReliefWeb dates: '2024-03-05T12:00:00+00:00' or 'YYYY-MM-DD'."""
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s[:19], fmt[:len(fmt)])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None
