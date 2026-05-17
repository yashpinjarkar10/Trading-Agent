"""
GDACS (Global Disaster Alert and Coordination System) scraper.

GDACS is the UN/EU real-time alerting system for natural disasters with
potential humanitarian impact. It pre-filters to events above a minimum
impact threshold — so unlike USGS (every M2.5+ quake) or FIRMS (every
pixel above FRP threshold), every GDACS event is already significant.

Feed: https://www.gdacs.org/xml/rss.xml   (no API key, no registration)

GDACS alert levels → our severity:
    Green  → 3  (noteworthy but low humanitarian impact)
    Orange → 6  (medium impact, regional concern)
    Red    → 9  (high impact, international response likely)
    +1 if > 1,000 people affected; +1 more if > 100,000 people affected

Cross-source dedup: GDACS EQ events typically appear in USGS within the
same 50km / 6h window, so the upsert_event RPC should produce
`action='merged'` for those rows. GDACS WF overlaps with FIRMS.

ZERO LLM calls.

Documentation: https://www.gdacs.org/xml/rss_user_guide.pdf
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import httpx

from ..common.models import EventCategory, NormalizedEvent

logger = logging.getLogger(__name__)

SOURCE_NAME = "gdacs"
GDACS_RSS_URL = "https://www.gdacs.org/xml/rss.xml"
REQUEST_TIMEOUT_S = 30

# XML namespaces used in GDACS RSS
NS = {
    "dc":     "http://purl.org/dc/elements/1.1/",
    "gdacs":  "http://www.gdacs.org",
    "geo":    "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "georss": "http://www.georss.org/georss",
}

# GDACS event type codes → our EventMap categories
_GDACS_TYPE_TO_CATEGORY: dict[str, EventCategory] = {
    "EQ": EventCategory.disaster_nat,  # Earthquake
    "TC": EventCategory.disaster_nat,  # Tropical Cyclone
    "FL": EventCategory.disaster_nat,  # Flood
    "VO": EventCategory.disaster_nat,  # Volcano
    "WF": EventCategory.disaster_nat,  # Wildfire
    "DR": EventCategory.disaster_nat,  # Drought
    "TS": EventCategory.disaster_nat,  # Tsunami
    "SS": EventCategory.disaster_nat,  # Storm Surge
}

# Base severity per alert level
_ALERT_SEVERITY: dict[str, int] = {
    "green":  3,
    "orange": 6,
    "red":    9,
}

# Subcategory strings per event type
_GDACS_TYPE_SUBCATEGORY: dict[str, str] = {
    "EQ": "earthquake",
    "TC": "tropical_cyclone",
    "FL": "flood",
    "VO": "volcano",
    "WF": "wildfire",
    "DR": "drought",
    "TS": "tsunami",
    "SS": "storm_surge",
}


def fetch_events() -> list[NormalizedEvent]:
    logger.info("GDACS: fetching %s", GDACS_RSS_URL)
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        r = client.get(
            GDACS_RSS_URL,
            headers={"User-Agent": "trading-agent-event-map/1.0"},
            follow_redirects=True,
        )
        r.raise_for_status()

    root = ET.fromstring(r.content)
    items = root.findall(".//item")
    logger.info("GDACS: feed contains %d items", len(items))

    events: list[NormalizedEvent] = []
    skipped = 0
    for item in items:
        try:
            ev = _normalize(item)
            if ev is None:
                skipped += 1
                continue
            events.append(ev)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            logger.debug("GDACS: normalize failed for %s: %s", _text(item, "guid"), e)

    logger.info(
        "GDACS: normalized %d / %d (skipped=%d)",
        len(events), len(items), skipped,
    )
    return events


def _normalize(item: ET.Element) -> NormalizedEvent | None:
    guid = _text(item, "guid") or ""
    if not guid:
        return None

    # Coordinates
    lat_s = _text(item, "geo:Point/geo:lat", NS) or _text(item, "georss:point", NS, split=0)
    lng_s = _text(item, "geo:Point/geo:long", NS) or _text(item, "georss:point", NS, split=1)
    if not lat_s or not lng_s:
        return None
    try:
        lat, lng = float(lat_s), float(lng_s)
    except ValueError:
        return None
    if lat == 0.0 and lng == 0.0:
        return None

    # Alert level + event type
    alert_raw = (_text(item, "gdacs:alertlevel", NS) or "green").lower()
    event_type = (_text(item, "gdacs:eventtype", NS) or "").upper()
    category = _GDACS_TYPE_TO_CATEGORY.get(event_type, EventCategory.disaster_nat)
    subcategory = _GDACS_TYPE_SUBCATEGORY.get(event_type)

    # Severity calculation
    base_sev = _ALERT_SEVERITY.get(alert_raw, 3)
    pop_val = 0
    pop_el = item.find("gdacs:population", NS)
    if pop_el is not None:
        try:
            pop_val = int(pop_el.attrib.get("value", 0) or 0)
        except (ValueError, TypeError):
            pass
    if pop_val > 100_000:
        base_sev = min(10, base_sev + 2)
    elif pop_val > 1_000:
        base_sev = min(10, base_sev + 1)

    # Dates
    pub_date = _text(item, "pubDate")
    occurred_at = _parse_rfc2822(pub_date) if pub_date else datetime.now(timezone.utc)

    # Location
    country = _text(item, "gdacs:country", NS) or ""
    iso3 = (_text(item, "gdacs:iso3", NS) or "").upper() or None

    # Title / description
    title = _text(item, "title") or f"{event_type} alert in {country}"
    severity_text = _text(item, "gdacs:severity", NS) or ""
    population_text = _text(item, "gdacs:population", NS) or ""
    description_parts = [p for p in [severity_text, population_text] if p]
    description = " — ".join(description_parts) or None

    link = _text(item, "link") or f"https://www.gdacs.org"

    return NormalizedEvent(
        source=SOURCE_NAME,
        source_event_id=guid,           # e.g. "EQ1406430" or "WF1028685"
        source_url=link,
        title=title[:200],
        description=description,
        category=category,
        subcategory=subcategory,
        location_name=country or None,
        country_iso2=None,              # ISO3 provided but not ISO2; enricher fills later
        lat=lat,
        lng=lng,
        occurred_at=occurred_at,
        severity_hint=base_sev,
        raw={
            "guid":          guid,
            "event_type":    event_type,
            "alert_level":   alert_raw,
            "alert_score":   _text(item, "gdacs:alertscore", NS),
            "population":    pop_val,
            "iso3":          iso3,
            "country":       country,
            "severity_text": severity_text,
        },
    )


# ── helpers ───────────────────────────────────────────────────────────────────
def _text(
    el: ET.Element,
    tag: str,
    ns: dict | None = None,
    split: int | None = None,
) -> str | None:
    """Return stripped text of a child element, or None if not found."""
    found = el.find(tag, ns or {})
    if found is None or found.text is None:
        return None
    text = found.text.strip()
    if split is not None:
        parts = text.split()
        return parts[split] if split < len(parts) else None
    return text or None


def _parse_rfc2822(s: str) -> datetime:
    """Parse RFC-2822 date like 'Sat, 16 May 2026 11:00:12 GMT'."""
    try:
        dt = parsedate_to_datetime(s)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)
