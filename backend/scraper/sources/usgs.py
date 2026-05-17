"""
USGS Earthquake Hazards Program — earthquakes scraper.

Source feed (no API key, no rate limit on this URL):
    https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson

We deliberately use the **M2.5+ day feed** (last 24h of magnitude ≥ 2.5
quakes) rather than `all_hour.geojson` for two reasons:
  1. Always has data — earthquakes don't politely batch into "the last 60
     minutes", and a quiet hour shouldn't mean an empty scrape run.
  2. Built-in 24h auto-backfill — if a GH Actions run misfires or the
     runner takes too long, the next run still catches every missed event
     because the feed window is 24h. Layer-1 dedup (within-source UPSERT
     in the upsert_event RPC) makes the repeated rows essentially free
     (~300 cheap UPDATEs / hour).

GH Actions cron: */5 * * * * (every 5 min, GitHub's minimum interval).
ZERO LLM calls — severity comes from `severity_from_magnitude()`.

Source documentation:
    https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from ..common.enrichment import severity_from_magnitude
from ..common.models import EventCategory, NormalizedEvent

logger = logging.getLogger(__name__)

USGS_FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
SOURCE_NAME = "usgs"
REQUEST_TIMEOUT_S = 30


def fetch_events() -> list[NormalizedEvent]:
    """Pull the last-hour USGS feed and return normalized events ready to upsert."""
    logger.info("USGS: fetching %s", USGS_FEED_URL)
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        r = client.get(
            USGS_FEED_URL,
            headers={"User-Agent": "trading-agent-event-map/1.0 (https://github.com)"},
        )
        r.raise_for_status()
        data = r.json()

    features = data.get("features", [])
    metadata = data.get("metadata", {})
    logger.info(
        "USGS: feed returned %d features (generated=%s)",
        len(features),
        metadata.get("generated"),
    )

    events: list[NormalizedEvent] = []
    dropped_low_mag = 0
    dropped_no_geom = 0

    for feat in features:
        try:
            normalized = _normalize(feat)
            if normalized is None:
                dropped_low_mag += 1
                continue
            events.append(normalized)
        except _NoGeometryError:
            dropped_no_geom += 1
            continue
        except Exception as e:
            logger.warning(
                "USGS: skip feature %s — %s: %s",
                feat.get("id"), type(e).__name__, e,
            )

    logger.info(
        "USGS: normalized %d / %d (dropped low-mag=%d, no-geom=%d)",
        len(events), len(features), dropped_low_mag, dropped_no_geom,
    )
    return events


class _NoGeometryError(Exception):
    pass


def _normalize(feat: dict) -> NormalizedEvent | None:
    """USGS GeoJSON feature → NormalizedEvent (or None if below threshold)."""
    feat_id = feat.get("id")
    if not feat_id:
        raise ValueError("missing feature id")

    props = feat.get("properties") or {}
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates") or []
    if len(coords) < 2 or coords[0] is None or coords[1] is None:
        raise _NoGeometryError()
    lng, lat = float(coords[0]), float(coords[1])
    # coords[2] is depth in km (optional)

    mag = props.get("mag")
    mag = float(mag) if mag is not None else None
    severity = severity_from_magnitude(mag)
    if severity is None:
        return None  # below M2.5 — skip

    place = (props.get("place") or "").strip()
    title = f"M{mag:.1f} earthquake" + (f" — {place}" if place else "")

    # time/updated are epoch ms UTC
    time_ms = props.get("time")
    if time_ms is None:
        raise ValueError("missing time")
    occurred_at = datetime.fromtimestamp(time_ms / 1000.0, tz=timezone.utc)

    # USGS's `code` is the geographic event id; `ids` is comma-delimited.
    # We use feat["id"] as the canonical source_event_id (e.g. "us7000mxyz").
    return NormalizedEvent(
        source=SOURCE_NAME,
        source_event_id=feat_id,
        source_url=props.get("url"),
        title=title,
        description=place or None,
        category=EventCategory.disaster_nat,
        subcategory="earthquake",
        location_name=place or None,
        country_iso2=None,  # populated later by an enrichment pass
        lat=lat,
        lng=lng,
        occurred_at=occurred_at,
        severity_hint=severity,
        raw=feat,
    )
