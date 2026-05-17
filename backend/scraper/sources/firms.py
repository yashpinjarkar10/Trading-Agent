"""
NASA FIRMS (Fire Information for Resource Management System) scraper.

We pull near-real-time active-fire detections from VIIRS_SNPP_NRT (global,
last 1 day), filter to medium+ FRP, and bucket pixels by a 0.1° geographic
grid + day so that the ~hundreds of satellite pixels making up a single
wildfire collapse into one row via the upsert_event RPC's Layer-1 dedup.

Source signup (free, ~60 seconds):
    https://firms.modaps.eosdis.nasa.gov/api/area/

API docs:
    https://firms.modaps.eosdis.nasa.gov/api/area/

URL pattern:
    https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/world/{DAYS}

ZERO LLM calls. Severity from `severity_from_fire_frp()` in enrichment.py.

Output bucket id (used as source_event_id):
    firms-{lat_bin:.1f}_{lng_bin:.1f}_{acq_date}      (≈10×10 km × 1 day)
"""
from __future__ import annotations

import csv
import io
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone

import httpx

from ..common.enrichment import severity_from_fire_frp
from ..common.models import EventCategory, NormalizedEvent

logger = logging.getLogger(__name__)

SOURCE_NAME = "firms"
FIRMS_HOST = "https://firms.modaps.eosdis.nasa.gov"
FIRMS_FEED = "VIIRS_SNPP_NRT"  # also valid: VIIRS_NOAA20_NRT, VIIRS_NOAA21_NRT, MODIS_NRT
FIRMS_DAYS_BACK = 1            # last 24h; cron runs every 30 min, so plenty of overlap
REQUEST_TIMEOUT_S = 60

# Spatial bucket size (degrees). 0.1° ≈ 11 km at the equator. Pixels within
# the same bucket on the same UTC day are treated as the same fire event.
BUCKET_DEG = 0.1

# Minimum FRP (Megawatts) — anything smaller is too small to matter for the
# Event Map (mostly agricultural burns / small brush). severity_from_fire_frp
# also enforces this internally; we filter early to avoid bucketing churn.
MIN_FRP_MW = 5.0

# Drop low-confidence detections (VIIRS confidence is l/n/h).
MIN_CONFIDENCE = {"n", "h"}


def fetch_events() -> list[NormalizedEvent]:
    map_key = os.getenv("NASA_FIRMS_API_KEY", "").strip()
    if not map_key:
        raise RuntimeError(
            "NASA_FIRMS_API_KEY missing. Get one (free, 60s) at "
            "https://firms.modaps.eosdis.nasa.gov/api/area/ and add it to "
            "backend/.env or as a GitHub Actions repo secret."
        )

    url = f"{FIRMS_HOST}/api/area/csv/{map_key}/{FIRMS_FEED}/world/{FIRMS_DAYS_BACK}"
    # Use the placeholder in the log so we never leak the key
    logger.info(
        "FIRMS: fetching %s/api/area/csv/<KEY>/%s/world/%d",
        FIRMS_HOST, FIRMS_FEED, FIRMS_DAYS_BACK,
    )

    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        r = client.get(url)
        r.raise_for_status()
    text = r.text
    logger.info("FIRMS: CSV size=%d bytes", len(text))

    # Quick guard: API errors come back as plain text (not CSV) e.g.
    # "Invalid MAP_KEY". DictReader against that would silently produce zero rows.
    if not text.strip() or "latitude" not in text.split("\n", 1)[0].lower():
        snippet = text[:300].replace("\n", " | ")
        raise RuntimeError(f"FIRMS API did not return CSV. First chars: {snippet!r}")

    pixels = _parse_pixels(text)
    logger.info("FIRMS: kept %d pixels after FRP/confidence filters", len(pixels))

    events = _bucket_to_events(pixels)
    logger.info(
        "FIRMS: aggregated %d pixels → %d 0.1° bucket-events",
        len(pixels), len(events),
    )
    return events


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────
def _parse_pixels(text: str) -> list[dict]:
    """One CSV row → one dict, after FRP and confidence filters."""
    out: list[dict] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            frp = float(row.get("frp", "") or 0)
        except ValueError:
            continue
        if frp < MIN_FRP_MW:
            continue

        conf = (row.get("confidence", "") or "").strip().lower()
        # VIIRS uses l/n/h; MODIS uses 0–100. Treat numeric ≥ 30 as nominal.
        if conf and conf in {"l", "n", "h"}:
            if conf not in MIN_CONFIDENCE:
                continue
        else:
            try:
                if int(conf) < 30:
                    continue
            except (ValueError, TypeError):
                # Unknown confidence format — keep it
                pass

        try:
            lat = float(row["latitude"])
            lng = float(row["longitude"])
        except (KeyError, ValueError):
            continue

        acq_date = (row.get("acq_date") or "").strip()
        acq_time = (row.get("acq_time") or "").strip().zfill(4)
        if not acq_date or not acq_time:
            continue

        try:
            occurred_at = datetime.strptime(
                f"{acq_date} {acq_time}", "%Y-%m-%d %H%M"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        out.append({
            "lat": lat,
            "lng": lng,
            "frp": frp,
            "acq_date": acq_date,
            "occurred_at": occurred_at,
            "satellite": (row.get("satellite") or "").strip(),
            "instrument": (row.get("instrument") or "").strip(),
            "daynight": (row.get("daynight") or "").strip(),
            "confidence": conf,
            "bright_ti4": _try_float(row.get("bright_ti4")),
            "bright_ti5": _try_float(row.get("bright_ti5")),
        })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Bucketing — collapse pixels of the same fire to one event
# ──────────────────────────────────────────────────────────────────────────────
def _bucket_key(lat: float, lng: float, acq_date: str) -> tuple[float, float, str]:
    return (
        round(lat / BUCKET_DEG) * BUCKET_DEG,
        round(lng / BUCKET_DEG) * BUCKET_DEG,
        acq_date,
    )


def _bucket_to_events(pixels: list[dict]) -> list[NormalizedEvent]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for p in pixels:
        grouped[_bucket_key(p["lat"], p["lng"], p["acq_date"])].append(p)

    events: list[NormalizedEvent] = []
    for (lat_bin, lng_bin, acq_date), group in grouped.items():
        events.append(_aggregate_bucket(lat_bin, lng_bin, acq_date, group))
    return events


def _aggregate_bucket(
    lat_bin: float,
    lng_bin: float,
    acq_date: str,
    group: list[dict],
) -> NormalizedEvent:
    """Reduce N pixels in one (bucket, day) to one NormalizedEvent."""
    n = len(group)
    max_frp = max(p["frp"] for p in group)
    sum_frp = sum(p["frp"] for p in group)
    # Weighted-average centroid (FRP-weighted) so we point at the hottest pixel
    cx = sum(p["lng"] * p["frp"] for p in group) / sum_frp
    cy = sum(p["lat"] * p["frp"] for p in group) / sum_frp
    # Latest occurred_at in the bucket
    occurred_at = max(p["occurred_at"] for p in group)

    severity = severity_from_fire_frp(max_frp)
    # Bucket id is stable across runs → Layer-1 UPDATE on subsequent polls
    bucket_id = f"firms-{lat_bin:+.1f}_{lng_bin:+.1f}_{acq_date}"

    title = (
        f"Active wildfire ({n} pixel{'s' if n > 1 else ''}, "
        f"max FRP {max_frp:.0f} MW)"
    )
    description = (
        f"{n} satellite detection{'s' if n > 1 else ''} on {acq_date}. "
        f"Total FRP {sum_frp:.0f} MW, max {max_frp:.0f} MW. "
        f"Source feed: {FIRMS_FEED}."
    )

    return NormalizedEvent(
        source=SOURCE_NAME,
        source_event_id=bucket_id,
        source_url=f"{FIRMS_HOST}/map/#d:24hrs;@{cy:.3f},{cx:.3f},6z",
        title=title,
        description=description,
        category=EventCategory.disaster_nat,
        subcategory="wildfire",
        location_name=None,  # filled later by reverse-geocoder enrichment
        country_iso2=None,
        lat=cy,
        lng=cx,
        occurred_at=occurred_at,
        severity_hint=severity,
        raw={
            "feed": FIRMS_FEED,
            "bucket_deg": BUCKET_DEG,
            "lat_bin": lat_bin,
            "lng_bin": lng_bin,
            "acq_date": acq_date,
            "pixel_count": n,
            "max_frp_mw": max_frp,
            "sum_frp_mw": sum_frp,
            "satellite": group[0]["satellite"],
            "instrument": group[0]["instrument"],
        },
    )


def _try_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None
