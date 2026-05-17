"""
GDELT 2.0 Events scraper.

GDELT publishes a tab-separated events file every 15 minutes containing
~50,000 geocoded news events worldwide. We pull the most recent file via
its `lastupdate.txt` index, filter aggressively, and upsert.

Filtering keeps only "high signal" events for the Event Map:
  • EventRootCode in INGESTED_ROOT_CODES   (protest / conflict / aid / threat)
  • IsRootEvent = 1                        (GDELT's own dedup flag)
  • NumMentions >= MIN_MENTIONS            (need >=5 sources, no rumours)
  • ActionGeo_Lat / Long present and not (0,0)

This typically reduces ~50,000 raw rows → 200–500 normalized events.

ZERO LLM calls — category from CAMEO mapping, severity from rule-based
formula in enrichment.py.

References:
  https://www.gdeltproject.org/data.html#rawdatafiles
  http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf
"""
from __future__ import annotations

import csv
import io
import logging
import zipfile
from collections import Counter
from datetime import datetime, timezone

import httpx

from ..common.cameo import (
    INGESTED_ROOT_CODES,
    root_description,
    root_to_category,
)
from ..common.enrichment import severity_from_gdelt
from ..common.models import NormalizedEvent

logger = logging.getLogger(__name__)

GDELT_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
SOURCE_NAME = "gdelt"
DOWNLOAD_TIMEOUT_S = 120

# Filtering thresholds
MIN_MENTIONS = 5

# Cap to protect Supabase from a runaway batch (e.g. a global crisis spike).
# A single 15-min file shouldn't exceed this with our filters; if it does,
# we keep the highest-severity rows.
MAX_EVENTS_PER_RUN = 1000

# 0-indexed column map for the GDELT 2.0 events CSV (61 fields total).
COL_GLOBALEVENTID         = 0
COL_ACTOR1_NAME           = 6
COL_ACTOR2_NAME           = 16
COL_IS_ROOT_EVENT         = 25
COL_EVENT_CODE            = 26
COL_EVENT_ROOT_CODE       = 28
COL_QUAD_CLASS            = 29
COL_GOLDSTEIN_SCALE       = 30
COL_NUM_MENTIONS          = 31
COL_NUM_ARTICLES          = 33
COL_AVG_TONE              = 34
COL_ACTIONGEO_FULLNAME    = 52
COL_ACTIONGEO_COUNTRY     = 53
COL_ACTIONGEO_LAT         = 56
COL_ACTIONGEO_LONG        = 57
COL_DATE_ADDED            = 59
COL_SOURCE_URL            = 60
EXPECTED_COLS             = 61


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
def fetch_events() -> list[NormalizedEvent]:
    csv_url = _resolve_latest_export_url()
    logger.info("GDELT: latest export = %s", csv_url)

    raw_text = _download_unzip_csv(csv_url)
    logger.info("GDELT: decompressed CSV = %d bytes", len(raw_text))

    events = _parse_and_filter(raw_text)
    if len(events) > MAX_EVENTS_PER_RUN:
        # Keep the highest-severity ones if we're truly above the cap.
        events.sort(key=lambda e: e.severity_hint or 0, reverse=True)
        logger.warning(
            "GDELT: capping batch %d → %d (kept top severity)",
            len(events), MAX_EVENTS_PER_RUN,
        )
        events = events[:MAX_EVENTS_PER_RUN]
    return events


# ──────────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_latest_export_url() -> str:
    """Read lastupdate.txt → first line → 3rd whitespace field is the export URL."""
    with httpx.Client(timeout=30) as client:
        r = client.get(GDELT_LASTUPDATE_URL)
        r.raise_for_status()
    first = r.text.strip().split("\n")[0]
    parts = first.split()
    if len(parts) < 3 or not parts[2].startswith("http"):
        raise RuntimeError(
            f"Unexpected lastupdate.txt format (first line): {first[:200]!r}"
        )
    return parts[2]


def _download_unzip_csv(url: str) -> str:
    with httpx.Client(timeout=DOWNLOAD_TIMEOUT_S) as client:
        r = client.get(url)
        r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    csv_names = [
        n for n in zf.namelist() if n.lower().endswith(".csv") or n.lower().endswith(".tsv")
    ]
    if not csv_names:
        raise RuntimeError(f"No CSV inside {url} — namelist={zf.namelist()}")
    return zf.read(csv_names[0]).decode("utf-8", errors="replace")


def _parse_and_filter(text: str) -> list[NormalizedEvent]:
    out: list[NormalizedEvent] = []
    drops: Counter[str] = Counter()
    seen = 0

    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for row in reader:
        seen += 1

        if len(row) != EXPECTED_COLS:
            drops["col_count"] += 1
            continue

        # Cheap filters first — short-circuit before parsing
        root = row[COL_EVENT_ROOT_CODE].strip()
        if root not in INGESTED_ROOT_CODES:
            drops["root_excluded"] += 1
            continue

        if row[COL_IS_ROOT_EVENT].strip() != "1":
            drops["not_root_event"] += 1
            continue

        try:
            mentions = int(row[COL_NUM_MENTIONS] or 0)
        except ValueError:
            drops["mentions_unparseable"] += 1
            continue
        if mentions < MIN_MENTIONS:
            drops["mentions_low"] += 1
            continue

        lat_s = row[COL_ACTIONGEO_LAT].strip()
        lng_s = row[COL_ACTIONGEO_LONG].strip()
        if not lat_s or not lng_s:
            drops["geo_missing"] += 1
            continue
        try:
            lat = float(lat_s)
            lng = float(lng_s)
        except ValueError:
            drops["geo_unparseable"] += 1
            continue
        if lat == 0.0 and lng == 0.0:
            drops["geo_null_island"] += 1
            continue

        try:
            ev = _normalize(row, root, lat, lng, mentions)
        except Exception as e:  # noqa: BLE001
            drops["normalize_error"] += 1
            logger.debug("normalize failed: %s", e)
            continue

        out.append(ev)

    logger.info(
        "GDELT: parsed=%d kept=%d dropped=%s",
        seen, len(out), dict(drops),
    )
    return out


def _normalize(
    row: list[str],
    root: str,
    lat: float,
    lng: float,
    mentions: int,
) -> NormalizedEvent:
    gid = row[COL_GLOBALEVENTID].strip()
    if not gid:
        raise ValueError("missing GLOBALEVENTID")

    event_code = row[COL_EVENT_CODE].strip()
    actor1 = row[COL_ACTOR1_NAME].strip()
    actor2 = row[COL_ACTOR2_NAME].strip()
    action_loc = row[COL_ACTIONGEO_FULLNAME].strip() or None
    country_fips = row[COL_ACTIONGEO_COUNTRY].strip() or None
    source_url = row[COL_SOURCE_URL].strip() or None

    goldstein = _try_float(row[COL_GOLDSTEIN_SCALE]) or 0.0
    avg_tone = _try_float(row[COL_AVG_TONE]) or 0.0
    quad_class = _try_int(row[COL_QUAD_CLASS]) or 0
    articles = _try_int(row[COL_NUM_ARTICLES]) or 0

    occurred_at = _parse_dateadded(row[COL_DATE_ADDED].strip())

    description_short = root_description(root)
    title = _make_title(actor1, actor2, description_short, action_loc)

    desc = (
        f"{mentions} mentions across {articles} articles. "
        f"Tone: {avg_tone:+.1f}. Goldstein: {goldstein:+.1f}."
    )

    severity = severity_from_gdelt(goldstein, mentions, articles, root)

    return NormalizedEvent(
        source=SOURCE_NAME,
        source_event_id=gid,
        source_url=source_url,
        title=title,
        description=desc,
        category=root_to_category(root),
        subcategory=event_code or None,
        location_name=action_loc,
        country_iso2=None,  # GDELT uses FIPS not ISO2; leave for later enrichment
        lat=lat,
        lng=lng,
        occurred_at=occurred_at,
        severity_hint=severity,
        raw={
            "globaleventid": gid,
            "event_code": event_code,
            "event_root_code": root,
            "quad_class": quad_class,
            "goldstein_scale": goldstein,
            "num_mentions": mentions,
            "num_articles": articles,
            "avg_tone": avg_tone,
            "actor1_name": actor1 or None,
            "actor2_name": actor2 or None,
            "action_geo_fips_country": country_fips,
        },
    )


def _make_title(actor1: str, actor2: str, action_desc: str, location: str | None) -> str:
    """Templated title from CAMEO description + actors + location.

    Examples:
        'Armed conflict: Russian Military → Civilians in Kharkiv, Ukraine'
        'Protest by Citizens (UKR) in Berlin, Germany'
        'Unconventional mass violence in Mogadishu, Somalia'
    """
    if actor1 and actor2:
        title = f"{action_desc}: {actor1.title()} \u2192 {actor2.title()}"
    elif actor1:
        title = f"{action_desc} by {actor1.title()}"
    else:
        title = action_desc
    if location:
        title = f"{title} in {location}"
    return title[:200]


def _parse_dateadded(s: str) -> datetime:
    """GDELT DATEADDED is YYYYMMDDHHMMSS in UTC."""
    return datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _try_float(s: str) -> float | None:
    try:
        return float(s) if s else None
    except ValueError:
        return None


def _try_int(s: str) -> int | None:
    try:
        return int(s) if s else None
    except ValueError:
        return None
