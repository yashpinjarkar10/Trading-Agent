"""
Rule-based scoring helpers. NO LLM calls in this file.

These give scrapers a `severity_hint` (1–10) at write time so the bulk of
events skip LLM enrichment entirely. A separate LLM enricher job (run as a
GH Action) only touches rows where:
    severity_hint >= 6  AND  affected_tickers IS empty  AND  enriched_at IS NULL

That keeps Gemini calls to <50/day even with thousands of events.
"""
from __future__ import annotations


def severity_from_magnitude(mag: float | None) -> int | None:
    """USGS earthquake magnitude → severity 1–10.

    Mapping (after Richter intuition + market-impact heuristic):
        M < 2.5    →  None  (drop, too small to ingest)
        M 2.5–3.9  →  1–2
        M 4.0–4.9  →  3
        M 5.0–5.9  →  5
        M 6.0–6.9  →  7  (felt widely; localised damage)
        M 7.0–7.9  →  9  (major; regional disruption)
        M ≥ 8.0    →  10 (great; cross-border impact)
    """
    if mag is None or mag < 2.5:
        return None
    if mag < 4.0:
        return max(1, int(round(mag - 1.5)))   # M2.5→1, M3.9→2
    if mag < 5.0:
        return 3
    if mag < 6.0:
        return 5
    if mag < 7.0:
        return 7
    if mag < 8.0:
        return 9
    return 10


def severity_from_fire_frp(frp_mw: float | None) -> int | None:
    """NASA FIRMS Fire Radiative Power (MW) → severity 1–10.

    FRP < 5 MW is small brush; >100 MW is major wildfire."""
    if frp_mw is None or frp_mw < 5:
        return None
    if frp_mw < 20:
        return 3
    if frp_mw < 50:
        return 5
    if frp_mw < 100:
        return 7
    if frp_mw < 250:
        return 8
    if frp_mw < 500:
        return 9
    return 10


def severity_from_gdelt(
    goldstein: float | None,
    num_mentions: int,
    num_articles: int,
    event_root_code: str,
) -> int | None:
    """GDELT row → severity 1–10.

    Three components combined:
      - Floor by event type        (20=7, 19=6, 18=5, 17=5, 14=4, 13=3, 07=2)
      - Goldstein-scale component  (max(0,-goldstein)/10 * 5)  — more negative = worse
      - Coverage component         (mentions/30 capped *2 + articles/50 capped *1)

    Goldstein scale ranges -10 (most violent) to +10 (most cooperative);
    only the negative side maps to severity. Coverage rewards corroboration.
    """
    floor_by_root = {"20": 7, "19": 6, "18": 5, "17": 5, "14": 4, "13": 3, "07": 2}
    floor = floor_by_root.get(event_root_code, 0)

    g = goldstein if goldstein is not None else 0.0
    g_component = max(0.0, -g) / 10.0 * 5.0  # 0..5

    coverage = (
        min(num_mentions / 30.0, 1.0) * 2.0
        + min(num_articles / 50.0, 1.0) * 1.0
    )

    raw = floor + g_component + coverage
    return max(1, min(10, round(raw)))


def severity_from_disaster_type(disaster_type: str | None, status: str | None) -> int | None:
    """ReliefWeb disaster type + status → severity 1–10.

    Heuristic mapping; the LLM enricher can override later for tickers/sectors.
    """
    if not disaster_type:
        return None
    t = disaster_type.lower()
    base = 5
    if any(k in t for k in ("earthquake", "tsunami", "war", "famine")):
        base = 8
    elif any(k in t for k in ("flood", "cyclone", "hurricane", "epidemic", "drought")):
        base = 7
    elif any(k in t for k in ("landslide", "wildfire", "fire", "storm")):
        base = 6
    elif any(k in t for k in ("technological", "transport accident")):
        base = 4

    if status and "ongoing" in status.lower():
        base = min(10, base + 1)
    return base
