"""
Wikipedia Current Events scraper with LLM text→structured parse.

Wikipedia's "Current events" page (https://en.wikipedia.org/wiki/Current_events)
contains human-curated summaries of recent news, organized by date and category.
Unlike the other scrapers (which have structured feeds), Wikipedia is free-form
text that requires LLM parsing to extract:
    - Event title
    - Location (city, country)
    - Coordinates (lat/lng)
    - Category (conflict, politics, economy, disaster, etc.)
    - Severity estimate
    - Affected tickers (if any)

This scraper:
    1. Fetches the Current Events page HTML
    2. Extracts date-sectioned text blocks
    3. Sends each block to Gemini (gemini-1.5-flash) with a structured prompt
    4. Parses the JSON response into NormalizedEvent
    5. Only processes the last 2 days of entries (to keep LLM calls <50/day)

LLM usage: ~1 call per day-section (2 days = ~2 calls/run), well under quota.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
from bs4 import BeautifulSoup
from google import genai

from ..common.models import EventCategory, NormalizedEvent

logger = logging.getLogger(__name__)

SOURCE_NAME = "wikipedia"
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Current_events"
REQUEST_TIMEOUT_S = 30

# Only process the last N days of entries to keep LLM calls bounded
DAYS_BACK = 2

# LLM prompt for text→structured event extraction
LLM_SYSTEM_PROMPT = """You are a structured data extraction assistant for a global event map.

Extract events from the Wikipedia Current Events text block. Return ONLY valid JSON
matching this schema:
{
  "events": [
    {
      "title": "Brief, factual title (max 100 chars)",
      "description": "1-2 sentence summary",
      "location_name": "City, Country or Region",
      "country_iso2": "2-letter ISO country code or null",
      "lat": float or null,
      "lng": float or null,
      "category": "conflict | protest | politics | disaster_nat | disaster_hum | economy | health | discovery | other",
      "subcategory": "lowercase_with_underscores or null",
      "severity": 1-10 (integer, estimate based on impact/scale),
      "affected_tickers": ["TICKER1", "TICKER2"] or [],
      "occurred_at": "ISO-8601 datetime or null"
    }
  ]
}

Rules:
- Extract ONLY events with clear geographic locations (skip generic "global" items)
- Use ISO-3166 alpha-2 for country codes (US, GB, JP, etc.)
- If coordinates are unknown, set lat/lng to null (we'll geocode later)
- Severity: 1-2 = minor/local, 3-5 = regional, 6-8 = national, 9-10 = global/catastrophic
- Tickernames: only extract if explicitly named in the text (e.g., "Tesla", "Apple")
- occurred_at: use the section date if the text doesn't specify a time
- Return empty array if no extractable events in the block

DO NOT include explanations or markdown. Just the JSON."""


def fetch_events() -> list[NormalizedEvent]:
    """Fetch Wikipedia Current Events, extract text blocks, LLM-parse to events."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — skipping Wikipedia scraper")
        return []

    logger.info("Wikipedia: fetching %s", WIKIPEDIA_URL)
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        r = client.get(WIKIPEDIA_URL, follow_redirects=True)
        r.raise_for_status()

    soup = BeautifulSoup(r.content, "html.parser")
    sections = _extract_date_sections(soup)
    logger.info("Wikipedia: found %d date-sections", len(sections))

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    all_events: list[NormalizedEvent] = []
    total_llm_calls = 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)

    for section in sections:
        section_date = section["date"]
        if section_date < cutoff:
            logger.debug("Wikipedia: skipping section %s (older than %s days)", section_date, DAYS_BACK)
            continue

        text = section["text"].strip()
        if len(text) < 50:
            logger.debug("Wikipedia: skipping section %s (too short)", section_date)
            continue

        try:
            total_llm_calls += 1
            logger.info("Wikipedia: LLM-parsing section %s (call #%d)", section_date, total_llm_calls)

            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[LLM_SYSTEM_PROMPT, text],
            )

            llm_output = response.text.strip()
            # Remove markdown code blocks if present
            if llm_output.startswith("```"):
                llm_output = llm_output.split("```", 2)[-1].strip()

            import json
            data = json.loads(llm_output)
            events_raw = data.get("events", [])

            for ev_raw in events_raw:
                try:
                    ev = _normalize(ev_raw, section_date)
                    if ev is None:
                        continue
                    all_events.append(ev)
                except Exception as e:
                    logger.debug("Wikipedia: normalize failed for event: %s", e)

        except json.JSONDecodeError as e:
            logger.error("Wikipedia: LLM returned invalid JSON for %s: %s", section_date, e)
        except Exception as e:
            logger.exception("Wikipedia: LLM call failed for %s", section_date)

    logger.info("Wikipedia: extracted %d events from %d LLM calls", len(all_events), total_llm_calls)
    return all_events


def _extract_date_sections(soup: BeautifulSoup) -> list[dict]:
    """Parse Wikipedia Current Events into date-sectioned text blocks."""
    sections = []

    # Wikipedia structure: <h2> with date, followed by content
    # We look for h2 elements within the main content area
    main_content = soup.find("div", {"id": "mw-content-text"})
    if not main_content:
        return sections

    h2_tags = main_content.find_all("h2")
    for h2 in h2_tags:
        # Extract date from h2 text
        header_text = h2.get_text(strip=True)
        if not header_text:
            continue

        # Parse date (format: "May 16, 2026" or similar)
        try:
            section_date = _parse_header_date(header_text)
        except Exception:
            logger.debug("Wikipedia: could not parse date from header: %s", header_text)
            continue

        # Collect all text until the next h2
        text_parts = []
        sibling = h2.find_next_sibling()
        while sibling and sibling.name != "h2":
            # Extract text from lists, paragraphs, etc.
            text_parts.append(sibling.get_text(separator=" ", strip=True))
            sibling = sibling.find_next_sibling()

        section_text = " ".join(text_parts)
        if section_text:
            sections.append({"date": section_date, "text": section_text})

    return sections


def _parse_header_date(header: str) -> datetime:
    """Parse Wikipedia section header like 'May 16, 2026 (Friday)' to datetime."""
    # Remove day-of-week and parentheses
    header = header.split("(")[0].strip()
    # Parse month day, year format
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(header, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Fallback: assume current year
    try:
        dt = datetime.strptime(header, "%B %d")
        return dt.replace(year=datetime.now().year, tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"Cannot parse date: {header}")


def _normalize(ev_raw: dict, section_date: datetime) -> NormalizedEvent | None:
    """Convert LLM JSON output to NormalizedEvent."""
    title = ev_raw.get("title") or ""
    if not title:
        return None

    description = ev_raw.get("description") or None
    location_name = ev_raw.get("location_name") or None
    country_iso2 = (ev_raw.get("country_iso2") or "").upper() or None

    lat = ev_raw.get("lat")
    lng = ev_raw.get("lng")
    if lat is None or lng is None:
        # LLM couldn't geocode — we'll accept null coords for now
        # (enricher can fill in later)
        lat, lng = None, None
    else:
        lat, lng = float(lat), float(lng)

    category_raw = ev_raw.get("category") or "other"
    try:
        category = EventCategory(category_raw)
    except ValueError:
        category = EventCategory.other

    subcategory = ev_raw.get("subcategory") or None

    severity = ev_raw.get("severity")
    if severity is not None:
        severity = max(1, min(10, int(severity)))
    else:
        severity = 5  # default mid-range

    affected_tickers = ev_raw.get("affected_tickers") or []
    if not isinstance(affected_tickers, list):
        affected_tickers = []

    occurred_at_str = ev_raw.get("occurred_at")
    if occurred_at_str:
        try:
            occurred_at = datetime.fromisoformat(occurred_at_str.replace("Z", "+00:00"))
            if occurred_at.tzinfo is None:
                occurred_at = occurred_at.replace(tzinfo=timezone.utc)
        except Exception:
            occurred_at = section_date
    else:
        occurred_at = section_date

    source_url = f"https://en.wikipedia.org/wiki/Current_events"

    return NormalizedEvent(
        source=SOURCE_NAME,
        source_event_id=f"wiki-{section_date.strftime('%Y%m%d')}-{hash(title) % 10000:04d}",
        source_url=source_url,
        title=title[:200],
        description=description,
        category=category,
        subcategory=subcategory,
        location_name=location_name,
        country_iso2=country_iso2,
        lat=lat,
        lng=lng,
        occurred_at=occurred_at,
        severity_hint=severity,
        raw={
            "wikipedia_section_date": section_date.isoformat(),
            "llm_extracted": True,
            "affected_tickers": affected_tickers,
        },
    )
