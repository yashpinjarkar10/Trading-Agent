"""
Pydantic DTOs for the Event Map read API.

No SQLAlchemy ORM — the backend never writes; supabase-py returns dicts
that we parse into these models for the API response.

The canonical column list lives in EVENT_MAP_SCHEMA.md §2 / the migration
SQL. These DTOs mirror it.
"""
from __future__ import annotations

import enum
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class EventCategory(str, enum.Enum):
    conflict = "conflict"
    protest = "protest"
    politics = "politics"
    disaster_nat = "disaster_nat"
    disaster_hum = "disaster_hum"
    economy = "economy"
    health = "health"
    discovery = "discovery"
    other = "other"


class EventOut(BaseModel):
    """Public shape returned by /api/events. Matches what supabase-py gives us
    after we unpack `geom` (a WKB/WKT) into lat/lng."""
    id: int
    source: str
    source_url: str | None = None
    also_seen_in: list[dict] = Field(default_factory=list)

    title: str
    description: str | None = None
    category: EventCategory
    subcategory: str | None = None
    summary_short: str | None = None

    location_name: str | None = None
    country_iso2: str | None = None
    lat: float | None = None
    lng: float | None = None

    occurred_at: datetime
    scraped_at: datetime

    severity: int | None = None
    market_impact: int | None = None
    confidence: int | None = None
    affected_sectors: list[str] = Field(default_factory=list)
    affected_tickers: list[str] = Field(default_factory=list)


class EventFilter(BaseModel):
    """Query-string params for GET /api/events. Parsed in the route handler."""
    since: datetime | None = None
    until: datetime | None = None
    categories: list[EventCategory] | None = None
    min_severity: int = Field(default=0, ge=0, le=10)
    min_market_impact: int = Field(default=0, ge=0, le=10)
    # bbox: (lng_min, lat_min, lng_max, lat_max). Optional viewport filter.
    bbox: tuple[float, float, float, float] | None = None
    tickers: list[str] | None = None
    limit: int = Field(default=500, ge=1, le=2000)
    cursor: int | None = None  # last-seen id, for keyset pagination

    @field_validator("bbox")
    @classmethod
    def _bbox_sane(cls, v: tuple[float, float, float, float] | None):
        if v is None:
            return v
        lng_min, lat_min, lng_max, lat_max = v
        if not (-180 <= lng_min < lng_max <= 180):
            raise ValueError("bbox longitudes invalid")
        if not (-90 <= lat_min < lat_max <= 90):
            raise ValueError("bbox latitudes invalid")
        return v
