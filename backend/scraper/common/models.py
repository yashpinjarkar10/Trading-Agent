"""
Pydantic DTOs used by every scraper.

`RawEvent`         — whatever a source returns, before normalization.
`NormalizedEvent`  — the canonical shape passed to upsert(). Matches the
                     events table columns 1-to-1 except `geom` is split into
                     `lat` / `lng` for ease of JSON transport via supabase-py.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class EventCategory(str, enum.Enum):
    conflict     = "conflict"
    protest      = "protest"
    politics     = "politics"
    disaster_nat = "disaster_nat"
    disaster_hum = "disaster_hum"
    economy      = "economy"
    health       = "health"
    discovery    = "discovery"
    other        = "other"


class RawEvent(BaseModel):
    """Loose holder for a source's raw payload before normalization."""
    source: str
    source_event_id: str
    payload: dict[str, Any]


class NormalizedEvent(BaseModel):
    """Output of a scraper's `normalize()`. Ready for upsert_event RPC."""
    source: str
    source_event_id: str
    source_url: Optional[str] = None

    title: str
    description: Optional[str] = None
    category: EventCategory
    subcategory: Optional[str] = None

    location_name: Optional[str] = None
    country_iso2: Optional[str] = None  # populated later by geocoder if NULL

    lat: float
    lng: float

    occurred_at: datetime

    # Rule-based severity hint from the source (e.g. magnitude * 1.5 for
    # earthquakes). NULL means "let the LLM enricher decide later".
    severity_hint: Optional[int] = Field(default=None, ge=1, le=10)

    raw: Optional[dict[str, Any]] = None

    @field_validator("country_iso2")
    @classmethod
    def _upper_iso2(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if v else v

    @field_validator("lat")
    @classmethod
    def _lat_range(cls, v: float) -> float:
        if not -90 <= v <= 90:
            raise ValueError(f"lat out of range: {v}")
        return v

    @field_validator("lng")
    @classmethod
    def _lng_range(cls, v: float) -> float:
        if not -180 <= v <= 180:
            raise ValueError(f"lng out of range: {v}")
        return v
