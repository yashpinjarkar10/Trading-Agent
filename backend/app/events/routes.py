"""
FastAPI router for the Event Map (read-only).

Endpoints:
    GET /api/events/_ping        liveness smoke test
    GET /api/events              list with filters       (Phase 2.7 — stub)
    GET /api/events/{id}         single event            (Phase 2.8 — stub)
    GET /api/events/stats        category aggregations   (Phase 2.9 — stub)

There is NO write endpoint here on purpose — writes happen via /scrapers
running on GitHub Actions, going straight to Supabase with the service-role
key. The frontend reads from this router and subscribes to Supabase Realtime
(via supabase-js) for push updates.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.events.db import EventsNotEnabledError, get_supabase
from app.events.models import EventCategory, EventOut, EventFilter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("/_ping")
async def ping() -> dict:
    """Liveness check — verifies the Supabase client wires up and the
    events table is reachable. Returns row count."""
    try:
        sb = get_supabase()
        res = sb.table("events").select("id", count="exact").limit(1).execute()
        return {
            "status": "ok",
            "supabase_url": str(sb.supabase_url),
            "events_row_count": res.count,
        }
    except EventsNotEnabledError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("events _ping failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("")
async def list_events(
    since: Optional[str] = Query(None, description="ISO-8601 datetime"),
    until: Optional[str] = Query(None, description="ISO-8601 datetime"),
    categories: Optional[list[str]] = Query(None, description="Filter by category"),
    min_severity: int = Query(0, ge=0, le=10),
    min_market_impact: int = Query(0, ge=0, le=10),
    limit: int = Query(500, ge=1, le=2000),
    cursor: Optional[int] = Query(None, description="Last-seen id for pagination"),
) -> dict:
    """List events with optional filters.
    
    Returns {items: EventOut[], total: int, next_cursor: int | null}.
    """
    try:
        sb = get_supabase()
        # Select all fields including geom (parse client-side)
        query = sb.table("events").select("*", count="exact")

        # Build filters
        if since:
            query = query.gte("occurred_at", since)
        if until:
            query = query.lte("occurred_at", until)
        if categories:
            query = query.in_("category", categories)
        if min_severity > 0:
            query = query.gte("severity", min_severity)
        if min_market_impact > 0:
            query = query.gte("market_impact", min_market_impact)
        if cursor:
            query = query.gt("id", cursor)

        query = query.order("occurred_at", desc=True).order("id", desc=True).limit(limit)
        res = query.execute()

        # Extract lat/lng from geom field
        items = [_parse_geom(row) for row in res.data]
        items = [EventOut(**row) for row in items]
        next_cursor = items[-1].id if len(items) == limit else None

        return {"items": items, "total": res.count, "next_cursor": next_cursor}
    except EventsNotEnabledError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("list_events failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


class StatsResponse(BaseModel):
    by_category: dict[str, int]
    last_24h: int
    total: int


@router.get("/stats")
async def get_stats() -> StatsResponse:
    """Aggregate statistics: counts by category, last 24h, total."""
    try:
        sb = get_supabase()
        
        # Total count
        total_res = sb.table("events").select("id", count="exact").limit(1).execute()
        total = total_res.count or 0

        # By category
        category_res = sb.table("events").select("category", count="exact").execute()
        by_category: dict[str, int] = {}
        for row in category_res.data:
            cat = row.get("category")
            if cat:
                by_category[cat] = by_category.get(cat, 0) + 1

        # Last 24h
        from datetime import timedelta
        cutoff_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z"
        recent_res = sb.table("events").select("id", count="exact").gte("occurred_at", cutoff_24h).execute()
        last_24h = recent_res.count or 0

        return StatsResponse(by_category=by_category, last_24h=last_24h, total=total)
    except EventsNotEnabledError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("get_stats failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/{event_id}")
async def get_event(event_id: int) -> EventOut:
    """Get a single event by ID."""
    try:
        sb = get_supabase()
        res = sb.table("events").select("*").eq("id", event_id).limit(1).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
        row = _parse_geom(res.data[0])
        return EventOut(**row)
    except EventsNotEnabledError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("get_event failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


def _parse_geom(row: dict) -> dict:
    """Parse Supabase's geom field format and extract lat/lng.
    
    Supabase returns geom as a dict with 'coordinates' array [lng, lat]
    or as a WKT string. We handle both.
    """
    geom = row.get("geom")
    lat, lng = None, None
    
    if geom:
        # Handle GeoJSON-like dict format
        if isinstance(geom, dict):
            coords = geom.get("coordinates")
            if coords and len(coords) == 2:
                lng, lat = coords[0], coords[1]
        # Handle WKT string format: "POINT(lng lat)"
        elif isinstance(geom, str) and "POINT" in geom:
            try:
                coords = geom.replace("POINT(", "").rstrip(")")
                lng_str, lat_str = coords.split()
                lng, lat = float(lng_str), float(lat_str)
            except (ValueError, IndexError):
                pass
    
    row["lat"] = lat
    row["lng"] = lng
    return row
