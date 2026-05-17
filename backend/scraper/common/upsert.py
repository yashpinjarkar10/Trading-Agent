"""
Thin wrapper around the `upsert_event` Postgres RPC.

The function lives in `backend/migrations/0002_upsert_event_rpc.sql` and
handles both dedup layers in SQL (see EVENT_MAP_SCHEMA.md §3). Scrapers
call this helper once per NormalizedEvent and we tally inserted/updated/merged.

`upsert_batch` runs calls concurrently via ThreadPoolExecutor (default 20
workers). httpx.Client is thread-safe (shared connection pool). Sequential
stats updates are protected by a threading.Lock.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import NormalizedEvent
from .supabase_client import get_supabase

# 10 workers is the sweet spot for Supabase's HTTP/2 connection pool on
# the free tier. Raising it above ~15 starts causing Server Disconnected
# errors under load (45/645 observed at workers=20).
DEFAULT_WORKERS = 10

# Per-row retry config for transient network errors (ServerDisconnected,
# RemoteProtocolError, ConnectTimeout). Each retry doubles the backoff.
_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY_S = 0.3

logger = logging.getLogger(__name__)


@dataclass
class UpsertStats:
    """Tally of one scraper run."""
    source: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    received: int = 0
    inserted: int = 0
    updated: int = 0
    merged: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        if not self.finished_at:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    def summary(self) -> str:
        return (
            f"[{self.source}] received={self.received} "
            f"inserted={self.inserted} updated={self.updated} "
            f"merged={self.merged} failed={self.failed} "
            f"in {self.duration_seconds:.1f}s"
        )


def upsert_event(ev: NormalizedEvent) -> str:
    """Send one NormalizedEvent through the upsert_event RPC.

    Returns the action: 'inserted' | 'updated' | 'merged'.
    Raises on hard failure (network, validation) — caller decides whether to
    swallow per-row errors.
    """
    sb = get_supabase()

    params = {
        "p_source":          ev.source,
        "p_source_event_id": ev.source_event_id,
        "p_title":           ev.title,
        "p_category":        ev.category.value,
        "p_lat":             ev.lat,
        "p_lng":             ev.lng,
        "p_occurred_at":     ev.occurred_at.isoformat(),
        "p_source_url":      ev.source_url,
        "p_description":     ev.description,
        "p_subcategory":     ev.subcategory,
        "p_location_name":   ev.location_name,
        "p_country_iso2":    ev.country_iso2,
        "p_severity":        ev.severity_hint,
        "p_raw":             ev.raw,
    }

    res = sb.rpc("upsert_event", params).execute()

    # supabase-py returns the SETOF row as a list of dicts.
    if not res.data:
        raise RuntimeError(f"upsert_event RPC returned no rows for {ev.source}/{ev.source_event_id}")
    return res.data[0]["action"]


def upsert_batch(
    events: list[NormalizedEvent],
    stats: UpsertStats,
    max_workers: int = DEFAULT_WORKERS,
) -> UpsertStats:
    """Send events concurrently through upsert_event, tallying outcomes.

    Uses a ThreadPoolExecutor so N RPC calls fly in parallel (default 20).
    For 645 FIRMS events this drops wall-time from ~158s → ~10s.
    Per-row errors are logged + counted but never abort the batch.
    """
    stats.received = len(events)
    lock = threading.Lock()

    def _call(ev: NormalizedEvent) -> None:
        delay = _RETRY_BASE_DELAY_S
        last_exc: Exception | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                action = upsert_event(ev)
                with lock:
                    if action == "inserted":
                        stats.inserted += 1
                    elif action == "updated":
                        stats.updated += 1
                    elif action == "merged":
                        stats.merged += 1
                    else:
                        logger.warning("Unknown action %r from upsert_event", action)
                return  # success
            except Exception as e:  # noqa: BLE001
                last_exc = e
                err_name = type(e).__name__
                # Only retry on transient network errors
                transient = any(k in err_name for k in ("Remote", "Connect", "Timeout", "Server"))
                if transient and attempt < _RETRY_ATTEMPTS - 1:
                    logger.debug(
                        "upsert transient %s for %s, retry %d in %.1fs",
                        err_name, ev.source_event_id, attempt + 1, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    break  # non-transient or last attempt

        err = f"{ev.source}/{ev.source_event_id}: {type(last_exc).__name__}: {last_exc}"
        with lock:
            stats.failed += 1
            stats.errors.append(err)
        logger.error("upsert failed after %d attempts for %s: %s", _RETRY_ATTEMPTS, ev.source_event_id, last_exc)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_call, ev): ev for ev in events}
        for future in as_completed(futures):
            # Exceptions are caught inside _call; re-raise only truly unexpected
            exc = future.exception()
            if exc is not None:
                ev = futures[future]
                logger.error("Unexpected future error for %s: %s", ev.source_event_id, exc)

    stats.finished_at = datetime.now(timezone.utc)
    return stats
