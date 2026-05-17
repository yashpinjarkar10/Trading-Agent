"""
Supabase client factory for the Event Map (read-only side).

The backend NEVER writes to the events table — that happens in /scrapers
running on GitHub Actions. Here we just build a thin singleton client used
by the /api/events routes to query Supabase.

We use the SERVICE_ROLE key (not anon) because the backend is a trusted
server, RLS gets in the way of admin queries (clustering, archived rows),
and the key never leaves the server. The frontend uses the ANON key
separately for its Realtime subscription.
"""
from __future__ import annotations

import logging

from supabase import Client, create_client

from app.config.settings import settings

logger = logging.getLogger(__name__)

_client: Client | None = None


class EventsNotEnabledError(RuntimeError):
    """Raised when event-map code is called but EVENTS_ENABLED is false
    or the Supabase credentials are missing."""


def _build_client() -> Client:
    if not settings.EVENTS_ENABLED:
        raise EventsNotEnabledError(
            "EVENTS_ENABLED=false — set it to true in .env to use the Event Map."
        )
    if not settings.SUPABASE_URL:
        raise EventsNotEnabledError("SUPABASE_URL is empty.")
    if not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise EventsNotEnabledError("SUPABASE_SERVICE_ROLE_KEY is empty.")
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)


def get_supabase() -> Client:
    """Return a process-wide singleton Supabase client.

    Usage:
        from app.events.db import get_supabase
        sb = get_supabase()
        rows = sb.table("events").select("*").limit(100).execute().data
    """
    global _client
    if _client is None:
        _client = _build_client()
        logger.info("Supabase client initialized for %s", settings.SUPABASE_URL)
    return _client


def reset() -> None:
    """Drop the cached client (used by tests to re-read env)."""
    global _client
    _client = None
