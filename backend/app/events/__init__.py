"""
Event Map module — READ-ONLY backend client.

Layout:
    db.py        Supabase client factory (singleton)
    models.py    Pydantic DTOs (EventOut, EventFilter, EventCategory)
    routes.py    FastAPI router under /api/events

Writes (scrapers, LLM enrichment, retention) live in /scrapers at repo root
and run on GitHub Actions — they're intentionally NOT imported here.

See EVENT_MAP_FEATURE.md for the design and EVENT_MAP_SCHEMA.md for the
canonical DB spec.
"""

__all__: list[str] = []
