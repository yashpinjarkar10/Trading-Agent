"""
Supabase client factory for the scrapers.

Pattern lifted from the old `web-scraper-microservices` project. We read env
vars directly (not through FastAPI settings) so the scrapers can run on
GitHub Actions without pulling in any FastAPI / LangGraph dependencies.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from supabase import Client, create_client

# Load backend/.env (one level up from scraper/common/).  No-op on GitHub
# Actions where the env vars come from repo secrets.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", ".env"))
if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH)


def _require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(
            f"Environment variable {name} is required for scrapers. "
            f"Set it in backend/.env locally or in GitHub Actions repo secrets."
        )
    return val


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """Process-wide singleton Supabase client (service-role).

    Service-role bypasses RLS — that's correct here because scrapers are the
    only thing writing to events and they run server-side.
    """
    return create_client(
        _require_env("SUPABASE_URL"),
        _require_env("SUPABASE_SERVICE_ROLE_KEY"),
    )
