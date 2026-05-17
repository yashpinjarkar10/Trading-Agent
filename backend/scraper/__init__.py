"""
Event Map scrapers — runs on GitHub Actions, writes directly to Supabase.

See backend/scraper/README.md for layout and how to add a new source.

Do NOT import from `app.*` here — scrapers must be runnable standalone with
just the supabase + httpx + python-dotenv deps (those are already in
backend's pyproject.toml so we share the env on dev, and GH Actions
installs them fresh each run).
"""

__all__: list[str] = []
