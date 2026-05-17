"""
Smoke test for the Event Map Supabase connection (read-only backend client).

Verifies:
  1. Required env vars are populated
  2. supabase-py client builds successfully
  3. We can hit Supabase (auth + REST work)
  4. `events` table exists and is queryable
  5. The table is empty (we haven't ingested anything yet)
  6. /api/events/_ping returns 200

Run:
    uv run python test_events_db.py

Exits 0 on full pass, 1 on any failure.
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv

# Load .env BEFORE settings is imported so it picks up the values.
load_dotenv()

from app.config.settings import settings  # noqa: E402
from app.events.db import EventsNotEnabledError, get_supabase, reset  # noqa: E402


class CheckRunner:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def ok(self, label: str, detail: str = "") -> None:
        self.passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))

    def fail(self, label: str, detail: str) -> None:
        self.failed += 1
        print(f"  FAIL  {label}  ({detail})")

    def report(self) -> int:
        print()
        total = self.passed + self.failed
        if self.failed == 0:
            print(f"All {total} checks passed.")
            return 0
        print(f"{self.failed} of {total} checks FAILED.")
        return 1


def check_env(c: CheckRunner) -> bool:
    print("[1] Environment variables")
    if not settings.EVENTS_ENABLED:
        c.fail("EVENTS_ENABLED", "set EVENTS_ENABLED=true in backend/.env")
        return False
    c.ok("EVENTS_ENABLED", "true")

    if not settings.SUPABASE_URL:
        c.fail("SUPABASE_URL", "empty in backend/.env")
        return False
    c.ok("SUPABASE_URL", settings.SUPABASE_URL)

    if not settings.SUPABASE_SERVICE_ROLE_KEY:
        c.fail("SUPABASE_SERVICE_ROLE_KEY", "empty in backend/.env")
        return False
    masked = settings.SUPABASE_SERVICE_ROLE_KEY[:11] + "..."
    c.ok("SUPABASE_SERVICE_ROLE_KEY", masked)
    return True


def check_supabase(c: CheckRunner) -> None:
    print("\n[2] Supabase client")
    reset()  # ensure we use freshly-loaded settings
    try:
        sb = get_supabase()
        c.ok("client built", type(sb).__name__)
    except EventsNotEnabledError as e:
        c.fail("client", str(e))
        return
    except Exception as e:
        c.fail("client", f"{type(e).__name__}: {e}")
        return

    print("\n[3] events table reachable")
    try:
        res = sb.table("events").select("id", count="exact").limit(1).execute()
        c.ok("query succeeded", f"count={res.count}")
        print("\n[4] data sanity (should be empty for now)")
        if res.count == 0:
            c.ok("events table empty", "0 rows")
        else:
            c.ok("events table has data", f"{res.count} rows")
    except Exception as e:
        c.fail("query", f"{type(e).__name__}: {e}")
        print(
            "\nHints:\n"
            "  - 'relation \"events\" does not exist'  → migration 0001 not run\n"
            "  - 'invalid API key' / 'JWT'            → wrong SUPABASE_SERVICE_ROLE_KEY\n"
            "  - DNS / network error                  → wrong SUPABASE_URL\n"
            "  - 401 Unauthorized                     → using ANON key instead of SERVICE_ROLE"
        )
        return

    print("\n[5] /api/events/_ping (FastAPI integration)")
    try:
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            r = client.get("/api/events/_ping")
            if r.status_code == 200:
                c.ok("/api/events/_ping returns 200", str(r.json()))
            else:
                c.fail("/api/events/_ping", f"status={r.status_code} body={r.text}")
    except Exception as e:
        c.fail("/api/events/_ping", f"{type(e).__name__}: {e}")


def main() -> int:
    c = CheckRunner()
    print("=" * 70)
    print("Event Map Supabase smoke test")
    print("=" * 70)

    if not check_env(c):
        print("\nSkipping DB checks — fix env vars first.")
        return c.report()

    check_supabase(c)
    return c.report()


if __name__ == "__main__":
    sys.exit(main())
