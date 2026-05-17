#!/usr/bin/env python3
"""
GitHub Actions entrypoint for the USGS earthquakes scraper.

Local:
    uv run python -m scraper.scripts.run_usgs

GH Actions:
    .github/workflows/scrape-usgs.yml runs `python -m scraper.scripts.run_usgs`
    on `*/5 * * * *` (every 5 min). Env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY.

Exit codes:
    0  success (even if 0 rows — feed may legitimately be empty for a quiet hour)
    1  fatal error (network failure, schema mismatch, etc.)
"""
from __future__ import annotations

import logging
import sys
import traceback

from scraper.common.upsert import UpsertStats, upsert_batch
from scraper.sources import usgs


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("run_usgs")

    log.info("Starting USGS earthquakes scraper")
    try:
        events = usgs.fetch_events()
    except Exception:
        log.error("Fetch failed:\n%s", traceback.format_exc())
        return 1

    if not events:
        log.info("No events to upsert (quiet hour or all below M2.5). Done.")
        return 0

    stats = UpsertStats(source=usgs.SOURCE_NAME)
    stats = upsert_batch(events, stats)
    log.info(stats.summary())

    if stats.errors:
        log.warning("First few errors:")
        for err in stats.errors[:5]:
            log.warning("  %s", err)

    # Non-zero exit if EVERY row failed (likely RPC misconfig or schema drift)
    if stats.received > 0 and stats.failed == stats.received:
        log.error("All rows failed — likely an RPC schema mismatch or auth issue")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
