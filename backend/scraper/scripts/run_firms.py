#!/usr/bin/env python3
"""
GitHub Actions entrypoint for the NASA FIRMS wildfire scraper.

Local:
    uv run python -m scraper.scripts.run_firms

Required env:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
    NASA_FIRMS_API_KEY        (free signup at https://firms.modaps.eosdis.nasa.gov/api/area/)
"""
from __future__ import annotations

import logging
import sys
import traceback

from scraper.common.upsert import UpsertStats, upsert_batch
from scraper.sources import firms


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("run_firms")

    log.info("Starting NASA FIRMS wildfire scraper")
    try:
        events = firms.fetch_events()
    except Exception:
        log.error("Fetch failed:\n%s", traceback.format_exc())
        return 1

    if not events:
        log.info("No fire events to upsert. Done.")
        return 0

    log.info("Upserting %d bucket-events to Supabase", len(events))
    stats = UpsertStats(source=firms.SOURCE_NAME)
    stats = upsert_batch(events, stats)
    log.info(stats.summary())

    if stats.errors:
        log.warning("First few errors:")
        for err in stats.errors[:5]:
            log.warning("  %s", err)

    if stats.received > 0 and stats.failed == stats.received:
        log.error("All rows failed — likely an RPC schema mismatch or auth issue")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
