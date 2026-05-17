#!/usr/bin/env python3
"""
GitHub Actions entrypoint for the ReliefWeb disasters scraper.

Local:
    uv run python -m scraper.scripts.run_reliefweb

GH Actions:
    .github/workflows/scrape-reliefweb.yml runs on `*/20 * * * *`.
    Required env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY (no extra API key).
"""
from __future__ import annotations

import logging
import sys
import traceback

from scraper.common.upsert import UpsertStats, upsert_batch
from scraper.sources import reliefweb


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("run_reliefweb")

    log.info("Starting ReliefWeb disasters scraper")
    try:
        events = reliefweb.fetch_events()
    except Exception:
        log.error("Fetch failed:\n%s", traceback.format_exc())
        return 1

    if not events:
        log.info("No events to upsert (quiet window or all records lacked coordinates). Done.")
        return 0

    log.info("Upserting %d disaster events to Supabase", len(events))
    stats = UpsertStats(source=reliefweb.SOURCE_NAME)
    stats = upsert_batch(events, stats)
    log.info(stats.summary())

    if stats.errors:
        log.warning("First few errors:")
        for err in stats.errors[:5]:
            log.warning("  %s", err)

    if stats.received > 0 and stats.failed == stats.received:
        log.error("All rows failed — RPC issue or schema mismatch")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
