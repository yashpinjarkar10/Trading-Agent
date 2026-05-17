#!/usr/bin/env python3
"""
GitHub Actions entrypoint for the GDELT events scraper.

Local:
    uv run python -m scraper.scripts.run_gdelt

GH Actions:
    .github/workflows/scrape-gdelt.yml runs `python -m scraper.scripts.run_gdelt`
    on `*/15 * * * *`. Env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY.

Exit codes:
    0  success (even if 0 rows survived filtering)
    1  fatal error (network, schema mismatch, all rows failed)
"""
from __future__ import annotations

import logging
import sys
import traceback

from scraper.common.upsert import UpsertStats, upsert_batch
from scraper.sources import gdelt


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("run_gdelt")

    log.info("Starting GDELT scraper")
    try:
        events = gdelt.fetch_events()
    except Exception:
        log.error("Fetch failed:\n%s", traceback.format_exc())
        return 1

    if not events:
        log.info("No events to upsert after filtering. Done.")
        return 0

    log.info("Upserting %d events to Supabase", len(events))
    stats = UpsertStats(source=gdelt.SOURCE_NAME)
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
