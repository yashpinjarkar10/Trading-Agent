#!/usr/bin/env python3
"""
GitHub Actions entrypoint for the GDACS (UN) disaster alerts scraper.

Local:
    uv run python -m scraper.scripts.run_gdacs

GH Actions:
    .github/workflows/scrape-gdacs.yml runs on `*/20 * * * *`.
    Required env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY (no extra API key).

GDACS is the UN/EU system that pre-filters disasters above a humanitarian
impact threshold, so every event here is significant. The GDACS EQ events
will commonly trigger Layer-2 cross-source dedup with USGS earthquakes
(same quake, different source → merged row). Watch for action='merged' in
the stats on active seismic days.
"""
from __future__ import annotations

import logging
import sys
import traceback

from scraper.common.upsert import UpsertStats, upsert_batch
from scraper.sources import gdacs


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("run_gdacs")

    log.info("Starting GDACS disasters scraper")
    try:
        events = gdacs.fetch_events()
    except Exception:
        log.error("Fetch failed:\n%s", traceback.format_exc())
        return 1

    if not events:
        log.info("No events to upsert. Done.")
        return 0

    log.info("Upserting %d GDACS events to Supabase", len(events))
    stats = UpsertStats(source=gdacs.SOURCE_NAME)
    stats = upsert_batch(events, stats)
    log.info(stats.summary())

    if stats.merged > 0:
        log.info("Layer-2 dedup fired: %d events merged with existing sources", stats.merged)

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
