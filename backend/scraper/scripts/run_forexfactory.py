#!/usr/bin/env python3
"""
GitHub Actions / Local entrypoint for the Forex Factory scraper.

Local:
    uv run python -m scraper.scripts.run_forexfactory
"""
from __future__ import annotations

import logging
import sys
import traceback

from scraper.common.upsert import UpsertStats, upsert_batch
from scraper.sources import forexfactory


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("run_forexfactory")

    log.info("Starting Forex Factory macro events scraper")
    try:
        events = forexfactory.fetch_events()
    except Exception:
        log.error("Fetch failed:\n%s", traceback.format_exc())
        return 1

    if not events:
        log.info("No events to upsert. Done.")
        return 0

    stats = UpsertStats(source=forexfactory.SOURCE_NAME)
    stats = upsert_batch(events, stats)
    log.info(stats.summary())

    if stats.errors:
        log.warning("First few errors:")
        for err in stats.errors[:5]:
            log.warning("  %s", err)

    # Non-zero exit if EVERY row failed (indicates DB/RPC problem)
    if stats.received > 0 and stats.failed == stats.received:
        log.error("All rows failed — likely an RPC schema mismatch or auth issue")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
