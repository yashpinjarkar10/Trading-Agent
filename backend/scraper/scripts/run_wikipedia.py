#!/usr/bin/env python3
"""
GitHub Actions entrypoint for the Wikipedia Current Events scraper.

Local:
    uv run python -m scraper.scripts.run_wikipedia

GH Actions:
    .github/workflows/scrape-wikipedia.yml runs on `0 */4 * * *` (every 4 hours).
    Required env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, GEMINI_API_KEY.

Wikipedia requires LLM parsing (Gemini) to extract structured events from
free-form text. This is the only scraper that uses an LLM at write time.
To keep costs low, we only process the last 2 days of entries (~2 LLM calls/run).
"""
from __future__ import annotations

import logging
import sys
import traceback

from scraper.common.upsert import UpsertStats, upsert_batch
from scraper.sources import wikipedia


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("run_wikipedia")

    log.info("Starting Wikipedia Current Events scraper")
    try:
        events = wikipedia.fetch_events()
    except Exception:
        log.error("Fetch failed:\n%s", traceback.format_exc())
        return 1

    if not events:
        log.info("No events extracted (LLM may have found no extractable items). Done.")
        return 0

    log.info("Upserting %d Wikipedia events to Supabase", len(events))
    stats = UpsertStats(source=wikipedia.SOURCE_NAME)
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
