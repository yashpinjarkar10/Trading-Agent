# Event Map Scrapers

Standalone Python scripts that pull events from public sources and write
them to Supabase. Run on **GitHub Actions cron** (not inside the FastAPI
backend) so a scraper crash never touches the chat agent.

## Layout

```
scraper/
├── common/
│   ├── supabase_client.py   # singleton supabase-py client (service-role)
│   ├── models.py            # RawEvent + NormalizedEvent Pydantic DTOs
│   ├── upsert.py            # wraps the upsert_event Postgres RPC
│   └── enrichment.py        # rule-based severity scoring (NO LLM here)
├── sources/
│   ├── usgs.py              # earthquakes (M ≥ 2.5)         — ZERO LLM
│   ├── firms.py             # NASA wildfires (FRP-based)    — ZERO LLM (TODO)
│   ├── reliefweb.py         # humanitarian disasters        — ZERO LLM (TODO)
│   ├── gdelt.py             # news events (CAMEO codes)     — light LLM (TODO)
│   └── wikipedia.py         # current events portal         — LLM needed (TODO)
└── scripts/
    └── run_usgs.py          # GH Actions entrypoint per source
```

## Flow per scrape

```
GitHub Actions cron
    └── uv run python -m scraper.scripts.run_<source>
            │
            ├── 1. sources/<source>.py → fetch_events() → list[NormalizedEvent]
            │      • pull source feed
            │      • normalize fields
            │      • apply rule-based severity_hint (no LLM)
            │
            └── 2. common/upsert.py → upsert_batch(events)
                   • calls Postgres RPC `upsert_event(...)` per row
                   • RPC handles both dedup layers (see EVENT_MAP_SCHEMA.md §3)
                   • tallies inserted / updated / merged / failed
```

## LLM usage strategy (kept minimal)

| Stage | LLM? | Why |
|---|---|---|
| Scraper writes | **No** | All severity comes from source data via rules in `enrichment.py` |
| Daily category mapping (GDELT) | **No** | CAMEO event codes are a fixed ontology — pure lookup (`common/cameo.py`) |
| Daily category mapping (Wikipedia) | **Yes** | Free-form text, no taxonomy provided |
| Ticker / sector enrichment | **Yes, batched** | Separate `jobs/enricher.py` picks `severity_hint ≥ 6 AND enriched_at IS NULL` rows, batches 10/call to Gemini, runs every 5 min |
| Frontend rendering | **No** | Pure DB read |

Target: **< 50 LLM calls/day** even with > 5 000 events/day ingested.

## Adding a new source

1. **Create `sources/<name>.py`** with a top-level `fetch_events() -> list[NormalizedEvent]`.
2. **Pick a rule-based severity formula** and add it to `common/enrichment.py`. Only fall back to severity=None if you genuinely can't score it.
3. **Create `scripts/run_<name>.py`** — copy `run_usgs.py` and change the import.
4. **Create `.github/workflows/scrape-<name>.yml`** — copy `scrape-usgs.yml` and adjust the cron + secrets list if the source needs an API key.
5. **Run locally first**:
   ```powershell
   cd backend
   uv run python -m scraper.scripts.run_<name>
   ```
6. **Verify rows landed** via Supabase dashboard or `uv run python test_events_db.py`.

## Local development

```powershell
# From backend/
cd backend

# Make sure backend/.env has SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY
# (plus any source-specific key like NASA_FIRMS_API_KEY)

# Run any scraper once
uv run python -m scraper.scripts.run_usgs

# Verify rows appeared in Supabase
uv run python test_events_db.py
```

## GitHub Actions setup (one-time)

In **Repo → Settings → Secrets and variables → Actions**, add:

| Secret | Required for |
|---|---|
| `SUPABASE_URL` | every scraper |
| `SUPABASE_SERVICE_ROLE_KEY` | every scraper |
| `NASA_FIRMS_API_KEY` | `scrape-firms.yml` |
| `GEMINI_API_KEY` | `enrich-events.yml` (future) |

GitHub will block-list workflow runs if the cron interval is too short — `*/5 * * * *` is the floor.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Workflow says "0 events upserted" repeatedly | Source feed is genuinely quiet — check the feed URL in a browser |
| All rows fail with `function upsert_event(...) does not exist` | Migration 0002 not applied to Supabase |
| All rows fail with `permission denied for function upsert_event` | `GRANT EXECUTE` clause in 0002 didn't run; re-run that grant manually |
| Local run works, GH Actions fails | Secrets not set in repo settings |
| Random 401 from Supabase | Service-role key was rotated; update both `backend/.env` AND the GH secret |
