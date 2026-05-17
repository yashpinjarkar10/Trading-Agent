-- =============================================================================
-- 0001_init_events.sql
--
-- Initial schema for the Event Map feature.  Matches EVENT_MAP_SCHEMA.md §2.
-- Run via the Supabase SQL Editor (Project → SQL → New query → paste → Run)
-- or via `psql` against $DATABASE_URL.
--
-- Idempotent: safe to re-run.  Drops nothing.
-- =============================================================================

-- ── Extensions ───────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ── Enum ─────────────────────────────────────────────────────────────────────
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'event_category') THEN
        CREATE TYPE event_category AS ENUM (
            'conflict',      -- terrorist attack, war, military strike
            'protest',       -- civil unrest, rallies, strikes
            'politics',      -- election, sanction, central bank, regulation
            'disaster_nat',  -- earthquake, wildfire, flood, storm, volcano
            'disaster_hum',  -- famine, refugee crisis
            'economy',       -- earnings, default, layoff, supply chain
            'health',        -- outbreak, pandemic update
            'discovery',     -- scientific, space, resource find
            'other'
        );
    END IF;
END$$;

-- ── Main table ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS events (
    id              BIGSERIAL PRIMARY KEY,

    -- Primary source (the one we ingested first)
    source          TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    source_url      TEXT,

    -- Cross-source corroboration (Layer-2 dedup) — see schema doc §3
    -- Shape: [{"source": "...", "source_event_id": "...", "source_url": "...", "scraped_at": "..."}]
    also_seen_in    JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Content
    title           TEXT NOT NULL,
    description     TEXT,
    category        event_category NOT NULL,
    subcategory     TEXT,
    summary_short   TEXT,                       -- 1-line LLM summary for the chat agent

    -- Where
    location_name   TEXT,                       -- "Tokyo, Japan"
    country_iso2    CHAR(2),
    geom            GEOGRAPHY(POINT, 4326) NOT NULL,

    -- When
    occurred_at     TIMESTAMPTZ NOT NULL,
    scraped_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    enriched_at     TIMESTAMPTZ,                -- NULL until the LLM enricher has run

    -- Scoring (filled by enrichment)
    severity        SMALLINT,
    market_impact   SMALLINT,
    confidence      SMALLINT,                   -- bumps when also_seen_in grows
    affected_sectors TEXT[] NOT NULL DEFAULT '{}',
    affected_tickers TEXT[] NOT NULL DEFAULT '{}',

    -- Operational
    archived        BOOLEAN NOT NULL DEFAULT false,
    raw             JSONB,                      -- nulled out by retention job after 30d

    CONSTRAINT events_within_source UNIQUE (source, source_event_id),
    CONSTRAINT events_severity_range      CHECK (severity      IS NULL OR severity      BETWEEN 1 AND 10),
    CONSTRAINT events_market_impact_range CHECK (market_impact IS NULL OR market_impact BETWEEN 1 AND 10),
    CONSTRAINT events_confidence_range    CHECK (confidence    IS NULL OR confidence    BETWEEN 1 AND 10)
);

-- ── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS events_geom_idx       ON events USING GIST (geom);
CREATE INDEX IF NOT EXISTS events_occurred_idx   ON events (occurred_at DESC) WHERE NOT archived;
CREATE INDEX IF NOT EXISTS events_category_idx   ON events (category)         WHERE NOT archived;
CREATE INDEX IF NOT EXISTS events_severity_idx   ON events (severity DESC)    WHERE NOT archived;
CREATE INDEX IF NOT EXISTS events_country_idx    ON events (country_iso2)     WHERE NOT archived;
CREATE INDEX IF NOT EXISTS events_tickers_gin    ON events USING GIN (affected_tickers);
CREATE INDEX IF NOT EXISTS events_title_trgm     ON events USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS events_pending_enrich ON events (scraped_at)        WHERE enriched_at IS NULL;

-- ── Row-level security ───────────────────────────────────────────────────────
ALTER TABLE events ENABLE ROW LEVEL SECURITY;

-- Public read of non-archived rows.  Service role bypasses RLS automatically
-- and is the only role allowed to write (no INSERT/UPDATE/DELETE policy).
DROP POLICY IF EXISTS events_public_read ON events;
CREATE POLICY events_public_read ON events
    FOR SELECT
    TO anon, authenticated
    USING (NOT archived);

-- ── Realtime publication ─────────────────────────────────────────────────────
-- Supabase auto-creates the `supabase_realtime` publication.  Adding the table
-- enables WebSocket push of INSERT / UPDATE / DELETE to subscribed clients.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_publication WHERE pubname = 'supabase_realtime') THEN
        BEGIN
            ALTER PUBLICATION supabase_realtime ADD TABLE events;
        EXCEPTION
            WHEN duplicate_object THEN
                -- already added; no-op
                NULL;
        END;
    END IF;
END$$;

-- ── Sanity: confirm everything was created ───────────────────────────────────
-- SELECT COUNT(*) AS event_rows FROM events;
-- SELECT pg_size_pretty(pg_relation_size('events')) AS table_size;
