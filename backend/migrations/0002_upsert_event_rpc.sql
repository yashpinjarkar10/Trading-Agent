-- =============================================================================
-- 0002_upsert_event_rpc.sql
--
-- PL/pgSQL function `upsert_event(...)` implementing the two-layer dedup
-- described in EVENT_MAP_SCHEMA.md §3.
--
-- Layer 1 (within-source rerun guard):
--     If (source, source_event_id) already exists, refresh changed fields and
--     return action='updated'.
--
-- Layer 2 (cross-source corroboration):
--     Else, search for a canonical row from a DIFFERENT source within
--     50 km / ±6 hours / same category. If found, append to also_seen_in,
--     bump confidence, take max(severity), return action='merged'.
--
-- Otherwise insert a new canonical row, return action='inserted'.
--
-- Scrapers (Python on GH Actions) call this via supabase-py:
--     sb.rpc('upsert_event', { 'p_source': 'usgs', ... }).execute()
--
-- Idempotent: safe to re-run (uses CREATE OR REPLACE).
-- =============================================================================

CREATE OR REPLACE FUNCTION upsert_event(
    p_source            TEXT,
    p_source_event_id   TEXT,
    p_title             TEXT,
    p_category          event_category,
    p_lat               DOUBLE PRECISION,
    p_lng               DOUBLE PRECISION,
    p_occurred_at       TIMESTAMPTZ,
    p_source_url        TEXT     DEFAULT NULL,
    p_description       TEXT     DEFAULT NULL,
    p_subcategory       TEXT     DEFAULT NULL,
    p_location_name     TEXT     DEFAULT NULL,
    p_country_iso2      CHAR(2)  DEFAULT NULL,
    p_severity          SMALLINT DEFAULT NULL,
    p_raw               JSONB    DEFAULT NULL,
    -- Dedup tuning knobs; defaults from schema doc §3
    p_dedup_radius_m    INTEGER  DEFAULT 50000,
    p_dedup_window_sec  INTEGER  DEFAULT 21600  -- 6 h
)
RETURNS TABLE (event_id BIGINT, action TEXT)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_existing_id  BIGINT;
    v_canonical_id BIGINT;
    v_geom         GEOGRAPHY(POINT, 4326);
BEGIN
    -- Validate inputs
    IF p_lat IS NULL OR p_lng IS NULL THEN
        RAISE EXCEPTION 'upsert_event: lat/lng required (got % / %)', p_lat, p_lng;
    END IF;
    IF p_lat NOT BETWEEN -90 AND 90 OR p_lng NOT BETWEEN -180 AND 180 THEN
        RAISE EXCEPTION 'upsert_event: lat/lng out of range (% / %)', p_lat, p_lng;
    END IF;

    v_geom := ST_SetSRID(ST_MakePoint(p_lng, p_lat), 4326)::geography;

    -- ── Layer 1: same source + source_event_id already in DB? ──────────────
    SELECT id INTO v_existing_id
    FROM events
    WHERE source = p_source
      AND source_event_id = p_source_event_id;

    IF v_existing_id IS NOT NULL THEN
        UPDATE events
        SET title       = p_title,
            description = COALESCE(p_description, description),
            source_url  = COALESCE(p_source_url, source_url),
            severity    = COALESCE(p_severity, severity),
            raw         = COALESCE(p_raw, raw),
            geom        = v_geom,
            occurred_at = p_occurred_at
        WHERE id = v_existing_id;

        RETURN QUERY SELECT v_existing_id, 'updated'::TEXT;
        RETURN;
    END IF;

    -- ── Layer 2: cross-source corroboration? ───────────────────────────────
    -- Match same category, different source, near in space + time, not archived.
    SELECT id INTO v_canonical_id
    FROM events
    WHERE category = p_category
      AND source <> p_source
      AND NOT archived
      AND ST_DWithin(geom, v_geom, p_dedup_radius_m)
      AND ABS(EXTRACT(EPOCH FROM (occurred_at - p_occurred_at))) <= p_dedup_window_sec
    ORDER BY ABS(EXTRACT(EPOCH FROM (occurred_at - p_occurred_at))) ASC
    LIMIT 1;

    IF v_canonical_id IS NOT NULL THEN
        UPDATE events
        SET also_seen_in = also_seen_in || jsonb_build_object(
                'source',          p_source,
                'source_event_id', p_source_event_id,
                'source_url',      p_source_url,
                'scraped_at',      now()
            ),
            -- Bump confidence by 1 per corroboration, cap at 10
            confidence = LEAST(10, COALESCE(confidence, 5) + 1),
            -- Take the higher severity hint
            severity   = GREATEST(COALESCE(severity, 0), COALESCE(p_severity, 0))
        WHERE id = v_canonical_id;

        RETURN QUERY SELECT v_canonical_id, 'merged'::TEXT;
        RETURN;
    END IF;

    -- ── New canonical row ──────────────────────────────────────────────────
    INSERT INTO events (
        source, source_event_id, source_url,
        title, description,
        category, subcategory,
        location_name, country_iso2,
        geom,
        occurred_at,
        severity,
        raw
    ) VALUES (
        p_source, p_source_event_id, p_source_url,
        p_title, p_description,
        p_category, p_subcategory,
        p_location_name, p_country_iso2,
        v_geom,
        p_occurred_at,
        p_severity,
        p_raw
    )
    RETURNING id INTO v_existing_id;

    RETURN QUERY SELECT v_existing_id, 'inserted'::TEXT;
END;
$$;

-- Grant execute to the service_role (backend + scrapers) and authenticated
-- (for future server-side admin tools). anon stays read-only.
GRANT EXECUTE ON FUNCTION upsert_event(
    TEXT, TEXT, TEXT, event_category, DOUBLE PRECISION, DOUBLE PRECISION,
    TIMESTAMPTZ, TEXT, TEXT, TEXT, TEXT, CHAR, SMALLINT, JSONB, INTEGER, INTEGER
) TO service_role, authenticated;

COMMENT ON FUNCTION upsert_event IS
'Two-layer dedup upsert for the events table. See EVENT_MAP_SCHEMA.md §3. '
'Returns (event_id, action) where action is one of: inserted, updated, merged.';
