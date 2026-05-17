import { useEffect, useMemo, useRef, useState } from 'react';
import Globe from 'react-globe.gl';
import { CATEGORY_COLOR } from '../../services/dummyEvents';

/**
 * EventGlobe — interactive 3D globe (react-globe.gl, Three.js).
 * Renders events as colored dots sized by market_impact; severity >= 7 also
 * spawns a pulsing ring. Auto-rotates until the user selects an event.
 *
 * Phase 3C — items 3.6, 3.7, 3.8, 3.9, 3.10
 */
export default function EventGlobe({ events, onSelect, selectedId }) {
  const containerRef = useRef(null);
  const globeRef = useRef(null);
  const [size, setSize] = useState({ w: 600, h: 600 });

  // Resize observer — keeps globe filling its container
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ w: Math.max(320, width), h: Math.max(320, height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Camera & controls setup
  useEffect(() => {
    const g = globeRef.current;
    if (!g) return;
    const c = g.controls();
    c.autoRotate = !selectedId;
    c.autoRotateSpeed = 0.35;
    c.enableZoom = true;
    c.enablePan = false;
    c.minDistance = 200;
    c.maxDistance = 600;
    // Slight initial tilt for a cinematic feel
    g.pointOfView({ lat: 25, lng: 10, altitude: 2.4 }, 0);
  }, []);

  // Pause rotation when an event is selected
  useEffect(() => {
    const g = globeRef.current;
    if (!g) return;
    g.controls().autoRotate = !selectedId;
  }, [selectedId]);

  // Focus globe on a selected event
  useEffect(() => {
    if (!selectedId) return;
    const g = globeRef.current;
    const ev = events.find((e) => e.id === selectedId);
    if (g && ev) g.pointOfView({ lat: ev.lat, lng: ev.lng, altitude: 1.8 }, 1200);
  }, [selectedId, events]);

  const points = useMemo(
    () =>
      events.map((e) => ({
        ...e,
        ptColor: CATEGORY_COLOR[e.category] || '#94a3b8',
        ptAltitude: 0.02 + Math.max(0, (e.market_impact ?? 1)) * 0.012,
        ptRadius: 0.22 + Math.max(0, (e.market_impact ?? 1)) * 0.035,
        isSelected: e.id === selectedId,
      })),
    [events, selectedId],
  );

  const rings = useMemo(
    () =>
      events
        .filter((e) => (e.severity ?? 0) >= 7 || (e.market_impact ?? 0) >= 7)
        .map((e) => ({
          lat: e.lat,
          lng: e.lng,
          maxR: 3.5 + (e.market_impact ?? 5) * 0.45,
          propagationSpeed: 2.2,
          repeatPeriod: 1800,
          color: CATEGORY_COLOR[e.category] || '#22d3ee',
        })),
    [events],
  );

  return (
    <div ref={containerRef} className="relative w-full h-full overflow-hidden rounded-2xl">
      {/* Background glow behind the globe */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'radial-gradient(circle at 50% 55%, rgba(34,211,238,0.10), transparent 55%), radial-gradient(circle at 50% 90%, rgba(139,92,246,0.08), transparent 60%)',
        }}
      />
      <Globe
        ref={globeRef}
        width={size.w}
        height={size.h}
        backgroundColor="rgba(0,0,0,0)"
        globeImageUrl="//unpkg.com/three-globe/example/img/earth-dark.jpg"
        bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
        showAtmosphere
        atmosphereColor="#22d3ee"
        atmosphereAltitude={0.18}

        pointsData={points}
        pointLat="lat"
        pointLng="lng"
        pointColor="ptColor"
        pointAltitude="ptAltitude"
        pointRadius="ptRadius"
        pointsMerge={false}
        pointResolution={6}
        pointLabel={(p) =>
          `<div style="font-family:Inter,sans-serif;background:rgba(13,15,26,0.95);border:1px solid rgba(255,255,255,0.12);padding:8px 10px;border-radius:8px;color:#f5f7fb;font-size:12px;max-width:260px;">
            <div style="font-weight:600;margin-bottom:2px;">${escapeHtml(p.title)}</div>
            <div style="color:#a3a8bd;font-size:11px;">${escapeHtml(p.location_name || '')} · sev ${p.severity ?? '?'} · impact ${p.market_impact ?? '?'}</div>
          </div>`
        }
        onPointClick={(p) => onSelect?.(p)}
        onPointHover={(p) => {
          document.body.style.cursor = p ? 'pointer' : '';
        }}

        ringsData={rings}
        ringLat="lat"
        ringLng="lng"
        ringColor={(d) => (t) => hexToRgba(d.color, 1 - t)}
        ringMaxRadius="maxR"
        ringPropagationSpeed="propagationSpeed"
        ringRepeatPeriod="repeatPeriod"
        ringAltitude={0.01}
      />
    </div>
  );
}

function escapeHtml(s) {
  return String(s ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function hexToRgba(hex, alpha) {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
}
