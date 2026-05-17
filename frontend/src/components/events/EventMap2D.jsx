import { useEffect, useRef, useState } from 'react';
import { ComposableMap, Geographies, Geography, Marker, ZoomableGroup } from 'react-simple-maps';
import { CATEGORY_COLOR, CATEGORY_LABEL } from '../../services/dummyEvents';

/**
 * EventMap2D — flat "scan" view of the same event set.
 * Intentionally calmer than the 3D globe: no rotation, no atmosphere, no
 * orbit. Optimized for seeing everything at once on small screens / low-end
 * GPUs / reduced-motion users.
 *
 * Same data, same onSelect contract as EventGlobe — drop-in renderer swap.
 */

// Reliable Natural-Earth low-res countries topojson via jsDelivr CDN.
const GEO_URL = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json';

export default function EventMap2D({ events, selectedId, onSelect }) {
  const containerRef = useRef(null);
  const [hover, setHover] = useState(null); // { event, x, y }

  // Clear hover when events list changes (filters re-apply)
  useEffect(() => setHover(null), [events]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden rounded-2xl"
      onMouseLeave={() => setHover(null)}
    >
      {/* Subtle backdrop glow to match the globe view aesthetic */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse 70% 55% at 50% 50%, rgba(34,211,238,0.06), transparent 60%), radial-gradient(ellipse 80% 60% at 50% 100%, rgba(139,92,246,0.05), transparent 60%)',
        }}
      />

      <ComposableMap
        projection="geoEqualEarth"
        projectionConfig={{ scale: 180, center: [10, 10] }}
        style={{ width: '100%', height: '100%' }}
      >
        <ZoomableGroup center={[10, 10]} zoom={1} minZoom={1} maxZoom={5}>
          <Geographies geography={GEO_URL}>
            {({ geographies }) =>
              geographies.map((geo) => (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  fill="#0d0f1a"
                  stroke="rgba(255,255,255,0.07)"
                  strokeWidth={0.4}
                  style={{
                    default: { outline: 'none' },
                    hover:   { fill: '#141826', outline: 'none', transition: 'fill 150ms' },
                    pressed: { outline: 'none' },
                  }}
                />
              ))
            }
          </Geographies>

          {events.map((e) => (
            <EventMarker
              key={e.id}
              event={e}
              isSelected={e.id === selectedId}
              onClick={() => onSelect?.(e)}
              onHover={(payload) => setHover(payload)}
            />
          ))}
        </ZoomableGroup>
      </ComposableMap>

      {/* Hover tooltip */}
      {hover?.event && (
        <div
          className="pointer-events-none fixed z-50 rounded-lg border border-border bg-bg-elevated/95 backdrop-blur-md px-3 py-2 shadow-lg max-w-xs"
          style={{ left: hover.x + 14, top: hover.y + 14 }}
        >
          <div className="text-xs font-semibold text-text-primary mb-0.5 line-clamp-2">
            {hover.event.title}
          </div>
          <div className="text-[10px] text-text-muted font-mono uppercase tracking-wide">
            {CATEGORY_LABEL[hover.event.category]} · sev {hover.event.severity} · impact {hover.event.market_impact}
          </div>
          <div className="text-[11px] text-text-secondary mt-0.5">
            {hover.event.location_name}
          </div>
        </div>
      )}
    </div>
  );
}

function EventMarker({ event, isSelected, onClick, onHover }) {
  const color = CATEGORY_COLOR[event.category] || '#94a3b8';
  const r = 2.2 + Math.max(0, (event.market_impact ?? 1)) * 0.55;
  const isHigh = (event.market_impact ?? 0) >= 7 || (event.severity ?? 0) >= 7;

  return (
    <Marker
      coordinates={[event.lng, event.lat]}
      onClick={onClick}
      onMouseEnter={(evt) => onHover({ event, x: evt.clientX, y: evt.clientY })}
      onMouseMove={(evt) => onHover({ event, x: evt.clientX, y: evt.clientY })}
      onMouseLeave={() => onHover(null)}
      style={{ default: { cursor: 'pointer' }, hover: { cursor: 'pointer' }, pressed: { cursor: 'pointer' } }}
    >
      {/* Pulsing ring for high-impact events */}
      {isHigh && (
        <circle r={r} fill="none" stroke={color} strokeWidth={0.8} opacity={0.7}>
          <animate
            attributeName="r"
            from={r}
            to={r * 4.5}
            dur="1.9s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            from="0.7"
            to="0"
            dur="1.9s"
            repeatCount="indefinite"
          />
        </circle>
      )}

      {/* Soft glow */}
      <circle r={r * 1.6} fill={color} opacity={0.18} />

      {/* Core dot */}
      <circle
        r={r}
        fill={color}
        stroke={isSelected ? '#ffffff' : 'rgba(0,0,0,0.4)'}
        strokeWidth={isSelected ? 1.1 : 0.4}
      />
    </Marker>
  );
}
