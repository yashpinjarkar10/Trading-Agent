import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Globe as GlobeIcon, Map as MapIcon, Loader2 } from 'lucide-react';
import { eventsAPI } from '../services/events';
import EventGlobe from '../components/events/EventGlobe';
import EventMap2D from '../components/events/EventMap2D';
import FilterSidebar, { loadFilters } from '../components/events/FilterSidebar';
import RecentList from '../components/events/RecentList';
import CategoryLegend from '../components/events/CategoryLegend';
import EventDetailPanel from '../components/events/EventDetailPanel';
import Badge from '../components/ui/Badge';
import { cn } from '../utils/cn';

/**
 * EventsPage — World Pulse. 3D globe + 2D flat map renderer with a toggle.
 *
 * Phase 3 — items 3.4, 3.5 + 2D/3D toggle.
 */

const VIEWMODE_KEY = 'tradingagent.eventViewMode.v1';

function loadViewMode() {
  try {
    const v = localStorage.getItem(VIEWMODE_KEY);
    if (v === '2d' || v === '3d') return v;
  } catch { /* noop */ }
  // Sensible defaults: reduced motion or narrow viewport → 2D
  if (typeof window !== 'undefined') {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return '2d';
    if (window.innerWidth < 768) return '2d';
  }
  return '3d';
}

function saveViewMode(mode) {
  try { localStorage.setItem(VIEWMODE_KEY, mode); } catch { /* noop */ }
}

export default function EventsPage() {
  const [filters, setFilters] = useState(() => loadFilters());
  const [viewMode, setViewMode] = useState(() => loadViewMode());
  const [selectedId, setSelectedId] = useState(null);

  useEffect(() => { saveViewMode(viewMode); }, [viewMode]);

  // Map UI filters → API filter shape
  const apiFilters = useMemo(() => {
    const hours = { '1h': 1, '24h': 24, '7d': 24 * 7, '30d': 24 * 30 }[filters.timeRangeId] ?? 24 * 7;
    return {
      since: new Date(Date.now() - hours * 3600 * 1000).toISOString(),
      categories: filters.categories,
      min_severity: filters.minSeverity,
      min_market_impact: filters.minMarketImpact,
      limit: 2000,
    };
  }, [filters]);

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ['events', apiFilters],
    queryFn: () => eventsAPI.list(apiFilters),
  });

  const events = data?.items ?? [];
  const selected = events.find((e) => e.id === selectedId) ?? null;

  useEffect(() => {
    if (selectedId && !events.find((e) => e.id === selectedId)) setSelectedId(null);
  }, [events, selectedId]);

  return (
    <div className="h-full p-4 lg:p-5 grid gap-4 grid-cols-1 lg:grid-cols-[320px_1fr] min-h-0">
      {/* LEFT — filters + list */}
      <aside className="flex flex-col gap-4 min-h-0">
        <FilterSidebar filters={filters} onChange={setFilters} />
        <div className="flex-1 min-h-[320px] lg:min-h-0">
          <RecentList
            events={events}
            selectedId={selectedId}
            onSelect={(e) => setSelectedId(e.id)}
          />
        </div>
      </aside>

      {/* RIGHT — viz + detail */}
      <section className="flex flex-col gap-4 min-h-0">
        {/* Header strip */}
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-accent-cyan/10 border border-accent-cyan/30 flex items-center justify-center">
              <GlobeIcon className="w-4 h-4 text-accent-cyan" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-text-primary leading-tight">World Pulse</h1>
              <p className="text-xs text-text-muted">Live map of market-moving global events</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <ViewModeToggle mode={viewMode} onChange={setViewMode} />
            {isFetching && !isLoading && (
              <Badge>
                <Loader2 className="w-3 h-3 animate-spin" /> refreshing
              </Badge>
            )}
            <Badge tone="cyan" dot>{events.length} events</Badge>
            <Badge tone="amber">dummy data</Badge>
          </div>
        </div>

        {/* Visualization */}
        <div className="relative flex-1 min-h-[420px] rounded-2xl border border-border bg-bg-elevated/30 overflow-hidden">
          {isLoading ? (
            <VizSkeleton mode={viewMode} />
          ) : events.length === 0 ? (
            <EmptyState />
          ) : (
            <>
              {viewMode === '3d' ? (
                <EventGlobe events={events} selectedId={selectedId} onSelect={(p) => setSelectedId(p.id)} />
              ) : (
                <EventMap2D events={events} selectedId={selectedId} onSelect={(p) => setSelectedId(p.id)} />
              )}
              <CategoryLegend />
            </>
          )}
        </div>

        {/* Detail panel */}
        <EventDetailPanel event={selected} onClose={() => setSelectedId(null)} />
      </section>
    </div>
  );
}

function ViewModeToggle({ mode, onChange }) {
  return (
    <div
      role="tablist"
      aria-label="Map view mode"
      className="inline-flex items-center rounded-lg border border-border bg-bg-subtle p-0.5"
    >
      <ToggleBtn active={mode === '3d'} onClick={() => onChange('3d')} icon={GlobeIcon} label="3D" />
      <ToggleBtn active={mode === '2d'} onClick={() => onChange('2d')} icon={MapIcon}    label="2D" />
    </div>
  );
}

function ToggleBtn({ active, onClick, icon: Icon, label }) {
  return (
    <button
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md text-xs font-medium transition-colors cursor-pointer focus-ring',
        active
          ? 'bg-accent-cyan/15 text-accent-cyan'
          : 'text-text-secondary hover:text-text-primary',
      )}
    >
      <Icon className="w-3.5 h-3.5" />
      {label}
    </button>
  );
}

function VizSkeleton({ mode }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center">
      {mode === '3d' ? (
        <div className="relative w-[380px] h-[380px] rounded-full bg-gradient-to-br from-accent-cyan/10 via-bg-elevated to-accent-violet/10 animate-pulse-soft" />
      ) : (
        <div className="w-[80%] h-[60%] rounded-2xl bg-gradient-to-br from-accent-cyan/5 via-bg-elevated to-accent-violet/5 animate-pulse-soft" />
      )}
      <div className="absolute bottom-6 text-xs text-text-muted font-mono">loading map…</div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-8">
      <GlobeIcon className="w-10 h-10 text-text-muted mb-3" />
      <h3 className="text-base font-semibold text-text-primary mb-1">No events match your filters</h3>
      <p className="text-sm text-text-muted max-w-sm">
        Try widening the time range, lowering severity / impact thresholds, or enabling more categories.
      </p>
    </div>
  );
}
