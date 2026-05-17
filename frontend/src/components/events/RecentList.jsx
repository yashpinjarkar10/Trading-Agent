import { Clock, MapPin } from 'lucide-react';
import { Card, CardBody, CardHeader, CardTitle } from '../ui/Card';
import Badge from '../ui/Badge';
import { CATEGORY_COLOR, CATEGORY_LABEL } from '../../services/dummyEvents';
import { cn } from '../../utils/cn';

/**
 * RecentList — vertical list of events ordered by time, clicking selects and
 * focuses the globe. Virtualization deferred until row count > 200 (current
 * dummy set is 35; real backend will paginate via cursor).
 *
 * Phase 3D — item 3.12
 */
export default function RecentList({ events, selectedId, onSelect }) {
  return (
    <Card className="flex flex-col h-full overflow-hidden">
      <CardHeader>
        <CardTitle icon={Clock}>Recent events</CardTitle>
        <Badge tone="cyan">{events.length}</Badge>
      </CardHeader>
      <div className="flex-1 min-h-0 overflow-y-auto">
        {events.length === 0 ? (
          <div className="p-6 text-center text-sm text-text-muted">
            No events match your filters.
          </div>
        ) : (
          <ul className="divide-y divide-border">
            {events.map((e) => (
              <li key={e.id}>
                <button
                  onClick={() => onSelect(e)}
                  className={cn(
                    'w-full text-left px-4 py-3 flex gap-3 items-start transition-colors cursor-pointer focus-ring',
                    e.id === selectedId
                      ? 'bg-accent-cyan/5'
                      : 'hover:bg-white/[0.03]',
                  )}
                >
                  <span
                    className="mt-1.5 w-2.5 h-2.5 rounded-full shrink-0"
                    style={{
                      background: CATEGORY_COLOR[e.category],
                      boxShadow: `0 0 8px ${CATEGORY_COLOR[e.category]}80`,
                    }}
                    aria-hidden
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-[10px] uppercase tracking-wide text-text-muted font-mono">
                        {CATEGORY_LABEL[e.category] || e.category}
                      </span>
                      <span className="text-[10px] text-text-muted">·</span>
                      <span className="text-[10px] text-text-muted">{timeAgo(e.occurred_at)}</span>
                    </div>
                    <h4 className="text-sm font-medium text-text-primary truncate">{e.title}</h4>
                    <div className="flex items-center gap-1 mt-1 text-[11px] text-text-muted">
                      <MapPin className="w-3 h-3 shrink-0" />
                      <span className="truncate">{e.location_name}</span>
                    </div>
                  </div>
                  <SeverityPill value={e.severity} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}

function SeverityPill({ value = 0 }) {
  const tone = value >= 8 ? 'red' : value >= 6 ? 'amber' : value >= 4 ? 'cyan' : 'neutral';
  return <Badge tone={tone}>sev {value}</Badge>;
}

function timeAgo(iso) {
  const t = new Date(iso).getTime();
  const s = Math.max(1, Math.floor((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}
