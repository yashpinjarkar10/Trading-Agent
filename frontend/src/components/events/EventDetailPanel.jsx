import { ExternalLink, MapPin, Clock, TrendingUp, ShieldAlert, X } from 'lucide-react';
import { Card, CardBody, CardHeader, CardTitle } from '../ui/Card';
import Badge from '../ui/Badge';
import Button from '../ui/Button';
import { CATEGORY_COLOR, CATEGORY_LABEL } from '../../services/dummyEvents';
import { emitOpenTicker } from '../../services/events';

/**
 * EventDetailPanel — full information about the selected event.
 * Clicking an affected_ticker chip dispatches a global event the chart listens for.
 *
 * Phase 3E — items 3.15, 3.16
 */
export default function EventDetailPanel({ event, onClose }) {
  if (!event) {
    return (
      <Card>
        <CardBody className="text-sm text-text-muted text-center py-8">
          Click an event on the globe or in the list to see details.
        </CardBody>
      </Card>
    );
  }

  const color = CATEGORY_COLOR[event.category] || '#94a3b8';

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="w-2.5 h-2.5 rounded-full shrink-0"
            style={{ background: color, boxShadow: `0 0 8px ${color}aa` }}
            aria-hidden
          />
          <CardTitle className="truncate">{event.title}</CardTitle>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close detail">
          <X className="w-4 h-4" />
        </Button>
      </CardHeader>

      <CardBody className="space-y-4">
        {/* Meta row */}
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="cyan">{CATEGORY_LABEL[event.category] || event.category}</Badge>
          {event.subcategory && <Badge>{event.subcategory}</Badge>}
          <Badge tone={event.severity >= 7 ? 'red' : event.severity >= 5 ? 'amber' : 'neutral'}>
            <ShieldAlert className="w-3 h-3" /> sev {event.severity ?? '?'}
          </Badge>
          <Badge tone={event.market_impact >= 7 ? 'violet' : 'neutral'}>
            <TrendingUp className="w-3 h-3" /> impact {event.market_impact ?? '?'}
          </Badge>
        </div>

        {/* Description */}
        {event.description && (
          <p className="text-sm text-text-secondary leading-relaxed">
            {event.description}
          </p>
        )}

        {/* Location & time */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
          <Row icon={MapPin} label={event.location_name || '—'} />
          <Row icon={Clock} label={new Date(event.occurred_at).toLocaleString()} />
        </div>

        {/* Affected sectors */}
        {event.affected_sectors?.length > 0 && (
          <Section title="Affected sectors">
            <div className="flex flex-wrap gap-1.5">
              {event.affected_sectors.map((s) => (
                <span key={s} className="chip capitalize">{s.replace(/_/g, ' ')}</span>
              ))}
            </div>
          </Section>
        )}

        {/* Affected tickers — clickable */}
        {event.affected_tickers?.length > 0 && (
          <Section title="Affected tickers">
            <div className="flex flex-wrap gap-1.5">
              {event.affected_tickers.map((t) => (
                <button
                  key={t}
                  onClick={() => emitOpenTicker(t)}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-accent-cyan/30 bg-accent-cyan/5 text-accent-cyan text-xs font-mono hover:bg-accent-cyan/10 hover:border-accent-cyan/50 transition-colors cursor-pointer focus-ring"
                  title={`Open ${t} in chart`}
                >
                  {t}
                  <ExternalLink className="w-3 h-3" />
                </button>
              ))}
            </div>
            <p className="mt-2 text-[11px] text-text-muted">
              Click a ticker to load it on the Dashboard chart.
            </p>
          </Section>
        )}

        {/* Source */}
        <div className="pt-2 border-t border-border flex items-center justify-between text-xs">
          <span className="text-text-muted">
            Source: <span className="font-mono uppercase">{event.source}</span>
          </span>
          {event.source_url && (
            <a
              href={event.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-accent-cyan hover:text-accent-violet transition-colors"
            >
              Open source <ExternalLink className="w-3 h-3" />
            </a>
          )}
        </div>
      </CardBody>
    </Card>
  );
}

function Row({ icon: Icon, label }) {
  return (
    <div className="flex items-center gap-2 text-text-secondary">
      <Icon className="w-3.5 h-3.5 text-text-muted shrink-0" />
      <span className="truncate">{label}</span>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-text-muted mb-1.5 font-mono">
        {title}
      </div>
      {children}
    </div>
  );
}
