import { useEffect } from 'react';
import { Filter, RotateCcw } from 'lucide-react';
import { CATEGORIES } from '../../services/dummyEvents';
import { Card, CardBody, CardHeader, CardTitle } from '../ui/Card';
import { Checkbox } from '../ui/Input';
import Button from '../ui/Button';
import { cn } from '../../utils/cn';

const TIME_RANGES = [
  { id: '1h',  label: '1h',  hours: 1 },
  { id: '24h', label: '24h', hours: 24 },
  { id: '7d',  label: '7d',  hours: 24 * 7 },
  { id: '30d', label: '30d', hours: 24 * 30 },
];

export const DEFAULT_FILTERS = {
  categories: CATEGORIES.map((c) => c.id),
  minSeverity: 1,
  minMarketImpact: 1,
  timeRangeId: '7d',
};

const LS_KEY = 'tradingagent.eventFilters.v1';

export function loadFilters() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return DEFAULT_FILTERS;
    return { ...DEFAULT_FILTERS, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_FILTERS;
  }
}
export function saveFilters(f) {
  try { localStorage.setItem(LS_KEY, JSON.stringify(f)); } catch { /* noop */ }
}

export default function FilterSidebar({ filters, onChange }) {
  // Phase 3D — 3.14: persist on every change
  useEffect(() => { saveFilters(filters); }, [filters]);

  const toggleCategory = (id) => {
    const next = filters.categories.includes(id)
      ? filters.categories.filter((c) => c !== id)
      : [...filters.categories, id];
    onChange({ ...filters, categories: next });
  };

  const reset = () => onChange(DEFAULT_FILTERS);

  return (
    <Card>
      <CardHeader>
        <CardTitle icon={Filter}>Filters</CardTitle>
        <Button variant="ghost" size="sm" onClick={reset} title="Reset filters">
          <RotateCcw className="w-3.5 h-3.5" /> Reset
        </Button>
      </CardHeader>

      <CardBody className="space-y-5">
        {/* Time range */}
        <div>
          <Label>Time range</Label>
          <div className="grid grid-cols-4 gap-1.5">
            {TIME_RANGES.map((t) => {
              const active = filters.timeRangeId === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => onChange({ ...filters, timeRangeId: t.id })}
                  className={cn(
                    'h-8 rounded-md text-xs font-medium border transition-colors cursor-pointer focus-ring',
                    active
                      ? 'border-accent-cyan/50 bg-accent-cyan/10 text-accent-cyan'
                      : 'border-border bg-bg-subtle text-text-secondary hover:text-text-primary hover:border-border-strong',
                  )}
                >
                  {t.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Severity slider */}
        <SliderRow
          label="Min severity"
          value={filters.minSeverity}
          onChange={(v) => onChange({ ...filters, minSeverity: v })}
        />

        {/* Market impact slider */}
        <SliderRow
          label="Min market impact"
          value={filters.minMarketImpact}
          onChange={(v) => onChange({ ...filters, minMarketImpact: v })}
        />

        {/* Categories */}
        <div>
          <Label>Categories</Label>
          <div className="space-y-1.5">
            {CATEGORIES.map((c) => (
              <div key={c.id} className="flex items-center justify-between">
                <Checkbox
                  checked={filters.categories.includes(c.id)}
                  onChange={() => toggleCategory(c.id)}
                  label={c.label}
                />
                <span
                  className="w-2.5 h-2.5 rounded-full shrink-0"
                  style={{ background: c.color, boxShadow: `0 0 8px ${c.color}80` }}
                  aria-hidden
                />
              </div>
            ))}
          </div>
        </div>
      </CardBody>
    </Card>
  );
}

function Label({ children }) {
  return <div className="text-xs uppercase tracking-wide text-text-muted mb-2">{children}</div>;
}

function SliderRow({ label, value, onChange }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs uppercase tracking-wide text-text-muted">{label}</span>
        <span className="text-xs font-mono text-text-primary">{value}+</span>
      </div>
      <input
        type="range"
        min="1"
        max="10"
        step="1"
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
        className="w-full accent-accent-cyan cursor-pointer"
      />
    </div>
  );
}
