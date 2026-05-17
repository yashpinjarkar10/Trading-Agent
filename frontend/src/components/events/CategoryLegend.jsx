import { CATEGORIES } from '../../services/dummyEvents';

/**
 * CategoryLegend — small overlay shown on the globe explaining dot colors.
 * Phase 3D — item 3.13
 */
export default function CategoryLegend() {
  return (
    <div className="absolute bottom-4 left-4 z-10 max-w-[260px] rounded-xl border border-border bg-bg-elevated/80 backdrop-blur-md p-3">
      <div className="text-[10px] uppercase tracking-wider text-text-muted mb-2 font-mono">
        Legend
      </div>
      <ul className="grid grid-cols-2 gap-x-3 gap-y-1.5">
        {CATEGORIES.map((c) => (
          <li key={c.id} className="flex items-center gap-2 text-[11px] text-text-secondary">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ background: c.color, boxShadow: `0 0 6px ${c.color}99` }}
              aria-hidden
            />
            <span className="truncate">{c.label}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
