import { motion } from 'framer-motion';
import { LayoutDashboard, LineChart, Globe, Newspaper, NotebookText, Settings, Sparkles, ChevronLeft } from 'lucide-react';
import { cn } from '../../utils/cn';

const nav = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'analysis', label: 'Analysis', icon: LineChart },
  { id: 'events', label: 'World Pulse', icon: Globe },
  { id: 'news', label: 'News & Sentiment', icon: Newspaper, badge: 'soon' },
  { id: 'journal', label: 'Journal', icon: NotebookText, badge: 'soon' },
];

const bottom = [
  { id: 'settings', label: 'Settings', icon: Settings },
];

export default function Sidebar({ current, onChange, collapsed, onToggle }) {
  return (
    <aside
      className={cn(
        'relative shrink-0 h-full border-r border-border bg-bg-elevated/40 backdrop-blur-xl',
        'flex flex-col transition-[width] duration-200 ease-out',
        collapsed ? 'w-[68px]' : 'w-[220px]',
      )}
    >
      <div className="flex items-center gap-2.5 h-16 px-4 border-b border-border">
        <div className="relative shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-accent-cyan to-accent-violet flex items-center justify-center shadow-glow">
          <Sparkles className="w-4 h-4 text-bg" strokeWidth={2.5} />
        </div>
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex flex-col leading-tight"
          >
            <span className="text-sm font-semibold text-text-primary">Trading Agent</span>
            <span className="text-[10px] text-text-muted font-mono uppercase tracking-wider">v1.0 · alpha</span>
          </motion.div>
        )}
      </div>

      <nav className="flex-1 px-2 py-3 space-y-0.5">
        {nav.map((item) => (
          <NavItem
            key={item.id}
            item={item}
            active={current === item.id}
            collapsed={collapsed}
            onClick={() => !item.badge && onChange(item.id)}
            disabled={!!item.badge}
          />
        ))}
      </nav>

      <div className="px-2 py-3 border-t border-border space-y-0.5">
        {bottom.map((item) => (
          <NavItem
            key={item.id}
            item={item}
            active={current === item.id}
            collapsed={collapsed}
            onClick={() => onChange(item.id)}
          />
        ))}
      </div>

      <button
        onClick={onToggle}
        aria-label="Toggle sidebar"
        className="absolute -right-3 top-20 w-6 h-6 rounded-full border border-border bg-bg-elevated hover:border-border-strong text-text-secondary hover:text-text-primary flex items-center justify-center transition-all"
      >
        <ChevronLeft
          className={cn('w-3.5 h-3.5 transition-transform', collapsed && 'rotate-180')}
        />
      </button>
    </aside>
  );
}

function NavItem({ item, active, collapsed, onClick, disabled }) {
  const Icon = item.icon;
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={collapsed ? item.label : undefined}
      className={cn(
        'group w-full flex items-center gap-3 px-2.5 h-9 rounded-lg text-sm font-medium transition-all',
        'focus-ring',
        active
          ? 'bg-white/5 text-text-primary border border-border-strong'
          : 'text-text-secondary hover:text-text-primary hover:bg-white/[0.03] border border-transparent',
        disabled && 'opacity-50 cursor-not-allowed hover:bg-transparent',
      )}
    >
      <Icon className={cn('w-4 h-4 shrink-0', active && 'text-accent-cyan')} />
      {!collapsed && (
        <>
          <span className="truncate">{item.label}</span>
          {item.badge && (
            <span className="ml-auto text-[9px] font-mono uppercase px-1.5 py-0.5 rounded bg-bg-subtle border border-border text-text-muted">
              {item.badge}
            </span>
          )}
        </>
      )}
    </button>
  );
}
