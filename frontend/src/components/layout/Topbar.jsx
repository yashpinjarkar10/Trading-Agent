import { useEffect, useState } from 'react';
import { Search, Activity, Github } from 'lucide-react';
import Badge from '../ui/Badge';
import { healthAPI } from '../../services/api';

export default function Topbar({ symbol, onSymbolChange }) {
  const [input, setInput] = useState(symbol || '');
  const [healthy, setHealthy] = useState(null);

  useEffect(() => setInput(symbol || ''), [symbol]);

  useEffect(() => {
    let cancelled = false;
    healthAPI.check().then(() => !cancelled && setHealthy(true)).catch(() => !cancelled && setHealthy(false));
    const t = setInterval(() => {
      healthAPI.check().then(() => !cancelled && setHealthy(true)).catch(() => !cancelled && setHealthy(false));
    }, 30000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  const submit = (e) => {
    e.preventDefault();
    const v = input.trim().toUpperCase();
    if (v) onSymbolChange?.(v);
  };

  return (
    <header className="h-16 shrink-0 px-5 border-b border-border bg-bg-elevated/30 backdrop-blur-xl flex items-center gap-4">
      <form onSubmit={submit} className="flex-1 max-w-xl">
        <div className="relative">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted pointer-events-none" />
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Search ticker or ask anything…  (e.g. AAPL, BRK.B, BTC-USD)"
            className="w-full h-10 pl-10 pr-20 rounded-lg bg-bg-subtle border border-border text-sm text-text-primary placeholder:text-text-muted focus-ring focus:border-accent-cyan/50 transition-colors"
          />
          <kbd className="hidden sm:inline-flex absolute right-3 top-1/2 -translate-y-1/2 items-center gap-1 px-1.5 h-5 rounded border border-border bg-bg text-[10px] font-mono text-text-muted">
            <span>↵</span>
          </kbd>
        </div>
      </form>

      <div className="flex items-center gap-2">
        <Badge tone={healthy ? 'green' : healthy === false ? 'red' : 'neutral'} dot>
          <Activity className="w-3 h-3" />
          {healthy === null ? 'connecting' : healthy ? 'live' : 'offline'}
        </Badge>
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className="hidden sm:inline-flex h-9 w-9 items-center justify-center rounded-lg border border-border text-text-secondary hover:text-text-primary hover:border-border-strong transition-colors"
          aria-label="GitHub"
        >
          <Github className="w-4 h-4" />
        </a>
      </div>
    </header>
  );
}
