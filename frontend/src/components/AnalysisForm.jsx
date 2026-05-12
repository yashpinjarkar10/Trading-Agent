import { useState } from 'react';
import { LineChart, BarChart3, Newspaper, Loader2, Play, AlertTriangle } from 'lucide-react';
import { analysisAPI } from '../services/api';
import { TIME_PERIODS, DEFAULT_PERIOD, DEFAULT_DAYS_BACK } from '../config/constants';
import { Card, CardBody, CardHeader, CardTitle } from './ui/Card';
import { Input, Select, Checkbox } from './ui/Input';
import Button from './ui/Button';
import ResultsDisplay from './ResultsDisplay';

const TYPES = [
  { id: 'technical', label: 'Technical', icon: LineChart, desc: 'Indicators, trends, signals' },
  { id: 'fundamental', label: 'Fundamental', icon: BarChart3, desc: 'Valuation & financials' },
  { id: 'news', label: 'News & Sentiment', icon: Newspaper, desc: 'Latest catalysts' },
];

export default function AnalysisForm({ initialSymbol = 'AAPL' }) {
  const [ticker, setTicker] = useState(initialSymbol);
  const [selected, setSelected] = useState({ technical: true, fundamental: false, news: false });
  const [period, setPeriod] = useState(DEFAULT_PERIOD);
  const [daysBack, setDaysBack] = useState(DEFAULT_DAYS_BACK);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);

  const toggle = (id) => setSelected((s) => ({ ...s, [id]: !s[id] }));

  const submit = async (e) => {
    e.preventDefault();
    const picks = Object.keys(selected).filter((k) => selected[k]);
    if (!ticker.trim()) return setError('Enter a ticker');
    if (picks.length === 0) return setError('Pick at least one analysis');
    setError(null); setLoading(true); setResults([]);
    try {
      const out = [];
      for (const t of picks) {
        if (t === 'technical') out.push(await analysisAPI.technical(ticker, period));
        else if (t === 'fundamental') out.push(await analysisAPI.fundamental(ticker));
        else if (t === 'news') out.push(await analysisAPI.news(ticker, daysBack));
      }
      setResults(out);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed');
    } finally { setLoading(false); }
  };

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader>
          <CardTitle icon={LineChart}>Run Analysis</CardTitle>
          <span className="chip">Multi-modal</span>
        </CardHeader>
        <CardBody>
          <form onSubmit={submit} className="space-y-5">
            <div>
              <label className="block text-xs uppercase tracking-wide text-text-muted mb-2">Ticker</label>
              <Input
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="AAPL, BRK.B, BTC-USD…"
                autoComplete="off"
              />
            </div>

            <div>
              <label className="block text-xs uppercase tracking-wide text-text-muted mb-2">Analysis types</label>
              <div className="grid sm:grid-cols-3 gap-2">
                {TYPES.map((t) => {
                  const Icon = t.icon;
                  const active = selected[t.id];
                  return (
                    <button
                      type="button"
                      key={t.id}
                      onClick={() => toggle(t.id)}
                      className={`text-left p-3 rounded-xl border transition-all focus-ring ${
                        active
                          ? 'border-accent-cyan/40 bg-accent-cyan/5'
                          : 'border-border bg-bg-subtle hover:border-border-strong'
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <Icon className={`w-4 h-4 ${active ? 'text-accent-cyan' : 'text-text-secondary'}`} />
                        <span className="text-sm font-medium text-text-primary">{t.label}</span>
                      </div>
                      <p className="text-xs text-text-muted">{t.desc}</p>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="grid sm:grid-cols-2 gap-4">
              {selected.technical && (
                <div>
                  <label className="block text-xs uppercase tracking-wide text-text-muted mb-2">Period</label>
                  <Select value={period} onChange={(e) => setPeriod(e.target.value)}>
                    {TIME_PERIODS.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
                  </Select>
                </div>
              )}
              {selected.news && (
                <div>
                  <label className="block text-xs uppercase tracking-wide text-text-muted mb-2">Days back</label>
                  <Input type="number" min="1" max="30" value={daysBack} onChange={(e) => setDaysBack(parseInt(e.target.value || '7', 10))} />
                </div>
              )}
            </div>

            {error && (
              <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-accent-red/30 bg-accent-red/5 text-sm text-accent-red">
                <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <Button type="submit" size="lg" disabled={loading} className="w-full sm:w-auto">
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              {loading ? 'Analyzing…' : 'Run analysis'}
            </Button>
          </form>
        </CardBody>
      </Card>

      {results.length > 0 && <ResultsDisplay results={results} />}
    </div>
  );
}
