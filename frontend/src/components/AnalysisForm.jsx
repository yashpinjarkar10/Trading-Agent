import { useState } from 'react';
import { LineChart, BarChart3, Newspaper, MessageSquare, Loader2, Play, AlertTriangle } from 'lucide-react';
import { analysisAPI } from '../services/api';
import { Card, CardBody, CardHeader, CardTitle } from './ui/Card';
import { Input, Select, Checkbox } from './ui/Input';
import Button from './ui/Button';
import ResultsDisplay from './ResultsDisplay';

const TYPES = [
  { id: 'market', label: 'Market', icon: LineChart, desc: 'Technical indicators & trends' },
  { id: 'fundamentals', label: 'Fundamentals', icon: BarChart3, desc: 'Financials & valuations' },
  { id: 'news', label: 'News', icon: Newspaper, desc: 'Latest catalysts & events' },
  { id: 'sentiment', label: 'Sentiment', icon: MessageSquare, desc: 'Social & market mood' },
];

export default function AnalysisForm({ initialSymbol = 'AAPL' }) {
  const [ticker, setTicker] = useState(initialSymbol);
  const [selected, setSelected] = useState({ market: true, fundamentals: false, news: false, sentiment: false });
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
        if (t === 'market') out.push(await analysisAPI.market(ticker));
        else if (t === 'fundamentals') out.push(await analysisAPI.fundamentals(ticker));
        else if (t === 'news') out.push(await analysisAPI.news(ticker));
        else if (t === 'sentiment') out.push(await analysisAPI.sentiment(ticker));
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
              <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-2">
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
