import { useEffect, useState } from 'react';
import { Settings as SettingsIcon, Github, ExternalLink } from 'lucide-react';
import AppShell from './components/layout/AppShell';
import ChartView from './components/ChartView';
import ChatInterface from './components/ChatInterface';
import AnalysisForm from './components/AnalysisForm';
import EventsPage from './pages/EventsPage';
import { OPEN_TICKER_EVENT } from './services/events';
import { Card, CardBody, CardHeader, CardTitle } from './components/ui/Card';
import Badge from './components/ui/Badge';
import './styles/index.css';

export default function App() {
  const [page, setPage] = useState('dashboard');
  const [symbol, setSymbol] = useState('AAPL');

  // Phase 3E — 3.16: ticker chip in event detail panel switches the chart symbol
  // and navigates to the Dashboard.
  useEffect(() => {
    const handler = (e) => {
      const t = e?.detail?.ticker;
      if (!t) return;
      setSymbol(String(t).toUpperCase());
      setPage('dashboard');
    };
    window.addEventListener(OPEN_TICKER_EVENT, handler);
    return () => window.removeEventListener(OPEN_TICKER_EVENT, handler);
  }, []);

  return (
    <AppShell
      current={page}
      onChange={setPage}
      symbol={symbol}
      onSymbolChange={setSymbol}
    >
      {page === 'dashboard' && <Dashboard symbol={symbol} />}
      {page === 'analysis' && <AnalysisPage symbol={symbol} />}
      {page === 'events' && <EventsPage />}
      {page === 'settings' && <SettingsPage />}
    </AppShell>
  );
}

function Dashboard({ symbol }) {
  return (
    <div className="h-full p-5 grid gap-5 grid-cols-1 lg:grid-cols-[1.4fr_1fr]">
      <div className="min-h-[520px] lg:min-h-0">
        <ChartView symbol={symbol} />
      </div>
      <div className="min-h-[520px] lg:min-h-0">
        <ChatInterface />
      </div>
    </div>
  );
}

function AnalysisPage({ symbol }) {
  return (
    <div className="p-5 max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-text-primary mb-1">Multi-modal Analysis</h1>
        <p className="text-sm text-text-secondary">
          Run technical, fundamental, and sentiment analysis side-by-side.
        </p>
      </div>
      <AnalysisForm initialSymbol={symbol} />
    </div>
  );
}

function SettingsPage() {
  return (
    <div className="p-5 max-w-3xl mx-auto space-y-5">
      <h1 className="text-2xl font-semibold text-text-primary">Settings</h1>

      <Card>
        <CardHeader><CardTitle icon={SettingsIcon}>Environment</CardTitle></CardHeader>
        <CardBody className="space-y-3">
          <Row label="API base" value={import.meta.env.VITE_API_URL || 'http://localhost:8000'} />
          <Row label="Build mode" value={import.meta.env.MODE} />
          <Row label="Version" value="1.0.0-alpha" />
        </CardBody>
      </Card>

      <Card>
        <CardHeader><CardTitle>About</CardTitle></CardHeader>
        <CardBody className="space-y-3 text-sm text-text-secondary">
          <p>
            Trading Agent is an AI-native trading copilot that combines real-time TradingView
            charts with an LLM-powered analyst for technical, fundamental, and news analysis.
          </p>
          <div className="flex flex-wrap gap-2 pt-2">
            <Badge tone="cyan">React 18</Badge>
            <Badge tone="violet">FastAPI</Badge>
            <Badge tone="green">LangGraph</Badge>
            <Badge tone="amber">TradingView</Badge>
            <Badge>Gemini</Badge>
          </div>
          <div className="flex items-center gap-3 pt-2 text-xs">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-text-secondary hover:text-accent-cyan transition-colors"
            >
              <ExternalLink className="w-3.5 h-3.5" /> API docs
            </a>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-text-secondary hover:text-accent-cyan transition-colors"
            >
              <Github className="w-3.5 h-3.5" /> Source
            </a>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <span className="text-text-muted">{label}</span>
      <span className="font-mono text-text-secondary">{value}</span>
    </div>
  );
}
