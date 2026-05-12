import { useEffect, useRef } from 'react';
import { Card, CardBody, CardHeader, CardTitle } from './ui/Card';
import { CandlestickChart } from 'lucide-react';
import Badge from './ui/Badge';

function loadTradingView() {
  return new Promise((resolve) => {
    if (window.TradingView) return resolve(window.TradingView);
    const existing = document.getElementById('tv-script');
    if (existing) {
      existing.addEventListener('load', () => resolve(window.TradingView));
      return;
    }
    const s = document.createElement('script');
    s.id = 'tv-script';
    s.src = 'https://s3.tradingview.com/tv.js';
    s.async = true;
    s.onload = () => resolve(window.TradingView);
    document.body.appendChild(s);
  });
}

export default function ChartView({ symbol = 'AAPL', height = '100%' }) {
  const containerId = useRef(`tv_chart_${Math.random().toString(36).slice(2)}`).current;

  useEffect(() => {
    let cancelled = false;
    loadTradingView().then((TV) => {
      if (cancelled || !TV) return;
      const el = document.getElementById(containerId);
      if (el) el.innerHTML = '';
      // Bug #7: accept bare ticker; TV auto-resolves exchange.
      const s = String(symbol || 'AAPL').trim().toUpperCase();
      new TV.widget({
        autosize: true,
        symbol: s,
        interval: 'D',
        timezone: 'Etc/UTC',
        theme: 'dark',
        style: '1',
        locale: 'en',
        toolbar_bg: '#0d0f1a',
        enable_publishing: false,
        allow_symbol_change: true,
        container_id: containerId,
        studies: ['MASimple@tv-basicstudies', 'RSI@tv-basicstudies'],
        hide_side_toolbar: false,
        save_image: false,
        backgroundColor: 'rgba(13, 15, 26, 0)',
        gridColor: 'rgba(255, 255, 255, 0.04)',
      });
    });
    return () => { cancelled = true; };
  }, [symbol, containerId]);

  return (
    <Card className="flex flex-col h-full overflow-hidden">
      <CardHeader>
        <CardTitle icon={CandlestickChart}>Live Chart</CardTitle>
        <div className="flex items-center gap-2">
          <Badge tone="cyan" dot>{symbol}</Badge>
          <Badge tone="neutral">TradingView</Badge>
        </div>
      </CardHeader>
      <div className="flex-1 min-h-0 relative">
        <div id={containerId} style={{ height, width: '100%' }} className="absolute inset-0" />
      </div>
    </Card>
  );
}
