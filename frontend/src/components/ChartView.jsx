import { useState, useEffect } from 'react';

function ChartView({ currentSymbol, setCurrentSymbol, switchToAnalyst }) {
  const [symbolInput, setSymbolInput] = useState(currentSymbol);

  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/tv.js';
    script.async = true;
    script.onload = () => initializeTradingView();
    document.body.appendChild(script);

    return () => {
      if (document.body.contains(script)) {
        document.body.removeChild(script);
      }
    };
  }, [currentSymbol]);

  // Bug #7: do NOT hard-code NASDAQ. Accept either an explicit
  // "EXCHANGE:TICKER" form (e.g. "NYSE:JPM", "NSE:RELIANCE", "BINANCE:BTCUSDT")
  // or a bare ticker — TradingView will auto-resolve the latter to its primary
  // listing exchange. This unblocks NYSE, LSE, NSE, HKEX, crypto, FX, etc.
  const resolveSymbol = (input) => {
    if (!input) return 'AAPL';
    const trimmed = String(input).trim().toUpperCase();
    return trimmed.includes(':') ? trimmed : trimmed;
  };

  const initializeTradingView = () => {
    if (typeof TradingView !== 'undefined') {
      new TradingView.widget({
        autosize: true,
        symbol: resolveSymbol(currentSymbol),
        interval: 'D',
        timezone: 'Etc/UTC',
        theme: 'dark',
        style: '1',
        locale: 'en',
        toolbar_bg: '#0a0b0f',
        enable_publishing: false,
        allow_symbol_change: true,
        container_id: 'tradingview_chart',
        studies: [
          'MASimple@tv-basicstudies',
          'RSI@tv-basicstudies',
          'MACD@tv-basicstudies'
        ],
        hide_side_toolbar: false,
        save_image: false,
      });
    }
  };

  const handleUpdateSymbol = () => {
    const newSymbol = symbolInput.trim().toUpperCase();
    if (newSymbol) {
      setCurrentSymbol(newSymbol);
    }
  };

  const handleQuickAnalysis = (type) => {
    switchToAnalyst();
  };

  return (
    <div className="chart-section">
      <div className="section-header">
        <h2 className="section-title">
          <span className="section-icon">📊</span>
          Live Market Chart
        </h2>
        <div className="symbol-display">
          <span className="symbol-label">Current Symbol:</span>
          <span className="symbol-value" id="current-symbol">{currentSymbol}</span>
        </div>
      </div>

      <div className="chart-container glass-card">
        <div className="chart-controls">
          <div className="symbol-input-group">
            <input
              type="text"
              className="symbol-input"
              placeholder="Enter symbol (e.g. AAPL, NYSE:JPM, NSE:RELIANCE, BINANCE:BTCUSDT)"
              value={symbolInput}
              onChange={(e) => setSymbolInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleUpdateSymbol()}
            />
            <button className="btn btn-primary" onClick={handleUpdateSymbol}>
              Update Chart
            </button>
          </div>
        </div>

        <div id="tradingview_chart" style={{ height: '600px', width: '100%' }}></div>
      </div>

      <div className="quick-actions glass-card">
        <h3 className="quick-actions-title">Quick Analysis</h3>
        <div className="quick-actions-grid">
          <button
            className="analysis-btn technical"
            onClick={() => handleQuickAnalysis('technical')}
          >
            <span className="btn-icon">📈</span>
            <span className="btn-text">Technical Analysis</span>
          </button>
          <button
            className="analysis-btn fundamental"
            onClick={() => handleQuickAnalysis('fundamental')}
          >
            <span className="btn-icon">💰</span>
            <span className="btn-text">Fundamental Analysis</span>
          </button>
          <button
            className="analysis-btn news"
            onClick={() => handleQuickAnalysis('news')}
          >
            <span className="btn-icon">📰</span>
            <span className="btn-text">News Sentiment</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChartView;
