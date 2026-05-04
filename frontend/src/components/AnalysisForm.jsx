import { useState } from 'react';
import { analysisAPI } from '../services/api';
import ResultsDisplay from './ResultsDisplay';
import { TIME_PERIODS, DEFAULT_PERIOD, DEFAULT_DAYS_BACK } from '../config/constants';

function AnalysisForm({ initialSymbol }) {
  const [ticker, setTicker] = useState(initialSymbol || 'AAPL');
  const [selectedTypes, setSelectedTypes] = useState({
    technical: false,
    fundamental: false,
    news: false,
  });
  const [period, setPeriod] = useState(DEFAULT_PERIOD);
  const [daysBack, setDaysBack] = useState(DEFAULT_DAYS_BACK);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);

  const handleTypeChange = (type) => {
    setSelectedTypes((prev) => ({
      ...prev,
      [type]: !prev[type],
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const selected = Object.keys(selectedTypes).filter((key) => selectedTypes[key]);
    
    if (!ticker.trim()) {
      setError('Please enter a ticker symbol');
      return;
    }
    
    if (selected.length === 0) {
      setError('Please select at least one analysis type');
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);

    try {
      const analysisResults = [];
      
      for (const type of selected) {
        let result;
        if (type === 'technical') {
          result = await analysisAPI.technical(ticker, period);
        } else if (type === 'fundamental') {
          result = await analysisAPI.fundamental(ticker);
        } else if (type === 'news') {
          result = await analysisAPI.news(ticker, daysBack);
        }
        analysisResults.push(result);
      }
      
      setResults(analysisResults);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analysis-form-container">
      <div className="analysis-form glass-card">
        <h3 className="form-title">Run Stock Analysis</h3>
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Stock Ticker</label>
            <input
              type="text"
              className="form-input"
              placeholder="e.g., AAPL, TSLA, MSFT"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Analysis Types</label>
            <div className="checkbox-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={selectedTypes.technical}
                  onChange={() => handleTypeChange('technical')}
                />
                <span className="checkbox-text">📈 Technical Analysis</span>
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={selectedTypes.fundamental}
                  onChange={() => handleTypeChange('fundamental')}
                />
                <span className="checkbox-text">💰 Fundamental Analysis</span>
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={selectedTypes.news}
                  onChange={() => handleTypeChange('news')}
                />
                <span className="checkbox-text">📰 News Sentiment</span>
              </label>
            </div>
          </div>

          {selectedTypes.technical && (
            <div className="form-group" id="technical-options">
              <label className="form-label">Time Period</label>
              <select
                className="form-select"
                value={period}
                onChange={(e) => setPeriod(e.target.value)}
              >
                {TIME_PERIODS.map((p) => (
                  <option key={p.value} value={p.value}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>
          )}

          {selectedTypes.news && (
            <div className="form-group" id="news-options">
              <label className="form-label">Days Back</label>
              <input
                type="number"
                className="form-input"
                min="1"
                max="30"
                value={daysBack}
                onChange={(e) => setDaysBack(parseInt(e.target.value))}
              />
            </div>
          )}

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

          <button
            type="submit"
            className="btn btn-primary btn-large"
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              <>
                <span className="btn-icon">🚀</span>
                Run Analysis
              </>
            )}
          </button>
        </form>
      </div>

      {results.length > 0 && <ResultsDisplay results={results} />}
    </div>
  );
}

export default AnalysisForm;
