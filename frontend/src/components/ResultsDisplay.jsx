import { useState } from 'react';
import SafeHTML from './SafeHTML';

function ResultsDisplay({ results }) {
  const [expandedIndex, setExpandedIndex] = useState(0);

  const formatAnalysisOutput = (text) => {
    if (!text) return '';
    
    return text
      .replace(/^={40,}$/gm, '<div class="section-divider"></div>')
      .replace(/^([A-Z\s]+ANALYSIS\s+REPORT:\s+[A-Z]+)$/gm, '<h1 class="analysis-title">$1</h1>')
      .replace(/^(Generated on:\s*)(.+)$/gm, '<div class="generated-info">$1<span class="timestamp">$2</span></div>')
      .replace(/^([📊🔍💰📈👥⭐📝⚠️🎯]\s*[A-Z\s&]+)$/gm, '<h2 class="section-header">$1</h2>')
      .replace(/^-{20,}$/gm, '<div class="subsection-divider"></div>')
      .replace(/^\s*•\s*(.+)$/gm, '<div class="bullet-point">• $1</div>')
      .replace(/^\s*✓\s*(.+)$/gm, '<div class="success-point">✓ $1</div>')
      .replace(/^(\d+\.\s*)(.+)$/gm, '<div class="numbered-point">$1$2</div>')
      .replace(/^(\s*)([A-Za-z\s\(\)]+):\s*(.+)$/gm, (match, indent, key, value) => {
        if (key.match(/^[A-Z\s]+$/) || match.includes('📊') || match.includes('💰') || 
            key.includes('Metrics') || key.includes('Information') || key.includes('Analysis') ||
            key.includes('Factors') || key.includes('Recommendation') || key.includes('Generated')) {
          return match;
        }
        return `${indent}<div class="metric-row"><span class="metric-label">${key}:</span> <span class="metric-value">${value}</span></div>`;
      })
      .replace(/(\$[\d,]+[\w\s]*)/g, '<span class="currency">$1</span>')
      .replace(/(\d+\.\d+%)/g, '<span class="percentage">$1</span>')
      .replace(/(N\/A)/g, '<span class="na-value">N/A</span>')
      .replace(/\n/g, '<br>')
      .replace(/(<br>){3,}/g, '<br><br>');
  };

  const exportResults = () => {
    const exportData = results.map(r => ({
      type: r.analysis_type,
      ticker: r.ticker,
      timestamp: r.timestamp,
      result: r.result
    }));
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis-${results[0]?.ticker}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="analysis-results glass-card">
      <div className="results-header">
        <h3 className="results-title">Analysis Results</h3>
        <button className="btn btn-secondary" onClick={exportResults}>
          <span className="btn-icon">💾</span>
          Export Results
        </button>
      </div>

      <div className="results-tabs">
        {results.map((result, index) => (
          <button
            key={index}
            className={`result-tab ${expandedIndex === index ? 'active' : ''}`}
            onClick={() => setExpandedIndex(index)}
          >
            <span className="tab-icon">
              {result.analysis_type === 'technical' && '📈'}
              {result.analysis_type === 'fundamental' && '💰'}
              {result.analysis_type === 'news' && '📰'}
            </span>
            <span className="tab-text">
              {result.analysis_type.charAt(0).toUpperCase() + result.analysis_type.slice(1)}
            </span>
          </button>
        ))}
      </div>

      <div className="results-content">
        {results.map((result, index) => (
          <div
            key={index}
            className={`result-panel ${expandedIndex === index ? 'active' : ''}`}
          >
            <div className="analysis-meta">
              <div className="meta-item">
                <span className="meta-label">Symbol:</span>
                <span className="meta-value">{result.ticker}</span>
              </div>
              <div className="meta-item">
                <span className="meta-label">Type:</span>
                <span className="meta-value">{result.analysis_type}</span>
              </div>
              <div className="meta-item">
                <span className="meta-label">Time:</span>
                <span className="meta-value">
                  {new Date(result.timestamp).toLocaleString()}
                </span>
              </div>
            </div>
            <SafeHTML
              className="formatted-analysis"
              html={formatAnalysisOutput(result.result)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export default ResultsDisplay;
