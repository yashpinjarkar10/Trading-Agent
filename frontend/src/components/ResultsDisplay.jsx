import { useState } from 'react';
import { Download, LineChart, BarChart3, Newspaper } from 'lucide-react';
import { Card, CardBody, CardHeader, CardTitle } from './ui/Card';
import Button from './ui/Button';
import Badge from './ui/Badge';
import SafeHTML from './SafeHTML';

const ICONS = { technical: LineChart, fundamental: BarChart3, news: Newspaper };

function formatAnalysis(text) {
  if (!text) return '';
  return text
    .replace(/^={40,}$/gm, '<div class="section-divider"></div>')
    .replace(/^([A-Z\s]+ANALYSIS\s+REPORT:\s+[A-Z.\-]+)$/gm, '<h1 class="analysis-title">$1</h1>')
    .replace(/^(Generated on:\s*)(.+)$/gm, '<div class="generated-info">$1<span class="timestamp">$2</span></div>')
    .replace(/^([📊🔍💰📈👥⭐📝⚠️🎯]\s*[A-Z\s&]+)$/gm, '<h2 class="section-header">$1</h2>')
    .replace(/^-{20,}$/gm, '<div class="subsection-divider"></div>')
    .replace(/^\s*•\s*(.+)$/gm, '<div class="bullet-point">• $1</div>')
    .replace(/^\s*✓\s*(.+)$/gm, '<div class="success-point">✓ $1</div>')
    .replace(/^(\d+\.\s*)(.+)$/gm, '<div class="numbered-point">$1$2</div>')
    .replace(/^(\s*)([A-Za-z\s\(\)]+):\s*(.+)$/gm, (match, indent, key, value) => {
      if (key.match(/^[A-Z\s]+$/) || /[📊💰]/.test(match) ||
          /Metrics|Information|Analysis|Factors|Recommendation|Generated/.test(key)) return match;
      return `${indent}<div class="metric-row"><span class="metric-label">${key}:</span><span class="metric-value">${value}</span></div>`;
    })
    .replace(/(\$[\d,]+[\w\s]*)/g, '<span class="currency">$1</span>')
    .replace(/(\d+\.\d+%)/g, '<span class="percentage">$1</span>')
    .replace(/(N\/A)/g, '<span class="na-value">N/A</span>')
    .replace(/\n/g, '<br>')
    .replace(/(<br>){3,}/g, '<br><br>');
}

export default function ResultsDisplay({ results }) {
  const [active, setActive] = useState(0);

  const exportResults = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis-${results[0]?.ticker}-${Date.now()}.json`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Results</CardTitle>
        <Button variant="secondary" size="sm" onClick={exportResults}>
          <Download className="w-3.5 h-3.5" /> Export JSON
        </Button>
      </CardHeader>

      <div className="px-5 pt-4 flex flex-wrap gap-1.5 border-b border-border">
        {results.map((r, i) => {
          const Icon = ICONS[r.analysis_type] || LineChart;
          const isActive = active === i;
          return (
            <button
              key={i}
              onClick={() => setActive(i)}
              className={`inline-flex items-center gap-2 px-3 h-9 -mb-px rounded-t-md text-sm border-b-2 transition-colors ${
                isActive
                  ? 'border-accent-cyan text-text-primary'
                  : 'border-transparent text-text-secondary hover:text-text-primary'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span className="capitalize">{r.analysis_type}</span>
            </button>
          );
        })}
      </div>

      <CardBody>
        {results.map((r, i) => i === active && (
          <div key={i}>
            <div className="analysis-meta">
              <div className="meta-item"><span className="meta-label">Symbol</span><Badge tone="cyan">{r.ticker}</Badge></div>
              <div className="meta-item"><span className="meta-label">Type</span><span className="meta-value capitalize">{r.analysis_type}</span></div>
              <div className="meta-item"><span className="meta-label">Generated</span><span className="meta-value">{new Date(r.timestamp).toLocaleString()}</span></div>
            </div>
            <SafeHTML className="formatted-analysis text-sm leading-relaxed text-text-secondary" html={formatAnalysis(r.result)} />
          </div>
        ))}
      </CardBody>
    </Card>
  );
}
