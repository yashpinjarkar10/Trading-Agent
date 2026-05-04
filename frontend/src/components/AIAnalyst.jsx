import { useState } from 'react';
import AnalysisForm from './AnalysisForm';
import ChatInterface from './ChatInterface';

function AIAnalyst({ initialSymbol }) {
  const [currentMode, setCurrentMode] = useState('direct');

  return (
    <div className="analyst-section">
      <div className="section-header">
        <h2 className="section-title">
          <span className="section-icon">🤖</span>
          AI Trading Analyst
        </h2>
        <div className="mode-selector">
          <button
            className={`mode-btn ${currentMode === 'direct' ? 'active' : ''}`}
            onClick={() => setCurrentMode('direct')}
          >
            <span className="mode-icon">📊</span>
            Direct Analysis
          </button>
          <button
            className={`mode-btn ${currentMode === 'chat' ? 'active' : ''}`}
            onClick={() => setCurrentMode('chat')}
          >
            <span className="mode-icon">💬</span>
            AI Chat
          </button>
        </div>
      </div>

      <div className={`mode-content ${currentMode === 'direct' ? 'active' : ''}`}>
        <AnalysisForm initialSymbol={initialSymbol} />
      </div>

      <div className={`mode-content ${currentMode === 'chat' ? 'active' : ''}`}>
        <ChatInterface />
      </div>
    </div>
  );
}

export default AIAnalyst;
