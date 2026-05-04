import { useState } from 'react';
import ChartView from './components/ChartView';
import AIAnalyst from './components/AIAnalyst';
import './styles/index.css';

function App() {
  const [currentSection, setCurrentSection] = useState('chart');
  const [currentSymbol, setCurrentSymbol] = useState('AAPL');

  return (
    <>
      <div className="background">
        <div className="bg-gradient-1"></div>
        <div className="bg-gradient-2"></div>
      </div>

      <div className="container">
        <header className="header">
          <div className="header-content">
            <div className="logo">
              <span className="logo-icon">📈</span>
              <h1 className="logo-text">Trading Agent</h1>
            </div>
            <nav className="nav">
              <button
                className={`nav-link ${currentSection === 'chart' ? 'active' : ''}`}
                onClick={() => setCurrentSection('chart')}
              >
                <span className="nav-icon">📊</span>
                Chart View
              </button>
              <button
                className={`nav-link ${currentSection === 'analyst' ? 'active' : ''}`}
                onClick={() => setCurrentSection('analyst')}
              >
                <span className="nav-icon">🤖</span>
                AI Analyst
              </button>
            </nav>
          </div>
        </header>

        <main className="main-content">
          <div className={`content-section ${currentSection === 'chart' ? 'active' : ''}`}>
            <ChartView 
              currentSymbol={currentSymbol} 
              setCurrentSymbol={setCurrentSymbol}
              switchToAnalyst={() => setCurrentSection('analyst')}
            />
          </div>

          <div className={`content-section ${currentSection === 'analyst' ? 'active' : ''}`}>
            <AIAnalyst initialSymbol={currentSymbol} />
          </div>
        </main>

        <footer className="footer">
          <p>Trading Agent v1.0 - AI-Powered Stock Analysis Platform</p>
          <p className="footer-disclaimer">
            Disclaimer: This tool is for informational purposes only. Not financial advice.
          </p>
        </footer>
      </div>
    </>
  );
}

export default App;
