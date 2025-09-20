// Trading Agent Web Interface - Clean Version

class TradingAgentApp {
    constructor() {
        this.currentSection = 'chart';
        this.currentMode = 'direct';
        this.tradingViewWidget = null;
        this.isLoading = false;
        this.lastAnalysisData = null;
        
        this.init();
    }

    init() {
        console.log('🚀 Initializing Trading Agent...');
        this.hideLoading();
        this.setupEventListeners();
        this.initializeTradingView();
        this.setupFormHandlers();
        this.setupChatInterface();
        this.showNotification('Trading Agent initialized successfully!', 'success');
    }

    hideLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
        this.isLoading = false;
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = link.dataset.section;
                this.switchSection(section);
            });
        });

        // Mode switching
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = btn.dataset.mode;
                this.switchMode(mode);
            });
        });

        // Quick analysis buttons
        document.querySelectorAll('.analysis-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const type = btn.dataset.type;
                this.quickAnalysis(type);
            });
        });

        // Symbol update
        const symbolUpdateBtn = document.getElementById('symbol-update-btn');
        const symbolInput = document.getElementById('symbol-input');
        
        if (symbolUpdateBtn && symbolInput) {
            symbolUpdateBtn.addEventListener('click', () => {
                this.updateChartSymbol(symbolInput.value);
            });
            
            symbolInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.updateChartSymbol(symbolInput.value);
                }
            });
        }

        // Analysis type change handlers for checkboxes
        const checkboxes = document.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateAnalysisOptions();
            });
        });
    }

    switchSection(sectionName) {
        // Update navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-section="${sectionName}"]`).classList.add('active');

        // Update content sections
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(`${sectionName}-section`).classList.add('active');

        this.currentSection = sectionName;

        // Re-initialize TradingView if switching to chart
        if (sectionName === 'chart') {
            setTimeout(() => this.initializeTradingView(), 100);
        }
    }

    switchMode(modeName) {
        // Update mode buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-mode="${modeName}"]`).classList.add('active');

        // Update mode content
        document.querySelectorAll('.analyst-mode').forEach(mode => {
            mode.classList.remove('active');
        });
        document.getElementById(`${modeName}-mode`).classList.add('active');

        this.currentMode = modeName;
    }

    initializeTradingView() {
        const container = document.getElementById('tradingview_chart');
        if (!container) {
            console.log('TradingView container not found');
            return;
        }

        if (typeof TradingView === 'undefined') {
            console.log('TradingView not loaded, retrying...');
            setTimeout(() => this.initializeTradingView(), 1000);
            return;
        }

        // Clear existing widget
        container.innerHTML = '';

        try {
            this.tradingViewWidget = new TradingView.widget({
                "symbol": "NASDAQ:TSLA",
                "interval": "D",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#161b22",
                "enable_publishing": false,
                "allow_symbol_change": true,
                "container_id": "tradingview_chart",
                "height": 500,
                "width": "100%",
                "autosize": true,
                "studies": [],
                "hide_top_toolbar": false,
                "hide_legend": false,
                "save_image": false,
                "backgroundColor": "#161b22",
                "gridColor": "#21262d",
                "overrides": {
                    "paneProperties.background": "#161b22",
                    "paneProperties.backgroundType": "solid",
                    "mainSeriesProperties.candleStyle.upColor": "#10b981",
                    "mainSeriesProperties.candleStyle.downColor": "#ef4444",
                    "mainSeriesProperties.candleStyle.borderUpColor": "#10b981",
                    "mainSeriesProperties.candleStyle.borderDownColor": "#ef4444"
                }
            });
            
            console.log('✅ TradingView widget initialized');
        } catch (error) {
            console.error('❌ TradingView error:', error);
            this.showNotification('Failed to load chart', 'error');
        }
    }

    updateChartSymbol(symbol) {
        const formattedSymbol = symbol.toUpperCase();
        document.getElementById('current-symbol').textContent = formattedSymbol;
        
        if (this.tradingViewWidget && this.tradingViewWidget.chart) {
            try {
                this.tradingViewWidget.chart().setSymbol(`NASDAQ:${formattedSymbol}`);
                this.showNotification(`Chart updated to ${formattedSymbol}`, 'success');
            } catch (error) {
                console.error('Chart update error:', error);
                this.initializeTradingView();
            }
        }
    }

    setupFormHandlers() {
        const analysisForm = document.getElementById('analysis-form');
        if (analysisForm) {
            analysisForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.performDirectAnalysis();
            });
        }
    }

    updateAnalysisOptions() {
        const technicalCheck = document.getElementById('technical-check');
        const newsCheck = document.getElementById('news-check');
        const technicalOptions = document.getElementById('technical-options');
        const newsOptions = document.getElementById('news-options');
        
        // Show technical options if technical analysis is selected
        if (technicalOptions) {
            technicalOptions.style.display = technicalCheck && technicalCheck.checked ? 'block' : 'none';
        }
        
        // Show news options if news analysis is selected
        if (newsOptions) {
            newsOptions.style.display = newsCheck && newsCheck.checked ? 'block' : 'none';
        }
    }

    async performDirectAnalysis() {
        const ticker = document.getElementById('analysis-ticker').value.trim().toUpperCase();
        
        // Get selected analysis types
        const selectedTypes = [];
        const technicalCheck = document.getElementById('technical-check');
        const fundamentalCheck = document.getElementById('fundamental-check');
        const newsCheck = document.getElementById('news-check');
        
        if (technicalCheck && technicalCheck.checked) selectedTypes.push('technical');
        if (fundamentalCheck && fundamentalCheck.checked) selectedTypes.push('fundamental');
        if (newsCheck && newsCheck.checked) selectedTypes.push('news');

        if (!ticker) {
            this.showNotification('Please enter a ticker symbol', 'warning');
            return;
        }
        
        if (selectedTypes.length === 0) {
            this.showNotification('Please select at least one analysis type', 'warning');
            return;
        }

        this.setLoading(true);
        const results = [];
        
        try {
            // Run all selected analyses
            for (const analysisType of selectedTypes) {
                console.log(`🔍 Running ${analysisType} analysis for ${ticker}`);
                
                const requestData = {
                    ticker: ticker,
                    period: document.getElementById('period').value || '1y',
                    days_back: parseInt(document.getElementById('days-back').value) || 7,
                    max_articles: 50
                };

                const response = await fetch(`/api/analysis/${analysisType}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestData)
                });

                if (!response.ok) {
                    throw new Error(`${analysisType} analysis failed: HTTP ${response.status}`);
                }

                const data = await response.json();
                results.push(data);
            }
            
            // Display all results
            this.displayMultipleAnalysisResults(results);
            this.showNotification(`Completed ${selectedTypes.length} analysis types for ${ticker}!`, 'success');

        } catch (error) {
            console.error('❌ Analysis error:', error);
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.setLoading(false);
        }
    }

    displayAnalysisResult(data) {
        const resultContainer = document.getElementById('analysis-output');
        const resultsDiv = document.querySelector('.analysis-results');
        const exportBtn = document.getElementById('export-btn');
        
        if (!resultContainer) return;

        resultsDiv.style.display = 'block';
        if (exportBtn) exportBtn.style.display = 'inline-flex';
        
        // Format the analysis result for better display
        const formattedResult = this.formatAnalysisOutput(data.result, data.analysis_type);
        
        resultContainer.innerHTML = `
            <div class="analysis-meta">
                <div class="meta-item">
                    <span class="meta-label">Symbol:</span>
                    <span class="meta-value">${data.ticker}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Type:</span>
                    <span class="meta-value">${data.analysis_type.charAt(0).toUpperCase() + data.analysis_type.slice(1)} Analysis</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Generated:</span>
                    <span class="meta-value">${new Date(data.timestamp).toLocaleString()}</span>
                </div>
            </div>
            <div class="analysis-content">
                ${formattedResult}
            </div>
        `;
        
        this.lastAnalysisData = data;
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    displayMultipleAnalysisResults(results) {
        const resultContainer = document.getElementById('analysis-output');
        const resultsDiv = document.querySelector('.analysis-results');
        const exportBtn = document.getElementById('export-btn');
        
        if (!resultContainer || results.length === 0) return;

        resultsDiv.style.display = 'block';
        if (exportBtn) exportBtn.style.display = 'inline-flex';
        
        // Create a combined report with all analysis types
        const ticker = results[0].ticker;
        const timestamp = new Date().toLocaleString();
        
        let combinedHtml = `
            <div class="analysis-meta">
                <div class="meta-item">
                    <span class="meta-label">Symbol:</span>
                    <span class="meta-value">${ticker}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Analysis Types:</span>
                    <span class="meta-value">${results.map(r => r.analysis_type.charAt(0).toUpperCase() + r.analysis_type.slice(1)).join(', ')}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Generated:</span>
                    <span class="meta-value">${timestamp}</span>
                </div>
            </div>
            <div class="multi-analysis-container">
        `;
        
        // Add each analysis result as a separate section
        results.forEach((data, index) => {
            const analysisTypeTitle = data.analysis_type.charAt(0).toUpperCase() + data.analysis_type.slice(1);
            const formattedResult = this.formatAnalysisOutput(data.result, data.analysis_type);
            
            combinedHtml += `
                <div class="analysis-section" data-analysis="${data.analysis_type}">
                    <div class="analysis-section-header">
                        <h2 class="analysis-section-title">
                            <i class="fas fa-${this.getAnalysisIcon(data.analysis_type)}"></i>
                            ${analysisTypeTitle} Analysis
                        </h2>
                        <div class="analysis-section-meta">
                            <span class="analysis-timestamp">${new Date(data.timestamp).toLocaleString()}</span>
                        </div>
                    </div>
                    <div class="analysis-section-content">
                        ${formattedResult}
                    </div>
                </div>
            `;
        });
        
        combinedHtml += '</div>';
        
        resultContainer.innerHTML = combinedHtml;
        
        // Store combined data for export
        this.lastAnalysisData = {
            ticker: ticker,
            analysis_type: 'multi',
            timestamp: new Date().toISOString(),
            result: results.map(r => `${'='.repeat(80)}\\n${r.analysis_type.toUpperCase()} ANALYSIS\\n${'='.repeat(80)}\\n${r.result}`).join('\\n\\n'),
            individual_results: results
        };
        
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    getAnalysisIcon(analysisType) {
        const icons = {
            'technical': 'chart-bar',
            'fundamental': 'building',
            'news': 'newspaper'
        };
        return icons[analysisType] || 'chart-line';
    }

    formatAnalysisOutput(result, analysisType) {
        // Convert the plain text analysis into formatted HTML
        let formatted = result
            // Handle section headers (lines with = symbols)
            .replace(/^={40,}$/gm, '<div class="section-divider"></div>')
            .replace(/^={10,}\s*$/gm, '<div class="section-divider"></div>')
            
            // Handle main title (ANALYSIS REPORT: TICKER)
            .replace(/^([A-Z\s]+ANALYSIS\s+REPORT:\s+[A-Z]+)$/gm, '<h1 class="analysis-title">$1</h1>')
            
            // Handle "Generated on:" line
            .replace(/^(Generated on:\s*)(.+)$/gm, '<div class="generated-info">$1<span class="timestamp">$2</span></div>')
            
            // Handle section headers with emojis and special symbols
            .replace(/^([📊🔍💰📈👥⭐📝⚠️🎯]\s*[A-Z\s&]+)$/gm, '<h2 class="section-header">$1</h2>')
            .replace(/^([A-Z][A-Z\s]+)$/gm, (match) => {
                // Only convert to header if it's all caps and looks like a section
                if (match.length > 3 && match.match(/^[A-Z\s]+$/)) {
                    return `<h2 class="section-header">${match}</h2>`;
                }
                return match;
            })
            
            // Handle subsection headers (ending with colon)
            .replace(/^([A-Za-z\s]+Metrics?:)$/gm, '<h3 class="subsection-header">$1</h3>')
            .replace(/^([A-Za-z\s]+Information:)$/gm, '<h3 class="subsection-header">$1</h3>')
            .replace(/^([A-Za-z\s]+Analysis:)$/gm, '<h3 class="subsection-header">$1</h3>')
            .replace(/^([A-Za-z\s]+Factors?:)$/gm, '<h3 class="subsection-header">$1</h3>')
            .replace(/^([A-Za-z\s]+Recommendation[s]?:)$/gm, '<h3 class="subsection-header">$1</h3>')
            
            // Handle dashed separators
            .replace(/^-{20,}$/gm, '<div class="subsection-divider"></div>')
            
            // Handle bullet points and checkmarks
            .replace(/^\s*•\s*(.+)$/gm, '<div class="bullet-point">• $1</div>')
            .replace(/^\s*✓\s*(.+)$/gm, '<div class="success-point">✓ $1</div>')
            
            // Handle numbered items in recommendations
            .replace(/^(\d+\.\s*)(.+)$/gm, '<div class="numbered-point">$1$2</div>')
            
            // Handle key-value pairs for metrics (but preserve already formatted headers)
            .replace(/^(\s*)([A-Za-z\s\(\)]+):\s*(.+)$/gm, (match, indent, key, value) => {
                // Skip if it's already a header or contains emojis
                if (key.match(/^[A-Z\s]+$/) || match.includes('📊') || match.includes('💰') || 
                    key.includes('Metrics') || key.includes('Information') || key.includes('Analysis') ||
                    key.includes('Factors') || key.includes('Recommendation') || key.includes('Generated')) {
                    return match;
                }
                return `${indent}<div class="metric-row"><span class="metric-label">${key}:</span> <span class="metric-value">${value}</span></div>`;
            })
            
            // Handle special formatting for different value types
            .replace(/(\$[\d,]+[\w\s]*)/g, '<span class="currency">$1</span>')
            .replace(/(\d+\.\d+%)/g, '<span class="percentage">$1</span>')
            .replace(/(N\/A)/g, '<span class="na-value">N/A</span>')
            
            // Handle recommendation text
            .replace(/(Recommendation:\s*)([\w\/\s]+)/g, '$1<span class="recommendation">$2</span>')
            .replace(/(Explanation:\s*)(.+)/g, '$1<span class="explanation">$2</span>')
            
            // Handle disclaimer section
            .replace(/(Disclaimer:[\s\S]*?)(?=<|$)/g, '<div class="disclaimer">$1</div>')
            
            // Handle newlines
            .replace(/\n/g, '<br>')
            
            // Clean up multiple consecutive <br> tags
            .replace(/(<br>){3,}/g, '<br><br>');

        return `<div class="formatted-analysis">${formatted}</div>`;
    }

    quickAnalysis(type) {
        const currentSymbol = document.getElementById('current-symbol').textContent || 'AAPL';
        
        this.switchSection('analyst');
        this.switchMode('direct');
        
        // Clear all checkboxes first
        document.getElementById('technical-check').checked = false;
        document.getElementById('fundamental-check').checked = false;
        document.getElementById('news-check').checked = false;
        
        // Check the selected type
        const checkbox = document.getElementById(`${type}-check`);
        if (checkbox) {
            checkbox.checked = true;
        }
        
        // Update form
        document.getElementById('analysis-ticker').value = currentSymbol;
        this.updateAnalysisOptions();
        
        setTimeout(() => this.performDirectAnalysis(), 100);
    }

    setupChatInterface() {
        const chatForm = document.getElementById('chat-form');
        const chatInput = document.getElementById('chat-input');
        
        if (chatForm && chatInput) {
            chatForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.sendChatMessage();
            });
        }

        // Chat suggestions
        document.querySelectorAll('.suggestion-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const text = btn.dataset.text;
                if (chatInput) {
                    chatInput.value = text;
                    this.sendChatMessage();
                }
            });
        });
    }

    async sendChatMessage() {
        const chatInput = document.getElementById('chat-input');
        const message = chatInput.value.trim();
        
        if (!message) return;
        
        chatInput.value = '';
        this.addChatMessage(message, 'user');
        this.showTypingIndicator();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    thread_id: 'web_session_' + Date.now()
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            this.removeTypingIndicator();
            this.addChatMessage(data.response, 'ai');
            
        } catch (error) {
            console.error('❌ Chat error:', error);
            this.removeTypingIndicator();
            this.addChatMessage('Sorry, I encountered an error. Please try again.', 'ai');
            this.showNotification('Chat service unavailable', 'error');
        }
    }

    addChatMessage(content, sender) {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const timeString = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-${sender === 'user' ? 'user' : 'robot'}"></i>
            </div>
            <div class="message-content">
                <p>${content.replace(/\n/g, '<br>')}</p>
                <div class="message-time">${timeString}</div>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    showTypingIndicator() {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) return;
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai-message typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    removeTypingIndicator() {
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    setLoading(loading) {
        this.isLoading = loading;
        const loadingOverlay = document.getElementById('loading-overlay');
        const analyzeBtn = document.querySelector('.analyze-btn');
        
        if (loadingOverlay) {
            loadingOverlay.style.display = loading ? 'flex' : 'none';
        }
        
        if (analyzeBtn) {
            analyzeBtn.disabled = loading;
            const btnText = analyzeBtn.querySelector('span');
            const btnIcon = analyzeBtn.querySelector('i');
            
            if (loading) {
                if (btnText) btnText.textContent = 'Analyzing...';
                if (btnIcon) btnIcon.className = 'fas fa-spinner fa-spin';
            } else {
                if (btnText) btnText.textContent = 'Run Analysis';
                if (btnIcon) btnIcon.className = 'fas fa-search';
            }
        }
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 12px;">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
        
        console.log(`${type.toUpperCase()}: ${message}`);
    }

    getNotificationIcon(type) {
        const icons = {
            'success': 'check-circle',
            'error': 'exclamation-circle',
            'warning': 'exclamation-triangle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    exportAnalysis() {
        if (!this.lastAnalysisData) {
            this.showNotification('No analysis data to export', 'warning');
            return;
        }
        
        const data = this.lastAnalysisData;
        const filename = `${data.ticker}_${data.analysis_type}_${new Date().toISOString().split('T')[0]}.txt`;
        
        const blob = new Blob([data.result], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        window.URL.revokeObjectURL(url);
        this.showNotification(`Exported: ${filename}`, 'success');
    }
}

// Additional CSS for typing indicator and other dynamic elements
const additionalCSS = `
.typing-indicator .message-content {
    padding: 12px 16px;
}

.typing-dots {
    display: flex;
    gap: 4px;
    align-items: center;
}

.typing-dots span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-primary);
    animation: typingDots 1.4s infinite ease-in-out;
}

.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes typingDots {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}

.analysis-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 24px;
    padding: 16px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.meta-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.meta-label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.meta-value {
    font-weight: 600;
    color: var(--text-primary);
}

.analysis-content pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: var(--font-family);
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-primary);
}
`;

// Inject additional CSS
const style = document.createElement('style');
style.textContent = additionalCSS;
document.head.appendChild(style);

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Hide loading overlay immediately
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
    
    window.tradingApp = new TradingAgentApp();
    
    // Setup export button
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            window.tradingApp.exportAnalysis();
        });
    }
    
    console.log('✅ Trading Agent Web Interface loaded!');
});