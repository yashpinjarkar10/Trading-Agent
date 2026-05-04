import { useState, useRef, useEffect } from 'react';
import { chatAPI } from '../services/api';
import SafeHTML from './SafeHTML';

function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  // Bug #9: real per-browser thread_id (no shared "default")
  const [threadId] = useState(() => chatAPI.getThreadId());
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const formatChatMessage = (content) => {
    return content
      .replace(/^##\s*(.+)$/gm, '<h2 class="chat-section-header">$1</h2>')
      .replace(/^###\s*(.+)$/gm, '<h3 class="chat-subsection-header">$1</h3>')
      .replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>')
      .replace(/^\s*\*\s*\*\*([^*]+?)\*\*:\s*(.+)$/gm, '<div class="chat-metric-row"><span class="chat-metric-label"><strong>$1</strong>:</span> <span class="chat-metric-value">$2</span></div>')
      .replace(/^\s*\*\s*(.+)$/gm, '<div class="chat-bullet-point">• $1</div>')
      .replace(/(?<!\*)\*([^*\n]+?)\*(?!\*)/g, '<em>$1</em>')
      .replace(/^(\d+\.\s*)(.+)$/gm, '<div class="chat-numbered-point">$1$2</div>')
      .replace(/(\$[\d,]+[\w\s]*)/g, '<span class="currency">$1</span>')
      .replace(/(\d+\.\d+%)/g, '<span class="percentage">$1</span>')
      .replace(/(\d+\/10)/g, '<span class="score-display">$1</span>')
      .replace(/\n\n/g, '<br><br>')
      .replace(/\n/g, '<br>')
      .replace(/(<br>){3,}/g, '<br><br>');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await chatAPI.sendMessage(input, threadId);
      const assistantMessage = { role: 'assistant', content: response.response };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${err.response?.data?.detail || err.message || 'Failed to get response'}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const quickPrompts = [
    'Analyze AAPL stock comprehensively',
    'Compare TSLA vs AAPL',
    'What are the best tech stocks right now?',
    'Explain P/E ratio in simple terms',
  ];

  return (
    <div className="chat-interface glass-card">
      <div className="chat-header">
        <h3 className="chat-title">
          <span className="chat-icon">💬</span>
          AI Trading Assistant
        </h3>
        <p className="chat-subtitle">Ask me anything about stocks and markets</p>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-welcome">
            <div className="welcome-icon">🤖</div>
            <h4>Welcome to AI Trading Assistant!</h4>
            <p>Ask me about stock analysis, market trends, or financial concepts.</p>
            <div className="quick-prompts">
              <p className="quick-prompts-label">Try these:</p>
              {quickPrompts.map((prompt, index) => (
                <button
                  key={index}
                  className="quick-prompt-btn"
                  onClick={() => setInput(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={`chat-message ${message.role}`}>
            <div className="message-avatar">
              {message.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="message-content">
              {message.role === 'assistant' ? (
                <SafeHTML
                  className="formatted-chat-content"
                  html={formatChatMessage(message.content)}
                />
              ) : (
                <p>{message.content}</p>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-message assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          className="chat-input"
          placeholder="Ask about stocks, markets, or analysis..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
        />
        <button type="submit" className="chat-send-btn" disabled={loading || !input.trim()}>
          <span className="send-icon">📤</span>
        </button>
      </form>
    </div>
  );
}

export default ChatInterface;
