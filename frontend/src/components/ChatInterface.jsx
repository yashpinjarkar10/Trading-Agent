import { useEffect, useRef, useState } from 'react';
import { Bot, Send, Sparkles, User, RotateCcw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { chatAPI } from '../services/api';
import SafeHTML from './SafeHTML';
import Button from './ui/Button';
import { Card, CardBody, CardHeader, CardTitle } from './ui/Card';
import HeroScene from './three/HeroScene';

const QUICK_PROMPTS = [
  'Analyze AAPL comprehensively',
  'Compare TSLA vs RIVN',
  'Best dividend stocks under $50?',
  'Explain RSI divergence',
];

function formatChat(content) {
  return content
    .replace(/^##\s*(.+)$/gm, '<h2 class="chat-section-header">$1</h2>')
    .replace(/^###\s*(.+)$/gm, '<h3 class="chat-subsection-header">$1</h3>')
    .replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^\s*\*\s*\*\*([^*]+?)\*\*:\s*(.+)$/gm, '<div class="chat-metric-row"><span class="chat-metric-label"><strong>$1</strong>:</span><span class="chat-metric-value">$2</span></div>')
    .replace(/^\s*\*\s*(.+)$/gm, '<div class="chat-bullet-point">• $1</div>')
    .replace(/(?<!\*)\*([^*\n]+?)\*(?!\*)/g, '<em>$1</em>')
    .replace(/^(\d+\.\s*)(.+)$/gm, '<div class="chat-numbered-point">$1$2</div>')
    .replace(/(\$[\d,]+[\w\s]*)/g, '<span class="currency">$1</span>')
    .replace(/(\d+\.\d+%)/g, '<span class="percentage">$1</span>')
    .replace(/(\d+\/10)/g, '<span class="score-display">$1</span>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>')
    .replace(/(<br>){3,}/g, '<br><br>');
}

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [threadId, setThreadId] = useState(() => chatAPI.getThreadId());
  const endRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, loading]);

  const send = async (text) => {
    const q = (text ?? input).trim();
    if (!q || loading) return;
    setInput('');
    setMessages((m) => [...m, { role: 'user', content: q }]);
    setLoading(true);
    try {
      const res = await chatAPI.sendMessage(q, threadId);
      setMessages((m) => [...m, { role: 'assistant', content: res.response }]);
    } catch (e) {
      setMessages((m) => [...m, { role: 'assistant', content: `**Error:** ${e.response?.data?.detail || e.message || 'Request failed'}` }]);
    } finally {
      setLoading(false);
    }
  };

  const resetThread = () => {
    setMessages([]);
    setThreadId(chatAPI.resetThread());
  };

  return (
    <Card className="flex flex-col h-full overflow-hidden">
      <CardHeader>
        <CardTitle icon={Bot}>AI Trading Assistant</CardTitle>
        <Button variant="ghost" size="sm" onClick={resetThread}>
          <RotateCcw className="w-3.5 h-3.5" /> New chat
        </Button>
      </CardHeader>

      <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4 space-y-4">
        {messages.length === 0 ? <EmptyState onPick={send} /> : (
          <AnimatePresence initial={false}>
            {messages.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2 }}
                className={`flex gap-3 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}
              >
                <div className={`shrink-0 w-8 h-8 rounded-lg flex items-center justify-center border ${
                  m.role === 'user'
                    ? 'bg-accent-violet/10 border-accent-violet/30 text-accent-violet'
                    : 'bg-accent-cyan/10 border-accent-cyan/30 text-accent-cyan'
                }`}>
                  {m.role === 'user' ? <User className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />}
                </div>
                <div className={`flex-1 max-w-[85%] ${m.role === 'user' ? 'text-right' : ''}`}>
                  <div className={`inline-block text-left rounded-xl px-4 py-3 text-sm leading-relaxed border ${
                    m.role === 'user'
                      ? 'bg-accent-violet/10 border-accent-violet/30 text-text-primary'
                      : 'bg-bg-subtle border-border text-text-secondary'
                  }`}>
                    {m.role === 'assistant'
                      ? <SafeHTML className="formatted-chat-content" html={formatChat(m.content)} />
                      : <p className="whitespace-pre-wrap">{m.content}</p>}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}

        {loading && (
          <div className="flex gap-3">
            <div className="shrink-0 w-8 h-8 rounded-lg flex items-center justify-center border bg-accent-cyan/10 border-accent-cyan/30 text-accent-cyan">
              <Sparkles className="w-4 h-4 animate-pulse" />
            </div>
            <div className="flex items-center gap-1.5 px-4 py-3 rounded-xl bg-bg-subtle border border-border">
              <span className="w-1.5 h-1.5 rounded-full bg-text-muted animate-pulse" style={{ animationDelay: '0ms' }} />
              <span className="w-1.5 h-1.5 rounded-full bg-text-muted animate-pulse" style={{ animationDelay: '150ms' }} />
              <span className="w-1.5 h-1.5 rounded-full bg-text-muted animate-pulse" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form
        onSubmit={(e) => { e.preventDefault(); send(); }}
        className="border-t border-border p-3 flex gap-2 bg-bg-elevated/30"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about stocks, strategies, or analysis…"
          disabled={loading}
          className="flex-1 h-11 rounded-lg bg-bg-subtle border border-border text-sm text-text-primary placeholder:text-text-muted px-4 focus-ring focus:border-accent-cyan/50 transition-colors"
        />
        <Button type="submit" disabled={loading || !input.trim()} size="md">
          <Send className="w-4 h-4" />
          <span className="hidden sm:inline">Send</span>
        </Button>
      </form>
    </Card>
  );
}

function EmptyState({ onPick }) {
  return (
    <div className="relative h-full min-h-[360px] flex flex-col items-center justify-center text-center px-6">
      <div className="absolute inset-0 opacity-50 pointer-events-none">
        <HeroScene />
      </div>
      <div className="relative z-10 max-w-md">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-br from-accent-cyan to-accent-violet shadow-glow mb-4">
          <Sparkles className="w-6 h-6 text-bg" strokeWidth={2.5} />
        </div>
        <h3 className="text-xl font-semibold gradient-text mb-2">
          Your AI trading copilot
        </h3>
        <p className="text-sm text-text-secondary mb-6">
          Ask anything about markets, run analyses, compare stocks, or get strategy ideas.
        </p>
        <div className="flex flex-wrap gap-2 justify-center">
          {QUICK_PROMPTS.map((p) => (
            <button
              key={p}
              onClick={() => onPick(p)}
              className="chip hover:border-accent-cyan/40 hover:text-text-primary transition-colors cursor-pointer"
            >
              {p}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
