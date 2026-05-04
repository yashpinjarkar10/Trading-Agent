import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analysisAPI = {
  technical: async (ticker, period = '1y') => {
    const response = await api.post('/api/analysis/technical', {
      ticker,
      period,
    });
    return response.data;
  },

  fundamental: async (ticker) => {
    const response = await api.post('/api/analysis/fundamental', {
      ticker,
    });
    return response.data;
  },

  news: async (ticker, daysBack = 7, maxArticles = 50) => {
    const response = await api.post('/api/analysis/news', {
      ticker,
      days_back: daysBack,
      max_articles: maxArticles,
    });
    return response.data;
  },
};

// Bug #9: Per-browser-session thread id; persisted in localStorage so a user's
// chat history follows them across reloads (until auth is added, after which
// thread_id should be derived from user_id + session_id).
const THREAD_ID_KEY = 'trading_agent_thread_id';
function getOrCreateThreadId() {
  let tid = localStorage.getItem(THREAD_ID_KEY);
  if (!tid) {
    const rand =
      (crypto && crypto.randomUUID && crypto.randomUUID()) ||
      `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    tid = `web-${rand}`;
    localStorage.setItem(THREAD_ID_KEY, tid);
  }
  return tid;
}

export const chatAPI = {
  getThreadId: getOrCreateThreadId,
  resetThread: () => {
    localStorage.removeItem(THREAD_ID_KEY);
    return getOrCreateThreadId();
  },
  sendMessage: async (message, threadId) => {
    const response = await api.post('/api/chat', {
      message,
      thread_id: threadId || getOrCreateThreadId(),
    });
    return response.data;
  },
};

export const healthAPI = {
  check: async () => {
    const response = await api.get('/api/health');
    return response.data;
  },

  getTickers: async () => {
    const response = await api.get('/api/tickers');
    return response.data;
  },
};

export default api;
