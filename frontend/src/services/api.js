import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analysisAPI = {
  market: async (ticker) => {
    const response = await api.post('/api/analysis/market', { ticker });
    return response.data;
  },

  fundamentals: async (ticker) => {
    const response = await api.post('/api/analysis/fundamentals', { ticker });
    return response.data;
  },

  news: async (ticker) => {
    const response = await api.post('/api/analysis/news', { ticker });
    return response.data;
  },
  
  sentiment: async (ticker) => {
    const response = await api.post('/api/analysis/sentiment', { ticker });
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

  /**
   * Stream a chat message via SSE (POST /api/chat).
   *
   * @param {string} message
   * @param {string} threadId
   * @param {object} callbacks
   * @param {(token: string) => void}       callbacks.onToken    — fired per LLM token
   * @param {(nodes: string[]) => void}     callbacks.onProgress — fired when active agents change
   * @param {(response: string) => void}    callbacks.onDone     — fired when the full response is ready
   * @param {(error: string) => void}       callbacks.onError    — fired on error
   * @returns {() => void} abort — call this to cancel the stream
   */
  sendMessageStream: (message, threadId, { onToken, onProgress, onDone, onError } = {}) => {
    const controller = new AbortController();

    (async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message,
            thread_id: threadId || getOrCreateThreadId(),
          }),
          signal: controller.signal,
        });

        if (!res.ok) {
          const text = await res.text();
          onError?.(text || `HTTP ${res.status}`);
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // keep incomplete line in buffer

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            try {
              const payload = JSON.parse(line.slice(6));
              switch (payload.type) {
                case 'token':
                  onToken?.(payload.content);
                  break;
                case 'progress':
                  onProgress?.(payload.nodes);
                  break;
                case 'done':
                  onDone?.(payload.response);
                  break;
                case 'error':
                  onError?.(payload.message);
                  break;
              }
            } catch { /* ignore malformed lines */ }
          }
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          onError?.(err.message || 'Stream failed');
        }
      }
    })();

    return () => controller.abort();
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
