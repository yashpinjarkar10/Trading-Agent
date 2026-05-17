/**
 * Events API service.
 * Calls the backend /api/events endpoints.
 *
 * Contract mirrors the design in EVENT_MAP_FEATURE.md §7:
 *   GET /api/events?since&until&categories&min_severity&min_market_impact&limit&cursor
 *   GET /api/events/:id
 *   GET /api/events/stats
 */

import api from './api';

// Build query string from filters object
function buildQueryParams(filters = {}) {
  const params = new URLSearchParams();
  
  if (filters.since) params.set('since', filters.since);
  if (filters.until) params.set('until', filters.until);
  if (filters.categories && filters.categories.length) {
    filters.categories.forEach(cat => params.append('categories', cat));
  }
  if (filters.min_severity > 0) params.set('min_severity', filters.min_severity);
  if (filters.min_market_impact > 0) params.set('min_market_impact', filters.min_market_impact);
  if (filters.limit) params.set('limit', filters.limit);
  if (filters.cursor) params.set('cursor', filters.cursor);
  
  return params.toString();
}

export const eventsAPI = {
  async list(filters = {}) {
    const query = buildQueryParams(filters);
    const response = await api.get(`/api/events?${query}`);
    return response.data;
  },

  async get(id) {
    const response = await api.get(`/api/events/${id}`);
    return response.data;
  },

  async stats() {
    const response = await api.get('/api/events/stats');
    return response.data;
  },

  async ping() {
    const response = await api.get('/api/events/_ping');
    return response.data;
  },
};

// Custom event dispatched when a ticker chip is clicked in the detail panel.
// The Dashboard / ChartView listens for this and switches the active symbol.
export const OPEN_TICKER_EVENT = 'tradingagent:open-ticker';
export function emitOpenTicker(ticker) {
  window.dispatchEvent(new CustomEvent(OPEN_TICKER_EVENT, { detail: { ticker } }));
}
