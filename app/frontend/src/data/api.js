/**
 * api.js — All calls to the MarketLens FastAPI backend.
 * Base URL defaults to http://localhost:8000.
 * Override via VITE_API_URL in a .env file.
 */

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function _get(url) {
  const res = await fetch(url)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (HTTP ${res.status})`)
  }
  return res.json()
}

/**
 * Stage 1 — prices + anomalies + sentiment (fast, uses disk cache in module1).
 * Returns: { ticker, start_date, end_date, total_return,
 *             prices[], spy_prices[], events[], anomalies[],
 *             sentiment_score, sentiment_label, news_available }
 */
export async function fetchAnalysis(ticker, start, end) {
  const params = new URLSearchParams({ start, end })
  return _get(`${BASE}/api/analyze/${encodeURIComponent(ticker)}?${params}`)
}

/**
 * Stage 2 — TransformerForecaster (slow first run, disk-cached thereafter).
 * Returns: { ticker, model_name, day5_price, forecast_5d,
 *             actual[], predicted[], test_dates[],
 *             dir_accuracy, mae, sector_name }
 */
export async function fetchForecast(ticker, start, end) {
  const params = new URLSearchParams({ start, end })
  return _get(`${BASE}/api/forecast/${encodeURIComponent(ticker)}?${params}`)
}

/**
 * Stage 3 — Claude AI report (on-demand, requires ANTHROPIC_API_KEY in backend env).
 * Returns: { ticker, report }
 */
export async function fetchReport(ticker, start, end) {
  const params = new URLSearchParams({ start, end })
  return _get(`${BASE}/api/report/${encodeURIComponent(ticker)}?${params}`)
}

/**
 * Live market metrics from yfinance.
 * Returns: { ticker, pe_ratio, market_cap, week52_position, beta,
 *             analyst_rating, analyst_target, upside,
 *             vix, vix_label, rel_perf_30d }
 */
export async function fetchMarketInfo(ticker) {
  return _get(`${BASE}/api/market-info/${encodeURIComponent(ticker)}`)
}
