/**
 * api.js — All calls to the MarketLens FastAPI backend.
 *
 * Base URL defaults to http://localhost:8000.
 * Override via VITE_API_URL env variable (e.g. in a .env file).
 */

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Run Modules 1–3 for a ticker and date range.
 * Returns: { ticker, start_date, end_date, total_return,
 *             prices[], events[], anomalies[],
 *             predicted_price, sentiment_score, sentiment_label }
 */
export async function fetchAnalysis(ticker, start, end, threshold = 0.5) {
  const params = new URLSearchParams({
    start,
    end,
    threshold: String(threshold),
  })
  const res = await fetch(
    `${BASE}/api/analyze/${encodeURIComponent(ticker)}?${params}`
  )
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Analysis failed (HTTP ${res.status})`)
  }
  return res.json()
}

/**
 * Run the full pipeline including Module 4 (Claude report).
 * Returns: { ticker, report }
 *
 * Requires ANTHROPIC_API_KEY set in the backend environment.
 * This call is intentionally separate — it makes a live Claude API call
 * and may take several seconds.
 */
export async function fetchReport(ticker, start, end, threshold = 0.5) {
  const params = new URLSearchParams({
    start,
    end,
    threshold: String(threshold),
  })
  const res = await fetch(
    `${BASE}/api/report/${encodeURIComponent(ticker)}?${params}`
  )
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Report generation failed (HTTP ${res.status})`)
  }
  return res.json()
}
