/**
 * AnomalyPanel — Displays the list of anomalies detected by Module 2.
 *
 * Each card shows:
 *   • Date and % change (green/red)
 *   • Auto-generated comment from FunnelDetector
 *   • Related news events from Module 1 (within ±2 days)
 */

import { TrendingUp, TrendingDown } from 'lucide-react'

const EVENT_COLORS = {
  EARNINGS:   '#10b981',
  ANALYST:    '#3b82f6',
  REGULATORY: '#f59e0b',
  MACRO:      '#ef4444',
  PRODUCT:    '#8b5cf6',
  OTHER:      '#6b7280',
}

function fmtDate(d) {
  return new Date(d + 'T00:00:00').toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export default function AnomalyPanel({ anomalies }) {
  if (!anomalies?.length) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 text-center">
        <p className="text-gray-500 text-sm">No anomalies detected in this range.</p>
        <p className="text-gray-700 text-xs mt-1">Try lowering the Anomaly Threshold.</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {anomalies.map((a) => (
        <div
          key={a.date}
          className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-gray-700 transition-colors"
        >
          {/* Header row */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              {a.is_gain
                ? <TrendingUp size={15} className="text-emerald-400" />
                : <TrendingDown size={15} className="text-red-400" />
              }
              <span
                className={`text-lg font-bold leading-none ${
                  a.is_gain ? 'text-emerald-400' : 'text-red-400'
                }`}
              >
                {a.is_gain ? '+' : ''}{a.percent_change.toFixed(2)}%
              </span>
              <span className="text-gray-600 text-xs">open→close</span>
            </div>
            <span className="text-gray-500 text-xs">{fmtDate(a.date)}</span>
          </div>

          {/* Detector comment */}
          <p className="text-gray-400 text-xs leading-relaxed mb-3">{a.comment}</p>

          {/* Related events */}
          {a.related_events.length > 0 && (
            <div className="border-t border-gray-800 pt-3 space-y-2">
              <p className="text-gray-600 text-xs font-semibold uppercase tracking-wide">
                Related news (±2 days)
              </p>
              {a.related_events.map((e, i) => (
                <div key={i} className="flex items-start gap-2">
                  <span
                    className="text-xs px-1.5 py-0.5 rounded font-semibold shrink-0 mt-0.5"
                    style={{
                      background: `${EVENT_COLORS[e.event_type] || '#555'}22`,
                      color: EVENT_COLORS[e.event_type] || '#aaa',
                    }}
                  >
                    {e.event_type}
                  </span>
                  <div>
                    <p className="text-gray-200 text-xs font-medium leading-snug">{e.title}</p>
                    <p className="text-gray-600 text-xs mt-0.5">{e.source} · {fmtDate(e.date)}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
