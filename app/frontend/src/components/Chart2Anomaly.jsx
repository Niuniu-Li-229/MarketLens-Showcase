/**
 * Chart2Anomaly — Module 2 output.
 * Mirrors module5's plot_anomaly_chart:
 *   • Price line (blue)
 *   • Green dots for positive anomalies, red dots for negative
 *   • Inline % labels on each anomaly
 *   • Sortable anomaly table below the chart with related news events
 */

import { useState } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { TrendingUp, TrendingDown, ChevronDown, ChevronUp } from 'lucide-react'

const C = {
  blue:  '#1d4ed8',
  green: '#16a34a',
  red:   '#dc2626',
  gray:  '#94a3b8',
}

const EVENT_COLORS = {
  EARNINGS:   '#16a34a',
  ANALYST:    '#1d4ed8',
  REGULATORY: '#f59e0b',
  MACRO:      '#dc2626',
  PRODUCT:    '#7c3aed',
  PERSONNEL:  '#0891b2',
  OTHER:      '#94a3b8',
}

function fmtDate(d) {
  return new Date(d + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
}
function fmtDateFull(d) {
  return new Date(d + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

// Custom dot — only renders on anomaly days
function AnomalyDot({ cx, cy, payload }) {
  if (!payload?.anomaly) return null
  const fill = payload.anomaly.is_gain ? C.green : C.red
  return <circle cx={cx} cy={cy} r={6} fill={fill} stroke="white" strokeWidth={1.5} />
}

// Custom label above/below each anomaly dot
function AnomalyLabel({ x, y, payload }) {
  if (!payload?.anomaly) return null
  const pct   = payload.anomaly.percent_change
  const color = pct > 0 ? C.green : C.red
  const yPos  = pct > 0 ? y - 14 : y + 18
  return (
    <text x={x} y={yPos} textAnchor="middle" fontSize={9} fontWeight="700" fill={color}>
      {pct > 0 ? '+' : ''}{pct.toFixed(1)}%
    </text>
  )
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null

  const events = d.anomaly?.related_events ?? []
  const top3 = events.slice(0, 3)
  const typeCounts = {}
  events.forEach(e => { typeCounts[e.event_type] = (typeCounts[e.event_type] || 0) + 1 })
  const typeList = Object.keys(typeCounts)

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-3 text-xs shadow-lg max-w-xs">
      <p className="text-gray-500 mb-1 font-medium">{fmtDateFull(label)}</p>
      <p className="font-bold text-gray-900">${d.close?.toFixed(2)}</p>
      {d.anomaly && (
        <div className="mt-2 pt-2 border-t border-gray-100">
          <p className="font-bold mb-1" style={{ color: d.anomaly.is_gain ? C.green : C.red }}>
            ⚡ {d.anomaly.is_gain ? '+' : ''}{d.anomaly.percent_change.toFixed(2)}% anomaly
          </p>
          <p className="text-gray-500 leading-snug">{'Triggered by:' + d.anomaly.comment.split('Triggered by:')[1]?.split(' Related events:')[0]}</p>
          {events.length > 0 && (
            <div className="mt-2 pt-2 border-t border-gray-100 space-y-1.5">
              <div className="flex flex-wrap gap-1 items-center">
                <span className="text-gray-400 mr-0.5">{events.length} event{events.length !== 1 ? 's' : ''}:</span>
                {typeList.map(type => (
                  <span
                    key={type}
                    className="px-1.5 py-0.5 rounded text-[10px] font-semibold"
                    style={{
                      background: `${EVENT_COLORS[type] || '#94a3b8'}1a`,
                      color: EVENT_COLORS[type] || '#94a3b8',
                    }}
                  >
                    {typeCounts[type] > 1 ? `${type} ×${typeCounts[type]}` : type}
                  </span>
                ))}
              </div>
              <p className="text-gray-400 font-semibold">Top 3:</p>
              {top3.map((e, i) => (
                <div key={i} className="flex items-start gap-1.5">
                  <span
                    className="px-1.5 py-0.5 rounded text-[10px] font-semibold shrink-0 mt-0.5"
                    style={{
                      background: `${EVENT_COLORS[e.event_type] || '#94a3b8'}1a`,
                      color: EVENT_COLORS[e.event_type] || '#94a3b8',
                    }}
                  >
                    {e.event_type}
                  </span>
                  <p className="text-gray-700 leading-snug">{e.title}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function AnomalyRow({ a }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors text-left"
      >
        <div className="flex items-center gap-3">
          {a.is_gain
            ? <TrendingUp size={14} style={{ color: C.green }} />
            : <TrendingDown size={14} style={{ color: C.red }} />}
          <span
            className="text-base font-bold"
            style={{ color: a.is_gain ? C.green : C.red }}
          >
            {a.is_gain ? '+' : ''}{a.percent_change.toFixed(2)}%
          </span>
          <span className="text-xs text-gray-400">open→close</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-400">{fmtDateFull(a.date)}</span>
          <span className="text-xs text-gray-400">
            {a.related_events.length} event{a.related_events.length !== 1 ? 's' : ''}
          </span>
          {open ? <ChevronUp size={13} className="text-gray-400" /> : <ChevronDown size={13} className="text-gray-400" />}
        </div>
      </button>

      {open && (
        <div className="px-4 pb-4 border-t border-gray-100 space-y-3">
          <p className="text-xs text-gray-500 leading-relaxed pt-3">{a.comment}</p>
          {a.related_events.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Related news (±2 days)</p>
              {a.related_events.map((e, i) => (
                <div key={i} className="flex items-start gap-2">
                  <span
                    className="text-xs px-1.5 py-0.5 rounded font-semibold shrink-0 mt-0.5"
                    style={{
                      background: `${EVENT_COLORS[e.event_type] || '#94a3b8'}1a`,
                      color: EVENT_COLORS[e.event_type] || '#94a3b8',
                    }}
                  >
                    {e.event_type}
                  </span>
                  <div>
                    <p className="text-xs font-medium text-gray-800 leading-snug">{e.title}</p>
                    <p className="text-xs text-gray-400 mt-0.5">{e.source} · {fmtDateFull(e.date)}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function Chart2Anomaly({ prices, anomalies, ticker }) {
  if (!prices?.length) return null

  const anomalyByDate = Object.fromEntries((anomalies || []).map(a => [a.date, a]))
  const data = prices.map(p => ({ ...p, anomaly: anomalyByDate[p.date] ?? null }))

  const closes  = prices.map(p => p.close)
  const pad     = (Math.max(...closes) - Math.min(...closes)) * 0.06 || 1
  const yDomain = [Math.min(...closes) - pad, Math.max(...closes) + pad]

  const sorted = [...(anomalies || [])].sort((a, b) => Math.abs(b.percent_change) - Math.abs(a.percent_change))
  const [showAll, setShowAll] = useState(false)
  const TOP_N = 10
  const visible = showAll ? sorted : sorted.slice(0, TOP_N)
  const tickCount = Math.min(8, prices.length)

  return (
    <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
      <p className="text-[10px] text-gray-400 font-semibold uppercase tracking-widest mb-3">
        MODULE 2 — OUTPUT
      </p>
      <p className="text-xs text-gray-500 font-medium mb-3">
        {ticker} — anomaly detection &nbsp;
        <span className="text-gray-400">(top {Math.min(15, sorted.length)} of {anomalies?.length ?? 0} detected, by magnitude)</span>
      </p>

      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 20, right: 16, left: 4, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis
            dataKey="date" tickFormatter={fmtDate}
            tick={{ fill: C.gray, fontSize: 10 }} tickLine={false}
            axisLine={{ stroke: '#e2e8f0' }}
            interval={Math.floor(data.length / tickCount)}
          />
          <YAxis
            domain={yDomain}
            tickFormatter={v => `$${v.toFixed(0)}`}
            tick={{ fill: C.gray, fontSize: 10 }} tickLine={false} axisLine={false}
            width={52}
          />
          <Tooltip content={<ChartTooltip />} />
          <Line
            type="monotone"
            dataKey="close"
            stroke={C.blue}
            strokeWidth={1.5}
            dot={<AnomalyDot />}
            label={<AnomalyLabel />}
            activeDot={{ r: 4, fill: C.blue, strokeWidth: 0 }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center gap-5 mt-3 text-xs text-gray-500">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-6 h-0.5 rounded" style={{ background: C.blue }} />
          {ticker} close
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full" style={{ background: C.green }} />
          Positive anomaly
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full" style={{ background: C.red }} />
          Negative anomaly
        </span>
      </div>

      {/* Anomaly list */}
      {sorted.length > 0 && (
        <div className="mt-5 space-y-2">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">
            Top {Math.min(TOP_N, sorted.length)} anomalies by magnitude
            {sorted.length > TOP_N && (
              <span className="text-gray-300 ml-1">(of {sorted.length} total)</span>
            )}
          </p>
          {visible.map(a => <AnomalyRow key={a.date} a={a} />)}
          {sorted.length > TOP_N && (
            <button
              onClick={() => setShowAll(v => !v)}
              className="w-full mt-1 py-2 text-xs font-semibold text-blue-600 hover:text-blue-700 border border-blue-100 hover:border-blue-200 rounded-xl bg-blue-50 hover:bg-blue-100 transition-colors"
            >
              {showAll
                ? '▴ Show less'
                : `▾ Show ${sorted.length - TOP_N} more anomalies`}
            </button>
          )}
        </div>
      )}

      {sorted.length === 0 && (
        <div className="mt-4 text-center py-6 text-gray-400 text-sm">
          No anomalies detected in this range.
        </div>
      )}
    </div>
  )
}
