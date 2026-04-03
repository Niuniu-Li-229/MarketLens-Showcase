/**
 * StockChart — Interactive price chart with anomaly markers.
 *
 * Data sources:
 *   prices[]   — from Module 1 (DataFetcher)
 *   anomalies[] — from Module 2 (FunnelDetector)
 *
 * Anomaly days are highlighted with a coloured dot on the price line:
 *   green  = price gained on that day
 *   red    = price dropped on that day
 */

import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Brush,
} from 'recharts'

const EVENT_COLORS = {
  EARNINGS:   '#10b981',
  ANALYST:    '#3b82f6',
  REGULATORY: '#f59e0b',
  MACRO:      '#ef4444',
  PRODUCT:    '#8b5cf6',
  OTHER:      '#6b7280',
}

function fmtDate(dateStr) {
  return new Date(dateStr + 'T00:00:00').toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
  })
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-3 text-xs shadow-2xl max-w-xs">
      <p className="text-gray-400 mb-1.5 font-medium">{fmtDate(d.date)}</p>
      <p className="text-white font-bold text-sm">${d.close.toFixed(2)}</p>
      <div className="mt-1 text-gray-400 space-y-0.5">
        <p>O {d.open.toFixed(2)} · H {d.high.toFixed(2)} · L {d.low.toFixed(2)}</p>
        <p>Vol {(d.volume / 1_000_000).toFixed(1)}M</p>
      </div>

      {d.anomaly && (
        <div className="mt-2 pt-2 border-t border-gray-700">
          <p className={`font-bold mb-1 ${d.anomaly.is_gain ? 'text-emerald-400' : 'text-red-400'}`}>
            ⚡ Anomaly: {d.anomaly.is_gain ? '+' : ''}{d.anomaly.percent_change.toFixed(2)}%
          </p>
          {d.anomaly.related_events.map((e, i) => (
            <div key={i} className="flex items-start gap-1.5 mt-1">
              <span
                className="text-xs font-semibold shrink-0 mt-0.5"
                style={{ color: EVENT_COLORS[e.event_type] || '#aaa' }}
              >
                [{e.event_type}]
              </span>
              <span className="text-gray-300">{e.title}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function AnomalyDot(props) {
  const { cx, cy, payload } = props
  if (!payload?.anomaly) return null
  const fill = payload.anomaly.is_gain ? '#10b981' : '#ef4444'
  return <circle cx={cx} cy={cy} r={6} fill={fill} stroke="#1f2937" strokeWidth={2} />
}

export default function StockChart({ prices, anomalies, ticker, tickerColor = '#3b82f6' }) {
  if (!prices?.length) return null

  // Index anomalies by date for O(1) lookup
  const anomalyByDate = Object.fromEntries(anomalies.map((a) => [a.date, a]))

  const data = prices.map((p) => ({
    ...p,
    anomaly: anomalyByDate[p.date] || null,
  }))

  const closes = prices.map((p) => p.close)
  const padding = (Math.max(...closes) - Math.min(...closes)) * 0.1 || 1
  const yMin = Math.min(...closes) - padding
  const yMax = Math.max(...closes) + padding

  return (
    <div>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="date"
            tickFormatter={fmtDate}
            tick={{ fill: '#6b7280', fontSize: 11 }}
            axisLine={{ stroke: '#374151' }}
            tickLine={false}
          />
          <YAxis
            domain={[yMin, yMax]}
            tickFormatter={(v) => `$${v.toFixed(0)}`}
            tick={{ fill: '#6b7280', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={56}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="close"
            stroke={tickerColor}
            strokeWidth={2.5}
            dot={<AnomalyDot />}
            activeDot={{ r: 4, fill: tickerColor, strokeWidth: 0 }}
          />
          {data.length > 6 && (
            <Brush
              dataKey="date"
              height={22}
              stroke="#374151"
              fill="#111827"
              tickFormatter={fmtDate}
              travellerWidth={7}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center gap-5 mt-3 text-xs text-gray-500">
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block w-6 h-0.5 rounded"
            style={{ background: tickerColor }}
          />
          {ticker} close
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full bg-emerald-400" />
          Anomaly gain
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-full bg-red-400" />
          Anomaly drop
        </span>
      </div>
    </div>
  )
}
