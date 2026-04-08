/**
 * Chart1Price — Module 1 output.
 * Mirrors module5's plot_price_chart layout:
 *   Panel 1: Close price + MA20 + MA60 + S&P500 (right axis)
 *   Panel 2: Volume bars
 *   Panel 3: Cumulative return % comparison (ticker vs S&P500)
 *   Footer:  stat boxes (latest close, ticker return, S&P500 return, vs S&P500)
 */

import {
  ComposedChart, LineChart,
  Line, Bar, YAxis, XAxis,
  CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts'

// Module5 colour palette
const C = {
  blue:   '#1d4ed8',
  orange: '#f97316',
  purple: '#7c3aed',
  gray:   '#94a3b8',
  green:  '#16a34a',
  red:    '#dc2626',
}

function fmtDate(d) {
  return new Date(d + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
}
function fmtDateFull(d) {
  return new Date(d + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function PriceTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-3 text-xs shadow-lg">
      <p className="text-gray-500 mb-1.5 font-medium">{fmtDateFull(label)}</p>
      <p className="font-bold text-gray-900">${d.close?.toFixed(2)}</p>
      <div className="mt-1 text-gray-500 space-y-0.5">
        <p>O {d.open?.toFixed(2)} · H {d.high?.toFixed(2)} · L {d.low?.toFixed(2)}</p>
        <p>Vol {d.volume ? (d.volume / 1_000_000).toFixed(1) + 'M' : '—'}</p>
        {d.ma20  && <p>MA20 <span style={{ color: C.orange }}>${d.ma20.toFixed(2)}</span></p>}
        {d.ma60  && <p>MA60 <span style={{ color: C.purple }}>${d.ma60.toFixed(2)}</span></p>}
        {d.spy_close && <p>S&P500 <span style={{ color: C.gray }}>${d.spy_close.toFixed(2)}</span></p>}
      </div>
    </div>
  )
}

function ReturnTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-3 text-xs shadow-lg">
      <p className="text-gray-500 mb-1">{fmtDateFull(label)}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value > 0 ? '+' : ''}{p.value?.toFixed(2)}%
        </p>
      ))}
    </div>
  )
}

function StatBox({ label, value, color }) {
  return (
    <div className="flex flex-col">
      <span className="text-xs text-gray-400">{label}</span>
      <span className="text-xl font-bold mt-0.5" style={{ color }}>{value}</span>
    </div>
  )
}

export default function Chart1Price({ prices, spyPrices, ticker }) {
  if (!prices?.length) return null

  // Build spy lookup by date
  const spyByDate = Object.fromEntries((spyPrices || []).map(s => [s.date, s.close]))

  // Merged data for price panel
  const data = prices.map(p => ({
    ...p,
    spy_close: spyByDate[p.date] ?? null,
  }))

  // Volume data
  const volData = prices.map(p => ({ date: p.date, volume: p.volume }))

  // Return % data — both series anchored to their own first value
  const baseClose = prices[0].close
  const spyDates  = spyPrices?.map(s => s.date) ?? []
  const baseSpy   = spyPrices?.[0]?.close

  const returnData = prices.map(p => {
    const tickerRet = ((p.close - baseClose) / baseClose) * 100
    const spyClose  = spyByDate[p.date]
    const spyRet    = baseSpy && spyClose ? ((spyClose - baseSpy) / baseSpy) * 100 : null
    return { date: p.date, tickerRet: +tickerRet.toFixed(2), spyRet: spyRet != null ? +spyRet.toFixed(2) : null }
  })

  // Summary stats
  const lastClose  = prices[prices.length - 1].close
  const totalRet   = ((lastClose - baseClose) / baseClose) * 100
  const lastSpy    = spyPrices?.[spyPrices.length - 1]?.close
  const spyTotalRet = baseSpy && lastSpy ? ((lastSpy - baseSpy) / baseSpy) * 100 : null
  const outperf     = spyTotalRet != null ? totalRet - spyTotalRet : null

  // Y domain for price panel
  const closes  = prices.map(p => p.close)
  const pad     = (Math.max(...closes) - Math.min(...closes)) * 0.06 || 1
  const yDomain = [Math.min(...closes) - pad, Math.max(...closes) + pad]

  // Y domain for S&P500 right axis
  const spyCloses = Object.values(spyByDate).filter(Boolean)
  const spyPad    = spyCloses.length ? (Math.max(...spyCloses) - Math.min(...spyCloses)) * 0.06 || 10 : 10
  const spyDomain = spyCloses.length ? [Math.min(...spyCloses) - spyPad, Math.max(...spyCloses) + spyPad] : ['auto','auto']

  const tickCount = Math.min(8, prices.length)

  return (
    <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm space-y-1">
      <p className="text-[10px] text-gray-400 font-semibold uppercase tracking-widest mb-3">
        MODULE 1 — OUTPUT
      </p>

      {/* Panel 1: Price + MAs + S&P500 */}
      <p className="text-xs text-gray-500 font-medium mb-1">
        {ticker} — price, moving averages &amp; S&amp;P500 comparison
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={data} margin={{ top: 4, right: 56, left: 4, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis
            dataKey="date" tickFormatter={fmtDate}
            tick={{ fill: C.gray, fontSize: 10 }} tickLine={false}
            axisLine={{ stroke: '#e2e8f0' }}
            interval={Math.floor(data.length / tickCount)}
          />
          <YAxis
            yAxisId="left"
            domain={yDomain}
            tickFormatter={v => `$${v.toFixed(0)}`}
            tick={{ fill: C.gray, fontSize: 10 }} tickLine={false} axisLine={false}
            width={52}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            domain={spyDomain}
            tickFormatter={v => `$${v.toFixed(0)}`}
            tick={{ fill: C.gray, fontSize: 10 }} tickLine={false} axisLine={false}
            width={52}
          />
          <Tooltip content={<PriceTooltip />} />
          <Legend
            iconType="line" iconSize={12}
            formatter={(v) => <span style={{ fontSize: 11, color: '#64748b' }}>{v}</span>}
          />
          <Line yAxisId="left"  dataKey="close"     name="Close price" stroke={C.blue}   strokeWidth={1.5} dot={false} />
          <Line yAxisId="left"  dataKey="ma20"       name="MA20"        stroke={C.orange} strokeWidth={1}   dot={false} connectNulls />
          <Line yAxisId="left"  dataKey="ma60"       name="MA60"        stroke={C.purple} strokeWidth={1}   dot={false} connectNulls />
          {spyCloses.length > 0 && (
            <Line yAxisId="right" dataKey="spy_close" name="S&P500 (right)" stroke={C.gray} strokeWidth={1} dot={false} strokeDasharray="4 2" connectNulls />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Panel 2: Volume */}
      <p className="text-xs text-gray-400 font-medium mt-3 mb-1">Volume (M shares)</p>
      <ResponsiveContainer width="100%" height={70}>
        <ComposedChart data={volData} margin={{ top: 0, right: 56, left: 4, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
          <XAxis dataKey="date" hide />
          <YAxis
            tickFormatter={v => `${(v / 1e6).toFixed(0)}M`}
            tick={{ fill: C.gray, fontSize: 9 }} tickLine={false} axisLine={false}
            width={52}
          />
          <Bar dataKey="volume" fill={C.gray} opacity={0.5} radius={[1,1,0,0]} />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Panel 3: Return % comparison */}
      {spyTotalRet != null && (
        <>
          <p className="text-xs text-gray-400 font-medium mt-3 mb-1">Cumulative return %</p>
          <ResponsiveContainer width="100%" height={100}>
            <LineChart data={returnData} margin={{ top: 0, right: 56, left: 4, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis
                dataKey="date" tickFormatter={fmtDate}
                tick={{ fill: C.gray, fontSize: 9 }} tickLine={false} axisLine={{ stroke: '#e2e8f0' }}
                interval={Math.floor(returnData.length / tickCount)}
              />
              <YAxis
                tickFormatter={v => `${v > 0 ? '+' : ''}${v.toFixed(0)}%`}
                tick={{ fill: C.gray, fontSize: 9 }} tickLine={false} axisLine={false}
                width={52}
              />
              <Tooltip content={<ReturnTooltip />} />
              <Line dataKey="tickerRet" name={`${ticker} return`}  stroke={C.blue} strokeWidth={1.5} dot={false} />
              <Line dataKey="spyRet"    name="S&P500 return" stroke={C.gray} strokeWidth={1}   dot={false} strokeDasharray="4 2" connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}

      {/* Footer stat boxes */}
      <div className="flex flex-wrap gap-8 pt-4 mt-2 border-t border-gray-100">
        <StatBox label="Latest close"    value={`$${lastClose.toFixed(2)}`}            color={C.blue} />
        <StatBox label={`${ticker} return`} value={`${totalRet >= 0 ? '+' : ''}${totalRet.toFixed(1)}%`} color={totalRet >= 0 ? C.green : C.red} />
        {spyTotalRet != null && (
          <StatBox label="S&P500 return"  value={`${spyTotalRet >= 0 ? '+' : ''}${spyTotalRet.toFixed(1)}%`} color={C.gray} />
        )}
        {outperf != null && (
          <StatBox label="vs S&P500"      value={`${outperf >= 0 ? '+' : ''}${outperf.toFixed(1)}%`}        color={outperf >= 0 ? C.green : C.red} />
        )}
      </div>
    </div>
  )
}
