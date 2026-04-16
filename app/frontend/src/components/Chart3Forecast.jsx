/**
 * Chart3Forecast — Module 3 output (on-demand).
 * Mirrors module5's plot_prediction_chart layout:
 *   Left panel:  Transformer (actual vs predicted)
 *   Right panel: TFT (actual vs predicted)  ← new: side-by-side when TFT data present
 *   Side panel:  Sentiment gauge (FinBERT / mock)
 *
 * Falls back to single Transformer + sentiment if TFT fields are absent (old caches).
 */

import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'
import { BarChart2, Brain, Loader2, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { fetchForecast } from '../data/api'

const C = {
  blue:   '#1d4ed8',
  orange: '#f97316',
  green:  '#16a34a',
  red:    '#dc2626',
  gray:   '#94a3b8',
}

const SENTIMENT_CFG = {
  bullish: { color: C.green,  Icon: TrendingUp,   label: 'Bullish' },
  bearish: { color: C.red,    Icon: TrendingDown,  label: 'Bearish' },
  neutral: { color: C.gray,   Icon: Minus,         label: 'Neutral' },
}

function fmtDate(d) {
  return new Date(d + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
}
function fmtDateFull(d) {
  return new Date(d + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function ForecastTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-3 text-xs shadow-lg">
      <p className="text-gray-500 mb-1">{fmtDateFull(label)}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: ${p.value?.toFixed(2)}
        </p>
      ))}
    </div>
  )
}

function SentimentGauge({ score }) {
  const pct   = ((score + 1) / 2) * 100
  const color = score > 0.1 ? C.green : score < -0.1 ? C.red : C.gray
  return (
    <div className="mt-3">
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>Bearish −1</span>
        <span>Neutral 0</span>
        <span>Bullish +1</span>
      </div>
      <div className="relative h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <p className="text-xs text-gray-400 mt-1.5 text-center">
        score: {score > 0 ? '+' : ''}{score.toFixed(2)}
      </p>
    </div>
  )
}

/** One forecast panel (Transformer or TFT). */
function ModelPanel({ title, modelName, chartData, day5Price, dirAccuracy, mae, sectorName }) {
  const hasChart = chartData?.length > 0
  const tickCount = Math.min(8, chartData?.length ?? 0)

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <BarChart2 size={13} style={{ color: C.blue }} />
        <p className="text-xs font-semibold text-gray-600 truncate">
          {modelName ?? title}
        </p>
      </div>

      {hasChart ? (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData} margin={{ top: 4, right: 8, left: 4, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              dataKey="date" tickFormatter={fmtDate}
              tick={{ fill: C.gray, fontSize: 10 }} tickLine={false}
              axisLine={{ stroke: '#e2e8f0' }}
              interval={Math.floor(chartData.length / tickCount)}
            />
            <YAxis
              tickFormatter={v => `$${v.toFixed(0)}`}
              tick={{ fill: C.gray, fontSize: 10 }} tickLine={false} axisLine={false}
              width={52}
            />
            <Tooltip content={<ForecastTooltip />} />
            <Legend
              iconType="line" iconSize={12}
              formatter={v => <span style={{ fontSize: 11, color: '#64748b' }}>{v}</span>}
            />
            <Line dataKey="actual"    name="Actual price" stroke={C.blue}   strokeWidth={1.8} dot={false} />
            <Line dataKey="predicted" name="Prediction"   stroke={C.orange} strokeWidth={1.5} dot={false} strokeDasharray="5 3" connectNulls />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="h-[200px] bg-gray-50 rounded-xl border border-dashed border-gray-200 flex items-center justify-center">
          <p className="text-xs text-gray-400">No data</p>
        </div>
      )}

      {day5Price != null && (
        <div className="flex flex-wrap gap-5 mt-3 pt-3 border-t border-gray-100 text-sm">
          <div>
            <p className="text-xs text-gray-400">Day-5 prediction</p>
            <p className="font-bold text-lg" style={{ color: C.blue }}>
              ${day5Price.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Directional accuracy</p>
            <p className="font-bold text-lg text-gray-800">
              {(dirAccuracy * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400">MAE</p>
            <p className="font-bold text-lg text-gray-800">${mae.toFixed(2)}</p>
          </div>
          {sectorName && (
            <div>
              <p className="text-xs text-gray-400">Sector</p>
              <p className="font-bold text-gray-800 text-sm">{sectorName}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function Chart3Forecast({
  ticker, startDate, endDate,
  sentimentScore, sentimentLabel, eventCount,
  onForecastLoaded,
}) {
  const [forecast, setForecast] = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchForecast(ticker, startDate, endDate)
      setForecast(data)
      onForecastLoaded?.(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const conf     = SENTIMENT_CFG[sentimentLabel] ?? SENTIMENT_CFG.neutral
  const SentIcon = conf.Icon

  // Transformer chart data
  const tfChartData = forecast?.test_dates?.length
    ? forecast.test_dates.map((d, i) => ({
        date:      d,
        actual:    forecast.actual[i]    != null ? +forecast.actual[i].toFixed(2)    : null,
        predicted: forecast.predicted[i] != null ? +forecast.predicted[i].toFixed(2) : null,
      }))
    : []

  // TFT chart data — present on new runs, absent from old caches
  const hasTFT = !!(forecast?.tft_model_name && forecast?.tft_test_dates?.length)
  const tftChartData = hasTFT
    ? forecast.tft_test_dates.map((d, i) => ({
        date:      d,
        actual:    forecast.tft_actual[i]    != null ? +forecast.tft_actual[i].toFixed(2)    : null,
        predicted: forecast.tft_predicted[i] != null ? +forecast.tft_predicted[i].toFixed(2) : null,
      }))
    : []

  // Grid columns:
  //   loaded + TFT → 3 equal cols (Transformer | TFT | Sentiment)
  //   loaded / loading / placeholder → col-span-2 chart + Sentiment
  const gridCols = hasTFT ? 'md:grid-cols-3' : 'md:grid-cols-3'

  return (
    <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
      <p className="text-[10px] text-gray-400 font-semibold uppercase tracking-widest mb-4">
        MODULE 3 — OUTPUT
      </p>

      {/* Run / Refresh button row */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-xs font-semibold text-gray-500">
          {forecast
            ? hasTFT
              ? `${ticker} — Transformer vs TFT`
              : `${ticker} — Transformer forecast`
            : 'Transformer & TFT Price Forecast'}
        </p>
        <div className="flex items-center gap-2">
          {!forecast && (
            <button
              onClick={run}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors disabled:opacity-50 disabled:cursor-not-allowed bg-blue-50 text-blue-600 border-blue-200 hover:bg-blue-100"
            >
              {loading
                ? <Loader2 size={12} className="animate-spin" />
                : <BarChart2 size={12} />}
              {loading ? 'Training…' : 'Run Forecast'}
            </button>
          )}
          {forecast && (
            <button
              onClick={run}
              disabled={loading}
              className="text-xs text-gray-400 hover:text-gray-600 transition-colors flex items-center gap-1"
            >
              {loading ? <Loader2 size={11} className="animate-spin" /> : '↺'}
              {loading ? 'Re-training…' : 'Refresh'}
            </button>
          )}
        </div>
      </div>

      {error && (
        <p className="text-xs text-red-500 bg-red-50 border border-red-200 rounded-lg p-3 mb-4">{error}</p>
      )}

      <div className={`grid grid-cols-1 ${gridCols} gap-5`}>

        {/* ── Forecast chart(s) ── */}
        {!forecast && !loading && !error && (
          <div className="md:col-span-2">
            <div className="flex items-center justify-center h-52 bg-gray-50 rounded-xl border border-dashed border-gray-200">
              <div className="text-center text-gray-400">
                <BarChart2 size={32} className="mx-auto mb-2 opacity-30" />
                <p className="text-sm">Click "Run Forecast" to train Transformer &amp; TFT.</p>
                <p className="text-xs mt-1 text-gray-400">First run: ~1–2 min · Cached thereafter</p>
              </div>
            </div>
          </div>
        )}

        {loading && !forecast && (
          <div className="md:col-span-2">
            <div className="h-52 bg-gray-50 rounded-xl border border-gray-100 flex items-center justify-center">
              <div className="text-center text-gray-400">
                <Loader2 size={28} className="mx-auto mb-2 animate-spin opacity-40" />
                <p className="text-sm">Training Transformer &amp; TFT…</p>
                <p className="text-xs mt-1">This may take 1–2 minutes on first run</p>
              </div>
            </div>
          </div>
        )}

        {forecast && !hasTFT && (
          <div className="md:col-span-2">
            <ModelPanel
              title="Transformer Forecast"
              modelName={forecast.model_name}
              chartData={tfChartData}
              day5Price={forecast.day5_price}
              dirAccuracy={forecast.dir_accuracy}
              mae={forecast.mae}
              sectorName={forecast.sector_name}
            />
          </div>
        )}

        {forecast && hasTFT && (
          <>
            <ModelPanel
              title="Transformer"
              modelName={forecast.model_name}
              chartData={tfChartData}
              day5Price={forecast.day5_price}
              dirAccuracy={forecast.dir_accuracy}
              mae={forecast.mae}
              sectorName={forecast.sector_name}
            />
            <ModelPanel
              title="TFT"
              modelName={forecast.tft_model_name}
              chartData={tftChartData}
              day5Price={forecast.tft_day5_price}
              dirAccuracy={forecast.tft_dir_accuracy}
              mae={forecast.tft_mae}
              sectorName={null}
            />
          </>
        )}

        {/* ── Sentiment panel (always right col) ── */}
        <div className="border border-gray-100 rounded-xl p-4 bg-gray-50">
          <div className="flex items-center gap-2 mb-4">
            <Brain size={14} style={{ color: '#7c3aed' }} />
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
              News Sentiment
            </p>
          </div>

          <div className="flex items-center gap-3 mb-2">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
              style={{ background: `${conf.color}1a` }}
            >
              <SentIcon size={20} style={{ color: conf.color }} />
            </div>
            <div>
              <p className="font-bold text-gray-900 text-xl">{conf.label}</p>
              <p className="text-gray-400 text-xs">
                {eventCount ?? 0} news event{eventCount !== 1 ? 's' : ''} analysed
              </p>
            </div>
          </div>

          <SentimentGauge score={sentimentScore ?? 0} />

          <p className="text-xs text-gray-400 mt-4 pt-3 border-t border-gray-200 leading-relaxed">
            Sentiment from news events. Run with FinBERT for NLP-based scores; currently uses heuristic weighting.
          </p>
        </div>

      </div>
    </div>
  )
}
