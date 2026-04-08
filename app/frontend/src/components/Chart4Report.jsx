/**
 * Chart4Report — Module 4 output (on-demand).
 * Mirrors module5's plot_report_card layout:
 *   Row 1: Core stats (ticker, return, sentiment, predicted D5, anomalies)
 *   Row 2: Market indicators (P/E, market cap, 52w position, beta,
 *           analyst target, analyst rating, VIX, vs S&P500 30d)
 *   Row 3: Three-paragraph Claude AI report (Performance · Anomalies · Outlook)
 *
 * Market metrics load independently via /api/market-info.
 * Claude report loads via /api/report.
 */

import { useState, useEffect } from 'react'
import { FileText, Loader2, TrendingUp, TrendingDown, Minus, RefreshCw } from 'lucide-react'
import { fetchReport, fetchMarketInfo } from '../data/api'

const C = {
  blue:   '#1d4ed8',
  orange: '#f97316',
  purple: '#7c3aed',
  green:  '#16a34a',
  red:    '#dc2626',
  gray:   '#94a3b8',
}

const SENTIMENT_CFG = {
  bullish: { color: C.green  },
  bearish: { color: C.red    },
  neutral: { color: C.gray   },
}

const REC_COLORS = {
  STRONG_BUY: C.green, BUY: C.green,
  HOLD: C.orange,
  SELL: C.red, STRONG_SELL: C.red,
}

function StatCell({ label, value, color, large }) {
  return (
    <div className="flex flex-col">
      <span className="text-xs text-gray-400 mb-0.5">{label}</span>
      <span
        className={`font-bold ${large ? 'text-xl' : 'text-sm'}`}
        style={{ color: color || C.blue }}
      >
        {value ?? 'n/a'}
      </span>
    </div>
  )
}

function MktCell({ label, value, color }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-gray-400">{label}</span>
      <span className="font-bold text-sm" style={{ color: color || C.gray }}>
        {value ?? 'n/a'}
      </span>
    </div>
  )
}

export default function Chart4Report({
  ticker, startDate, endDate,
  totalReturn, anomalyCount, sentimentScore, sentimentLabel,
}) {
  const [mktInfo,  setMktInfo]  = useState(null)
  const [mktError, setMktError] = useState(null)

  const [report,  setReport]  = useState(null)
  const [repLoad, setRepLoad] = useState(false)
  const [repError, setRepError] = useState(null)
  const [collapsed, setCollapsed] = useState(false)

  // Market info loads automatically when component mounts
  useEffect(() => {
    fetchMarketInfo(ticker)
      .then(setMktInfo)
      .catch(e => setMktError(e.message))
  }, [ticker])

  const generateReport = async () => {
    setRepLoad(true)
    setRepError(null)
    try {
      const data = await fetchReport(ticker, startDate, endDate)
      setReport(data.report)
      setCollapsed(false)
    } catch (e) {
      setRepError(e.message)
    } finally {
      setRepLoad(false)
    }
  }

  const sentConf   = SENTIMENT_CFG[sentimentLabel] ?? SENTIMENT_CFG.neutral
  const retColor   = totalReturn >= 0 ? C.green : C.red
  const predPrice  = mktInfo?.analyst_target

  // Market metric colour helpers
  const peColor  = mktInfo?.pe_ratio ? (mktInfo.pe_ratio < 30 ? C.blue : C.orange) : C.gray
  const w52Color = mktInfo?.week52_position != null
    ? (mktInfo.week52_position > 60 ? C.green : mktInfo.week52_position < 30 ? C.red : C.orange)
    : C.gray
  const betaColor = mktInfo?.beta ? (mktInfo.beta > 1.2 ? C.orange : C.green) : C.gray
  const recColor  = REC_COLORS[mktInfo?.analyst_rating] ?? C.gray
  const vixColor  = mktInfo?.vix
    ? (mktInfo.vix < 15 ? C.green : mktInfo.vix < 25 ? C.orange : C.red)
    : C.gray
  const relColor  = mktInfo?.rel_perf_30d != null
    ? (mktInfo.rel_perf_30d > 0 ? C.green : C.red)
    : C.gray
  const upColor   = mktInfo?.upside != null
    ? (mktInfo.upside > 5 ? C.green : mktInfo.upside < -5 ? C.red : C.gray)
    : C.gray

  return (
    <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm space-y-5">
      <p className="text-[10px] text-gray-400 font-semibold uppercase tracking-widest">
        MODULE 4 — OUTPUT
      </p>

      <p className="text-sm font-medium text-gray-500">
        {ticker} — AI analysis report
      </p>

      {/* ── Row 1: Core stats ── */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 pb-4 border-b border-gray-100">
        <StatCell label="Ticker"        value={ticker}                                    color={C.blue}   large />
        <StatCell label="Period return" value={`${totalReturn >= 0 ? '+' : ''}${totalReturn?.toFixed(2)}%`} color={retColor} large />
        <StatCell
          label="Sentiment"
          value={`${sentimentScore > 0 ? '+' : ''}${sentimentScore?.toFixed(2)} (${sentimentLabel})`}
          color={sentConf.color}
          large
        />
        <StatCell
          label="Analyst target"
          value={mktInfo?.analyst_target ? `$${mktInfo.analyst_target.toFixed(2)}` : '…'}
          color={C.purple}
          large
        />
        <StatCell label="Anomalies" value={String(anomalyCount ?? 0)} color={C.orange} large />
      </div>

      {/* ── Row 2: Market indicators ── */}
      <div>
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">Market indicators</p>
        {mktError && (
          <p className="text-xs text-red-500 mb-2">{mktError}</p>
        )}
        {!mktInfo && !mktError && (
          <p className="text-xs text-gray-400 flex items-center gap-1.5">
            <Loader2 size={11} className="animate-spin" /> Loading market data…
          </p>
        )}
        {mktInfo && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-6 gap-y-4">
            <MktCell label="P/E ratio"
              value={mktInfo.pe_ratio ? `${mktInfo.pe_ratio}x` : null}
              color={peColor} />
            <MktCell label="Market cap"     value={mktInfo.market_cap}   color={C.blue} />
            <MktCell label="52w position"
              value={mktInfo.week52_position != null ? `${mktInfo.week52_position}% of range` : null}
              color={w52Color} />
            <MktCell label="Beta"
              value={mktInfo.beta != null ? `${mktInfo.beta}  ${mktInfo.beta > 1.5 ? 'High vol' : mktInfo.beta > 1.0 ? 'Above mkt' : 'Below mkt'}` : null}
              color={betaColor} />
            <MktCell label="Analyst target"
              value={mktInfo.analyst_target ? `$${mktInfo.analyst_target}${mktInfo.upside != null ? `  ${mktInfo.upside > 0 ? '+' : ''}${mktInfo.upside}% upside` : ''}` : null}
              color={upColor} />
            <MktCell label="Analyst rating" value={mktInfo.analyst_rating} color={recColor} />
            <MktCell label="VIX (fear)"
              value={mktInfo.vix != null ? `${mktInfo.vix}  ${mktInfo.vix_label}` : null}
              color={vixColor} />
            <MktCell label="vs S&P500 (30d)"
              value={mktInfo.rel_perf_30d != null ? `${mktInfo.rel_perf_30d > 0 ? '+' : ''}${mktInfo.rel_perf_30d}%  rel. perf` : null}
              color={relColor} />
          </div>
        )}
      </div>

      {/* ── Row 3: Claude report ── */}
      <div className="border-t border-gray-100 pt-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <FileText size={14} style={{ color: C.orange }} />
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
              AI-generated analyst report
            </p>
          </div>
          <div className="flex items-center gap-2">
            {report && (
              <button
                onClick={() => setCollapsed(v => !v)}
                className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
              >
                {collapsed ? 'Expand ▾' : 'Collapse ▴'}
              </button>
            )}
            <button
              onClick={generateReport}
              disabled={repLoad}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              style={{
                background: '#fff7ed',
                color: C.orange,
                borderColor: '#fed7aa',
              }}
            >
              {repLoad
                ? <Loader2 size={12} className="animate-spin" />
                : report
                  ? <RefreshCw size={12} />
                  : <FileText size={12} />}
              {repLoad ? 'Generating…' : report ? 'Regenerate' : 'Generate Report'}
            </button>
          </div>
        </div>

        {repError && (
          <p className="text-xs text-red-500 bg-red-50 border border-red-200 rounded-lg p-3">{repError}</p>
        )}

        {!report && !repLoad && !repError && (
          <div className="text-center py-8 text-gray-400">
            <FileText size={28} className="mx-auto mb-2 opacity-30" />
            <p className="text-sm">Click "Generate Report" to run Module 4.</p>
            <p className="text-xs mt-1 text-gray-400">
              Uses claude-sonnet-4-6 · Requires <code className="font-mono">ANTHROPIC_API_KEY</code> in the backend env.
            </p>
          </div>
        )}

        {repLoad && !report && (
          <div className="space-y-2 animate-pulse">
            {[100, 88, 94, 72, 83, 95, 68].map((w, i) => (
              <div key={i} className="h-3 bg-gray-100 rounded" style={{ width: `${w}%` }} />
            ))}
          </div>
        )}

        {report && !collapsed && (
          <div className="space-y-4">
            {report.split('\n\n').filter(Boolean).map((para, i) => {
              // Bold section headers (PERFORMANCE / ANOMALIES / OUTLOOK or ## headers)
              const isHeader = /^(#{1,3}\s+|PERFORMANCE|ANOMALIES|OUTLOOK)/i.test(para.trim())
              if (isHeader) {
                const clean = para.replace(/^#{1,3}\s+/, '').trim()
                return (
                  <p key={i} className="text-xs font-bold text-gray-500 uppercase tracking-widest mt-2">
                    {clean}
                  </p>
                )
              }
              return (
                <p key={i} className="text-sm text-gray-700 leading-relaxed">
                  {para}
                </p>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
