/**
 * ReportPanel — Triggers Module 4 (Claude AI report) on demand.
 *
 * The report is intentionally separate from the main Analyze call because
 * Module 4 makes a live Claude API call (may take several seconds).
 *
 * Requires ANTHROPIC_API_KEY set in the backend environment.
 */

import { useState } from 'react'
import { FileText, Loader2, AlertCircle, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react'
import { fetchReport } from '../data/api'

export default function ReportPanel({ ticker, startDate, endDate, threshold }) {
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [collapsed, setCollapsed] = useState(false)

  const generate = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchReport(ticker, startDate, endDate, threshold)
      setReport(data.report)
      setCollapsed(false)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <FileText size={15} className="text-amber-400" />
          <p className="text-gray-500 text-xs font-semibold uppercase tracking-widest">
            Module 4 · Claude AI Report
          </p>
        </div>
        <div className="flex items-center gap-2">
          {report && (
            <button
              onClick={() => setCollapsed((v) => !v)}
              className="text-gray-600 hover:text-gray-400 transition-colors"
              title={collapsed ? 'Expand' : 'Collapse'}
            >
              {collapsed ? <ChevronDown size={15} /> : <ChevronUp size={15} />}
            </button>
          )}
          <button
            onClick={generate}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors disabled:opacity-50 disabled:cursor-not-allowed bg-amber-500/15 text-amber-400 border-amber-500/30 hover:bg-amber-500/25"
          >
            {loading
              ? <Loader2 size={12} className="animate-spin" />
              : report
                ? <RefreshCw size={12} />
                : <FileText size={12} />
            }
            {loading ? 'Generating…' : report ? 'Regenerate' : 'Generate Report'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 text-red-400 text-xs bg-red-400/10 border border-red-400/20 rounded-lg p-3">
          <AlertCircle size={13} className="shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* Placeholder */}
      {!report && !loading && !error && (
        <div className="text-center py-6 text-gray-600">
          <p className="text-sm">Click "Generate Report" to run Module 4.</p>
          <p className="text-xs mt-1 text-gray-700">
            Uses claude-sonnet-4-6 · Requires <code className="font-mono">ANTHROPIC_API_KEY</code> in the backend env.
          </p>
        </div>
      )}

      {/* Loading shimmer */}
      {loading && (
        <div className="space-y-2 animate-pulse mt-2">
          {[100, 90, 95, 70, 85].map((w, i) => (
            <div
              key={i}
              className="h-3 bg-gray-800 rounded"
              style={{ width: `${w}%` }}
            />
          ))}
        </div>
      )}

      {/* Report text */}
      {report && !collapsed && (
        <div className="mt-2 space-y-3">
          {report.split('\n\n').filter(Boolean).map((para, i) => (
            <p key={i} className="text-gray-300 text-sm leading-relaxed">
              {para}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}
