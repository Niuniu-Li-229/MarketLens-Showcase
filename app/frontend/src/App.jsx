import { useState } from 'react'
import { Search, Settings2, TrendingUp, TrendingDown, Loader2, AlertCircle } from 'lucide-react'
import StockChart from './components/StockChart'
import AnomalyPanel from './components/AnomalyPanel'
import PredictionPanel from './components/PredictionPanel'
import ReportPanel from './components/ReportPanel'
import { fetchAnalysis } from './data/api'

export default function App() {
  const [ticker, setTicker] = useState('NVDA')
  const [startDate, setStartDate] = useState('2025-09-02')
  const [endDate, setEndDate] = useState('2025-09-05')
  const [threshold, setThreshold] = useState(0.5)

  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleAnalyze = async (e) => {
    e?.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const data = await fetchAnalysis(ticker.trim().toUpperCase(), startDate, endDate, threshold)
      setAnalysis(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const lastPrice = analysis?.prices?.[analysis.prices.length - 1]?.close
  const isPositive = analysis ? analysis.total_return >= 0 : null

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center font-bold text-blue-400 text-sm">
              ML
            </div>
            <div>
              <span className="font-bold text-white text-lg tracking-tight">MarketLens</span>
              <span className="text-gray-500 text-sm ml-2">· Markets don't move in a vacuum</span>
            </div>
          </div>
          <span className="text-xs text-gray-600 hidden sm:block">
            Anomaly Detection · Sentiment · AI Report
          </span>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8 space-y-6">

        {/* ── Control Panel ── */}
        <form
          onSubmit={handleAnalyze}
          className="bg-gray-900 border border-gray-800 rounded-2xl p-5"
        >
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
            {/* Ticker */}
            <div>
              <label className="block text-xs text-gray-500 mb-1.5 font-medium">Ticker</label>
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="e.g. NVDA"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm font-mono placeholder-gray-600 focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>

            {/* Start Date */}
            <div>
              <label className="block text-xs text-gray-500 mb-1.5 font-medium">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>

            {/* End Date */}
            <div>
              <label className="block text-xs text-gray-500 mb-1.5 font-medium">End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>

            {/* Threshold */}
            <div>
              <label className="block text-xs text-gray-500 mb-1.5 font-medium">
                Anomaly Threshold (%)
              </label>
              <input
                type="number"
                min="0.1"
                max="20"
                step="0.1"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={loading}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />}
              {loading ? 'Analyzing…' : 'Analyze'}
            </button>
            <p className="text-xs text-gray-600">
              Mock backend: any ticker uses NVDA data · Sep 2–5 2025
            </p>
          </div>
        </form>

        {/* ── Error ── */}
        {error && (
          <div className="flex items-start gap-2 text-red-400 text-sm bg-red-400/10 border border-red-400/20 rounded-xl p-4">
            <AlertCircle size={16} className="shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* ── Results ── */}
        {analysis && (
          <>
            {/* Stock header */}
            <div className="flex items-start justify-between flex-wrap gap-4">
              <div>
                <div className="flex items-center gap-3 mb-1">
                  <h1 className="text-4xl font-black tracking-tight text-blue-400">
                    {analysis.ticker}
                  </h1>
                  <span className="text-xs px-2 py-1 rounded-full border border-blue-400/30 bg-blue-400/10 text-blue-400">
                    {analysis.anomalies.length} anomal{analysis.anomalies.length !== 1 ? 'ies' : 'y'}
                  </span>
                </div>
                <p className="text-gray-500 text-sm">
                  {analysis.start_date} → {analysis.end_date}
                </p>
              </div>
              {lastPrice && (
                <div className="text-right">
                  <div className="text-3xl font-bold text-white">${lastPrice.toFixed(2)}</div>
                  <div
                    className={`flex items-center justify-end gap-1 text-sm font-semibold mt-0.5 ${
                      isPositive ? 'text-emerald-400' : 'text-red-400'
                    }`}
                  >
                    {isPositive ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                    {isPositive ? '+' : ''}
                    {analysis.total_return.toFixed(2)}% period return
                  </div>
                </div>
              )}
            </div>

            {/* Chart — Module 1 prices + Module 2 anomaly markers */}
            <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
              <p className="text-gray-500 text-xs font-semibold uppercase tracking-widest mb-4">
                Module 1 · Price History &nbsp;+&nbsp; Module 2 · Anomaly Markers
              </p>
              <StockChart
                prices={analysis.prices}
                anomalies={analysis.anomalies}
                ticker={analysis.ticker}
              />
            </div>

            {/* Anomaly list — Module 2 */}
            <div>
              <p className="text-gray-500 text-xs font-semibold uppercase tracking-widest mb-3">
                Module 2 · Detected Anomalies ({analysis.anomalies.length})
              </p>
              <AnomalyPanel anomalies={analysis.anomalies} />
            </div>

            {/* Prediction + Sentiment — Module 3 */}
            <div>
              <p className="text-gray-500 text-xs font-semibold uppercase tracking-widest mb-3">
                Module 3 · Forecast & Sentiment
              </p>
              <PredictionPanel analysis={analysis} />
            </div>

            {/* Claude Report — Module 4 */}
            <ReportPanel
              ticker={analysis.ticker}
              startDate={analysis.start_date}
              endDate={analysis.end_date}
              threshold={threshold}
            />
          </>
        )}

        {/* ── Empty state ── */}
        {!analysis && !loading && !error && (
          <div className="text-center py-20 text-gray-600">
            <div className="text-5xl mb-4">📈</div>
            <p className="text-lg font-medium text-gray-500">Enter a ticker and date range, then click Analyze</p>
            <p className="text-sm mt-2 text-gray-600">
              The pipeline runs Modules 1 → 2 → 3 live. Module 4 (Claude report) is on demand.
            </p>
          </div>
        )}
      </main>

      <footer className="border-t border-gray-800 mt-16 py-6 text-center text-xs text-gray-700">
        MarketLens · For educational purposes only · Not financial advice
      </footer>
    </div>
  )
}
