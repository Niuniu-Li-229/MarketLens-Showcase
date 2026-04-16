import { useState, useEffect } from 'react'
import { Search, Loader2, AlertCircle, AlertTriangle } from 'lucide-react'
import Chart1Price    from './components/Chart1Price'
import Chart2Anomaly  from './components/Chart2Anomaly'
import Chart3Forecast from './components/Chart3Forecast'
import Chart4Report   from './components/Chart4Report'
import { fetchAnalysis } from './data/api'

export default function App() {
  const today = new Date().toISOString().split('T')[0]

  const [ticker,    setTicker]    = useState('META')
  const [startDate, setStartDate] = useState('2021-01-01')
  const [endDate,   setEndDate]   = useState(today)

  const [analysis, setAnalysis] = useState(null)
  const [forecast, setForecast] = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)

  const handleAnalyze = async (e) => {
    e?.preventDefault()
    setLoading(true)
    setError(null)
    setAnalysis(null)
    setForecast(null)
    try {
      const data = await fetchAnalysis(ticker.trim().toUpperCase(), startDate, endDate)
      setAnalysis(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Auto-run on first load
  useEffect(() => { handleAnalyze() }, [])

  const lastPrice  = analysis?.prices?.[analysis.prices.length - 1]?.close
  const isPositive = analysis ? analysis.total_return >= 0 : null

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">

      {/* ── Header ── */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-white text-sm">
              ML
            </div>
            <div>
              <span className="font-bold text-gray-900 text-lg tracking-tight">MarketLens</span>
              <span className="text-gray-400 text-sm ml-2">· Markets don't move in a vacuum</span>
            </div>
          </div>
          <span className="text-xs text-gray-400 hidden sm:block">
            Data Fetcher · Anomaly Detection · Forecasting · AI Report
          </span>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">

        {/* ── Control Panel ── */}
        <form
          onSubmit={handleAnalyze}
          className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm"
        >
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
            <div>
              <label className="block text-xs text-gray-500 mb-1.5 font-medium">Ticker</label>
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="e.g. META"
                className="w-full bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-gray-900 text-sm font-mono placeholder-gray-400 focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1.5 font-medium">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-gray-900 text-sm focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1.5 font-medium">End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-gray-900 text-sm focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={loading}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />}
              {loading ? 'Analyzing…' : 'Analyze'}
            </button>
            <p className="text-xs text-gray-400">
              Module 1–2 runs on submit · Forecast &amp; Report load on demand
            </p>
          </div>
        </form>

        {/* ── Error ── */}
        {error && (
          <div className="flex items-start gap-2 text-red-600 text-sm bg-red-50 border border-red-200 rounded-xl p-4">
            <AlertCircle size={16} className="shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* ── Results ── */}
        {analysis && (
          <>
            {/* News unavailable alert */}
            {!analysis.news_available && (
              <div className="flex items-start gap-2 text-amber-700 text-sm bg-amber-50 border border-amber-200 rounded-xl p-3">
                <AlertTriangle size={15} className="shrink-0 mt-0.5" />
                <span>
                  <strong>News data unavailable</strong> — FINNHUB_API_KEY is not set.
                  Anomaly detection and sentiment run on price data only; related-event context is missing.
                </span>
              </div>
            )}

            {/* Stock summary header */}
            <div className="flex items-start justify-between flex-wrap gap-4">
              <div>
                <div className="flex items-center gap-3 mb-1">
                  <h1 className="text-4xl font-black tracking-tight text-blue-600">
                    {analysis.ticker}
                  </h1>
                  <span className="text-xs px-2 py-1 rounded-full border border-blue-200 bg-blue-50 text-blue-600">
                    {analysis.anomalies.length} anomal{analysis.anomalies.length !== 1 ? 'ies' : 'y'}
                  </span>
                </div>
                <p className="text-gray-400 text-sm">
                  {analysis.start_date} → {analysis.end_date}
                </p>
              </div>
              {lastPrice && (
                <div className="text-right">
                  <div className="text-3xl font-bold text-gray-900">${lastPrice.toFixed(2)}</div>
                  <div className={`text-sm font-semibold mt-0.5 ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                    {isPositive ? '+' : ''}{analysis.total_return.toFixed(2)}% period return
                  </div>
                </div>
              )}
            </div>

            {/* Chart 1 — Module 1 */}
            <section>
              <p className="text-gray-400 text-xs font-semibold uppercase tracking-widest mb-3">
                Module 1 · Price History · Moving Averages · S&amp;P500 Comparison
              </p>
              <Chart1Price
                prices={analysis.prices}
                spyPrices={analysis.spy_prices}
                ticker={analysis.ticker}
              />
            </section>

            {/* Chart 2 — Module 2 */}
            <section>
              <p className="text-gray-400 text-xs font-semibold uppercase tracking-widest mb-3">
                Module 2 · Anomaly Detection ({analysis.anomalies.length} detected)
              </p>
              <Chart2Anomaly
                prices={analysis.prices}
                anomalies={analysis.anomalies}
                ticker={analysis.ticker}
              />
            </section>

            {/* Chart 3 — Module 3 (on-demand) */}
            <section>
              <p className="text-gray-400 text-xs font-semibold uppercase tracking-widest mb-3">
                Module 3 · Transformer Forecast · Sentiment
              </p>
              <Chart3Forecast
                ticker={analysis.ticker}
                startDate={analysis.start_date}
                endDate={analysis.end_date}
                sentimentScore={analysis.sentiment_score}
                sentimentLabel={analysis.sentiment_label}
                eventCount={analysis.events.length}
                onForecastLoaded={setForecast}
              />
            </section>

            {/* Chart 4 — Module 4 (on-demand) */}
            <section>
              <p className="text-gray-400 text-xs font-semibold uppercase tracking-widest mb-3">
                Module 4 · AI Report · Market Metrics
              </p>
              <Chart4Report
                ticker={analysis.ticker}
                startDate={analysis.start_date}
                endDate={analysis.end_date}
                totalReturn={analysis.total_return}
                anomalyCount={analysis.anomalies.length}
                sentimentScore={analysis.sentiment_score}
                sentimentLabel={analysis.sentiment_label}
                day5Price={forecast?.tft_day5_price ?? forecast?.day5_price ?? null}
              />
            </section>
          </>
        )}

        {/* ── Initial loading state ── */}
        {!analysis && loading && (
          <div className="text-center py-24 text-gray-400">
            <Loader2 size={36} className="mx-auto mb-4 animate-spin text-blue-500" />
            <p className="text-lg font-medium text-gray-500">Loading {ticker}…</p>
            <p className="text-sm mt-2">Fetching prices, detecting anomalies</p>
          </div>
        )}
      </main>

      <footer className="border-t border-gray-200 mt-16 py-6 text-center text-xs text-gray-400">
        MarketLens · For educational purposes only · Not financial advice
      </footer>
    </div>
  )
}
