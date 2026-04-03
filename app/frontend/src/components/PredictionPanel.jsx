/**
 * PredictionPanel — Displays Module 3 outputs.
 *
 * Left card:  Price forecast (MockForecaster → LSTMForecaster)
 * Right card: Sentiment score + label (MockSentimentAnalyzer → FinBERTAnalyzer)
 *
 * The sentiment gauge maps score ∈ [-1, +1] to a horizontal bar.
 */

import { TrendingUp, TrendingDown, Minus, BarChart2, Brain } from 'lucide-react'

const SENTIMENT = {
  bullish: { color: '#10b981', Icon: TrendingUp,  label: 'Bullish' },
  bearish: { color: '#ef4444', Icon: TrendingDown, label: 'Bearish' },
  neutral: { color: '#6b7280', Icon: Minus,        label: 'Neutral' },
}

function SentimentGauge({ score }) {
  // normalise [-1, +1] → [0, 100]%
  const pct = ((score + 1) / 2) * 100
  const color = score > 0.1 ? '#10b981' : score < -0.1 ? '#ef4444' : '#6b7280'
  return (
    <div className="mt-3">
      <div className="flex justify-between text-xs text-gray-600 mb-1">
        <span>Bearish −1</span>
        <span>Neutral 0</span>
        <span>Bullish +1</span>
      </div>
      <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <p className="text-xs text-gray-600 mt-1.5 text-center">
        score: {score > 0 ? '+' : ''}{score.toFixed(2)}
      </p>
    </div>
  )
}

export default function PredictionPanel({ analysis }) {
  if (!analysis) return null

  const { predicted_price, sentiment_score, sentiment_label, prices, events } = analysis
  const lastClose = prices?.[prices.length - 1]?.close
  const priceChange =
    lastClose != null && predicted_price != null
      ? ((predicted_price - lastClose) / lastClose) * 100
      : null

  const conf = SENTIMENT[sentiment_label] ?? SENTIMENT.neutral
  const { Icon: SentIcon } = conf

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Price forecast */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <BarChart2 size={15} className="text-blue-400" />
          <p className="text-gray-500 text-xs font-semibold uppercase tracking-widest">
            Price Forecast (MockForecaster)
          </p>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-500 text-sm">Last close</span>
            <span className="text-white font-semibold">
              {lastClose != null ? `$${lastClose.toFixed(2)}` : '—'}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-500 text-sm">Predicted next</span>
            <span className="text-white font-bold text-xl">
              {predicted_price != null ? `$${predicted_price.toFixed(2)}` : '—'}
            </span>
          </div>
          {priceChange != null && (
            <div className="flex justify-between items-center pt-3 border-t border-gray-800">
              <span className="text-gray-500 text-sm">Expected move</span>
              <span
                className={`flex items-center gap-1 font-semibold text-sm ${
                  priceChange >= 0 ? 'text-emerald-400' : 'text-red-400'
                }`}
              >
                {priceChange >= 0 ? <TrendingUp size={13} /> : <TrendingDown size={13} />}
                {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
              </span>
            </div>
          )}
        </div>

        <p className="text-gray-700 text-xs mt-4 pt-3 border-t border-gray-800">
          Currently: last close ±1% nudge based on most recent anomaly direction.
          Replace MockForecaster with LSTMForecaster for real predictions.
        </p>
      </div>

      {/* Sentiment */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <Brain size={15} className="text-purple-400" />
          <p className="text-gray-500 text-xs font-semibold uppercase tracking-widest">
            News Sentiment (MockSentimentAnalyzer)
          </p>
        </div>

        <div className="flex items-center gap-3 mb-2">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: `${conf.color}20` }}
          >
            <SentIcon size={20} style={{ color: conf.color }} />
          </div>
          <div>
            <p className="font-bold text-white text-xl">{conf.label}</p>
            <p className="text-gray-500 text-xs">
              {events?.length ?? 0} news event{events?.length !== 1 ? 's' : ''} analysed
            </p>
          </div>
        </div>

        <SentimentGauge score={sentiment_score ?? 0} />

        <p className="text-gray-700 text-xs mt-4 pt-3 border-t border-gray-800">
          Currently: heuristic score (+0.3 per bullish event, −0.3 per bearish).
          Replace MockSentimentAnalyzer with FinBERTAnalyzer for real NLP scores.
        </p>
      </div>
    </div>
  )
}
