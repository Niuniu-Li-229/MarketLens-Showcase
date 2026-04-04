# Module 1 — Pending Changes for Other Files

Module 1 has undergone significant refactoring. This document tracks
what other files need to change to fully adapt.

---

## main_pipeline.py

### Already applied
- [x] Import rename: `MockDataFetcher` → `MockPriceFetcher` + `MockNewsFetcher`
- [x] `build_pipeline()`: single `fetcher` → `price_fetcher` + `news_fetcher`
- [x] `run_pipeline()`: split into two separate fetch calls

### Still needed
- [ ] **Switch to real fetchers**: change two import lines at top:
  ```python
  from module1_data_fetcher import YFinancePriceFetcher  as PriceFetcher
  from module1_data_fetcher import FinnhubNewsFetcher    as NewsFetcher
  ```
- [ ] **FinnhubNewsFetcher init requires API key**: `build_pipeline()` currently calls `NewsFetcher()` with no args — works for Mock, but `FinnhubNewsFetcher()` reads `FINNHUB_API_KEY` env var. Need to either ensure env var is set, or pass key explicitly
- [ ] **Multi-ticker support**: `run_pipeline()` currently accepts a single `ticker`. Module 1 now has data cached for 6 tickers (NVDA, META, GOOGL, AAPL, AMZN, TSLA). Consider adding a loop or batch mode to `run_pipeline` to process multiple tickers
- [ ] **Empty price list guard**: line 51 does `prices[-1].close` — will `IndexError` if `fetch_prices` returns `[]`. Already has `if prices` guard for `total_return`, but the rest of the pipeline (Module 2/3) would still receive an empty list. Consider early return or error message

---

## models.py

### No breaking changes needed — but potential enhancements:
- [ ] **`EventType` expansion**: current keywords classify ~70-75% of news as `OTHER`. Consider adding new types: `OPINION`, `MARKET_SUMMARY`, `STOCK_MOVEMENT` to reduce OTHER bucket. This affects `_classify_event()` in module1 and `MockSentimentAnalyzer` scoring weights in module3
- [ ] **`MarketEvent.url` field**: Finnhub returns article URLs (`article["url"]`), currently discarded. Adding an optional `url: str = ""` field would preserve source links for report generation (Module 4) and debugging. Non-breaking change
- [ ] **`MarketEvent.ticker` field**: current `MarketEvent` has no ticker field. With 6 companies sharing the same cache format, adding an optional `ticker` field could help when events are loaded from cache and mixed across tickers

---

## module2_anomaly_detector.py

### No code changes required for current functionality
- [x] Interface compatible — receives `list[PricePoint]` and `list[MarketEvent]`, unchanged

### Adaptation considerations
- [ ] **Data volume**: with 1 year of real data (~253 trading days per ticker), the ThresholdDetector's 5% threshold may flag very few anomalies for stable stocks (AAPL, GOOGL). May need to lower threshold or prioritize implementing ZScoreDetector / BollingerDetector which adapt to the stock's own volatility
- [ ] **FunnelDetector event matching**: `_find_nearby_events()` uses `window_days=2`. With Finnhub news density (~40-60 events/day for NVDA), each anomaly could link to 100+ events. Consider adding a relevance cap or filtering by EventType

---

## module3_sentiment_lstm.py

### No code changes required for current functionality
- [x] Interface compatible — receives `list[MarketEvent]`, unchanged

### Adaptation considerations
- [ ] **MockSentimentAnalyzer scoring overflow**: scoring is `+0.3 per bullish, -0.3 per bearish`, clamped to [-1, 1]. With real data (~10,000+ events per ticker), the raw score will immediately saturate to +1.0 or -1.0 regardless of actual sentiment. Need to normalize by event count (e.g. `score / len(events)`) or switch to per-event averaging
- [ ] **FinBERTAnalyzer input**: the stub uses `e.description` for NLP. Now that empty descriptions are filtered out, all remaining events have text. The `description` field is capped at 500 chars — within FinBERT's typical input range
- [ ] **LSTMForecaster training data**: with 253 days of cached OHLCV per ticker, there is now enough data to train/fine-tune an LSTM. `_build_features()` can read directly from `DataCache.load_prices()`

---

## module4_claude_report.py

### No code changes required for current functionality
- [x] Interface compatible — receives `AnalysisResult`, unchanged

### Adaptation considerations
- [ ] **Data source transparency**: `StandardReportBuilder` prompt does not mention where data comes from. Could add a line like "Data sources: yfinance (prices), Finnhub (news)" for report credibility
- [ ] **Event volume in prompt**: `_anomalies()` method includes ALL `related_events` for each anomaly. With real Finnhub data, a single anomaly day could have 100+ related events, creating an extremely long prompt. Consider capping to top-N events per anomaly (e.g. 5) or filtering by EventType relevance
- [ ] **Multi-ticker reports**: `AnalysisResult` is single-ticker. If `main_pipeline.py` adds batch mode, Module 4 may need a comparative report builder

---

## New files added by Module 1

| File | Purpose | Tracked by git? |
|------|---------|-----------------|
| `update_cache.py` | CLI script to refresh cached data for all tickers | Should be committed |
| `data_cache/*.csv` | Cached price & news CSVs (~74K news rows, ~1.5K price rows) | Add to `.gitignore` — regenerated by `update_cache.py` |

- [ ] **Create `.gitignore`**: add `data_cache/` to prevent committing large CSV files to repo

---

## Module 1 internal changes log

| Date       | Change |
|------------|--------|
| 2026-04-03 | Split `DataFetcher` → `PriceFetcher` + `NewsFetcher` (two ABCs) |
| 2026-04-03 | Added `YFinancePriceFetcher`, `FinnhubNewsFetcher` real implementations |
| 2026-04-03 | Added `_classify_event()` keyword-based EventType mapper |
| 2026-04-03 | Added CSV cache layer: `DataCache` class, `data_cache/` directory |
| 2026-04-03 | `MockPriceFetcher` / `MockNewsFetcher` now read from cache when available |
| 2026-04-03 | Added adaptive recursive bisect for Finnhub 250-cap windows |
| 2026-04-03 | Added API rate limiting (`time.sleep(1.1s)` per request) |
| 2026-04-03 | Added `update_cache.py` CLI script for batch cache refresh |
| 2026-04-03 | Expanded from NVDA-only to 6 tickers (NVDA, META, GOOGL, AAPL, AMZN, TSLA) |
| 2026-04-03 | Added data cleaning: filter out empty-description news at fetch time + cleaned existing cache |
