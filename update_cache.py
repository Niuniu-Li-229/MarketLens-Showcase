"""
update_cache.py — Refresh cached price & news data for tracked tickers.

Usage:
    python update_cache.py                          # all 6 tickers, last 1 year
    python update_cache.py NVDA TSLA                # specific tickers only
    python update_cache.py --start 2025-09-01       # custom start date
    python update_cache.py --days 90                # last 90 days
    python update_cache.py --prices-only            # skip news (faster)
    python update_cache.py --news-only              # skip prices
"""

import argparse
import logging
import time
from datetime import date, timedelta

from module1_data_fetcher import YFinancePriceFetcher, FinnhubNewsFetcher

DEFAULT_TICKERS = ["NVDA", "META", "GOOGL", "AAPL", "AMZN", "TSLA"]
DEFAULT_LOOKBACK_DAYS = 365


def main():
    parser = argparse.ArgumentParser(description="Update cached price & news data.")
    parser.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS,
                        help=f"Tickers to update (default: {' '.join(DEFAULT_TICKERS)})")
    parser.add_argument("--start", type=date.fromisoformat, default=None,
                        help="Start date (YYYY-MM-DD). Overrides --days.")
    parser.add_argument("--end", type=date.fromisoformat, default=None,
                        help="End date (YYYY-MM-DD, default: today)")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help=f"Lookback days from end date (default: {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--prices-only", action="store_true", help="Only update prices")
    parser.add_argument("--news-only", action="store_true", help="Only update news")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    end = args.end or date.today()
    start = args.start or (end - timedelta(days=args.days))
    tickers = [t.upper() for t in args.tickers]

    print(f"Tickers: {', '.join(tickers)}")
    print(f"Period:  {start} ~ {end}")
    print(f"Mode:    {'prices only' if args.prices_only else 'news only' if args.news_only else 'prices + news'}")
    print()

    price_fetcher = None if args.news_only else YFinancePriceFetcher()
    news_fetcher = None if args.prices_only else FinnhubNewsFetcher()

    t0 = time.time()
    for ticker in tickers:
        print(f"{'=' * 50}  {ticker}  {'=' * 50}")

        if price_fetcher:
            prices = price_fetcher.fetch_prices(ticker, start, end)
            print(f"  Prices: {len(prices)} trading days", end="")
            if prices:
                print(f"  ({prices[0].date} ~ {prices[-1].date})")
            else:
                print()

        if news_fetcher:
            events = news_fetcher.fetch_news(ticker, start, end)
            print(f"  News:   {len(events)} events")

        print()

    elapsed = time.time() - t0
    print(f"Done — {len(tickers)} ticker(s) updated in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
