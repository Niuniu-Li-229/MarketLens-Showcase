"""
Module 1c — SEC EDGAR 8-K Fetcher

Fetches 8-K filings from SEC EDGAR for a given ticker and date range.
Uses `reportDate` (not `filingDate`) as the event date so anomaly linking
in module2 is temporally accurate.

No API key required. Rate limit: 10 req/sec (enforced via sleep).

8-K Item → EventType mapping:
    1.01  Entry into material agreement       → PRODUCT   (partnerships, contracts)
    1.03  Bankruptcy/receivership             → REGULATORY
    2.01  Acquisition/disposition of assets   → PRODUCT   (M&A)
    2.02  Results of operations (earnings)    → EARNINGS
    2.04  Triggering events for obligations   → REGULATORY
    3.01  Delisting notice                    → REGULATORY
    4.01  Auditor changes                     → REGULATORY
    4.02  Non-reliance on financials          → REGULATORY
    5.01  Change in control                   → PERSONNEL
    5.02  Director/officer departure/appoint  → PERSONNEL
    5.03  Amendments to articles              → REGULATORY
    6.01–6.05 Asset-backed securities         → OTHER
    7.01  Regulation FD disclosure            → OTHER
    8.01  Other events                        → OTHER
    9.01  Financial statements / exhibits     → OTHER (skip, no narrative)

Usage:
    from module1c_sec_fetcher import SECFetcher
    fetcher = SECFetcher()
    events = fetcher.fetch_news("META", date(2021, 1, 1), date(2026, 4, 9))

Quick test:
    python module1c_sec_fetcher.py META 2021-01-01 2026-04-09
"""

import logging
import time
from datetime import date
from pathlib import Path

import requests

from models import EventType, MarketEvent
from module1_data_fetcher import DataCache, NewsFetcher

logger = logging.getLogger(__name__)

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_TICKER_URL      = "https://data.sec.gov/submissions/CIK{cik}.json"
_TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"

# Request headers required by SEC fair-access policy
_HEADERS = {
    "User-Agent": "MarketLens research@marketlens.local",
    "Accept-Encoding": "gzip, deflate",
}

REQUEST_SLEEP = 0.12   # stay well under 10 req/sec


# ── 8-K Item → EventType ──────────────────────────────────────────────────────

_ITEM_EVENT_TYPE: dict[str, EventType] = {
    "1.01": EventType.PRODUCT,      # material agreement (partnership, contract)
    "1.02": EventType.PRODUCT,      # termination of material agreement
    "1.03": EventType.REGULATORY,   # bankruptcy / receivership
    "2.01": EventType.PRODUCT,      # acquisition / disposition of assets (M&A)
    "2.02": EventType.EARNINGS,     # results of operations / earnings
    "2.03": EventType.OTHER,        # off-balance-sheet obligations
    "2.04": EventType.REGULATORY,   # triggering event for obligations
    "2.05": EventType.PERSONNEL,    # departure of named executive officers
    "2.06": EventType.OTHER,        # material impairment
    "3.01": EventType.REGULATORY,   # notice of delisting
    "3.02": EventType.REGULATORY,   # unregistered sale of equity
    "3.03": EventType.REGULATORY,   # material modifications to rights of security holders
    "4.01": EventType.REGULATORY,   # changes in registrant's certifying accountant
    "4.02": EventType.REGULATORY,   # non-reliance on financial statements
    "5.01": EventType.PERSONNEL,    # changes in control
    "5.02": EventType.PERSONNEL,    # departure / appointment of directors/officers
    "5.03": EventType.REGULATORY,   # amendments to articles of incorporation
    "5.04": EventType.REGULATORY,   # temporary suspension of trading under employee plans
    "5.05": EventType.REGULATORY,   # amendment to code of ethics
    "5.06": EventType.REGULATORY,   # change in shell company status
    "5.07": EventType.OTHER,        # submission of matters to a vote
    "5.08": EventType.OTHER,        # shareholder director nominations
    "6.01": EventType.OTHER,
    "6.02": EventType.OTHER,
    "6.03": EventType.OTHER,
    "6.04": EventType.OTHER,
    "6.05": EventType.OTHER,
    "7.01": EventType.OTHER,        # Reg FD disclosure
    "8.01": EventType.OTHER,        # other events (catch-all)
    "9.01": EventType.OTHER,        # financial statements / exhibits (skip below)
}

# Items with no meaningful narrative — skip when building description
_SKIP_ITEMS = {"9.01"}


def _item_to_event_type(items_str: str) -> EventType:
    """
    items_str is the raw 'items' field from EDGAR, e.g. '2.02,9.01' or '5.02'.
    Return the highest-priority EventType found; fall back to OTHER.
    """
    priority_order = [
        EventType.EARNINGS, EventType.PERSONNEL, EventType.PRODUCT,
        EventType.REGULATORY, EventType.OTHER,
    ]
    found: set[EventType] = set()
    for item in items_str.replace(";", ",").split(","):
        item = item.strip()
        if item in _ITEM_EVENT_TYPE and item not in _SKIP_ITEMS:
            found.add(_ITEM_EVENT_TYPE[item])
    for et in priority_order:
        if et in found:
            return et
    return EventType.OTHER


def _item_to_title(items_str: str, company_name: str) -> str:
    """Human-readable title derived from Item codes."""
    _ITEM_LABEL: dict[str, str] = {
        "1.01": "Entry into material agreement",
        "1.02": "Termination of material agreement",
        "1.03": "Bankruptcy or receivership",
        "2.01": "Completion of acquisition or disposition",
        "2.02": "Results of operations / earnings release",
        "2.04": "Triggering events for obligations",
        "3.01": "Notice of delisting",
        "4.01": "Change in certifying accountant",
        "4.02": "Non-reliance on prior financial statements",
        "5.01": "Change in control",
        "5.02": "Change in directors or principal officers",
        "5.03": "Amendment to articles of incorporation",
        "5.05": "Amendment to code of ethics",
        "7.01": "Regulation FD disclosure",
        "8.01": "Other material event",
    }
    labels = []
    for item in items_str.replace(";", ",").split(","):
        item = item.strip()
        if item in _ITEM_LABEL:
            labels.append(_ITEM_LABEL[item])
    if labels:
        return f"{company_name}: {labels[0]}"
    return f"{company_name}: SEC 8-K filing"


# ── CIK lookup ────────────────────────────────────────────────────────────────

_cik_cache: dict[str, str] = {}


def _lookup_cik(ticker: str) -> str | None:
    """Return zero-padded 10-digit CIK for a ticker, or None if not found."""
    ticker = ticker.upper()
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    time.sleep(REQUEST_SLEEP)
    try:
        resp = requests.get(_TICKERS_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker:
                cik = str(entry["cik_str"]).zfill(10)
                _cik_cache[ticker] = cik
                return cik
    except Exception as e:
        logger.error("CIK lookup failed for %s: %s", ticker, e)
    return None


# ── Main fetcher ──────────────────────────────────────────────────────────────

class SECFetcher(NewsFetcher):
    """
    Fetches 8-K filings from SEC EDGAR for a given ticker.
    Uses reportDate as the event date (= when the event occurred, not when
    the filing was submitted), so event-anomaly linking stays accurate.
    No API key required.
    """

    def __init__(self):
        self._cache = DataCache()

    def fetch_news(self, ticker: str, start: date, end: date) -> list[MarketEvent]:
        ticker = ticker.upper()

        cik = _lookup_cik(ticker)
        if not cik:
            logger.error("Could not resolve CIK for %s — no SEC events fetched", ticker)
            return []

        logger.info("Fetching SEC 8-K filings for %s (CIK %s) %s~%s", ticker, cik, start, end)

        time.sleep(REQUEST_SLEEP)
        try:
            resp = requests.get(
                _SUBMISSIONS_URL.format(cik=cik),
                headers=_HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("SEC submissions fetch failed for %s: %s", ticker, e)
            return []

        company_name = data.get("name", ticker)
        recent       = data.get("filings", {}).get("recent", {})

        forms        = recent.get("form",        [])
        report_dates = recent.get("reportDate",  [])
        filing_dates = recent.get("filingDate",  [])
        items_list   = recent.get("items",       [])
        accessions   = recent.get("accessionNumber", [])

        events: list[MarketEvent] = []
        seen:   set[str]          = set()

        for i, form in enumerate(forms):
            if form != "8-K":
                continue

            # Use reportDate; fall back to filingDate if missing
            raw_date = report_dates[i] if i < len(report_dates) and report_dates[i] else (
                filing_dates[i] if i < len(filing_dates) else None
            )
            if not raw_date:
                continue
            try:
                event_date = date.fromisoformat(raw_date)
            except ValueError:
                continue

            if not (start <= event_date <= end):
                continue

            items_str  = items_list[i] if i < len(items_list) else ""
            accession  = accessions[i] if i < len(accessions) else ""

            # Skip pure exhibit filings with no narrative content
            if not items_str or all(
                it.strip() in _SKIP_ITEMS
                for it in items_str.replace(";", ",").split(",")
                if it.strip()
            ):
                continue

            event_type  = _item_to_event_type(items_str)
            title       = _item_to_title(items_str, company_name)
            description = (
                f"SEC Form 8-K filed by {company_name} "
                f"(Items: {items_str}). "
                f"Accession: {accession.replace('-', '')}."
            )

            key = f"{event_date}|{items_str}|{accession}"
            if key in seen:
                continue
            seen.add(key)

            events.append(MarketEvent(
                date        = event_date,
                title       = title,
                description = description,
                source      = "SEC EDGAR",
                event_type  = event_type,
            ))

        events.sort(key=lambda e: e.date)

        # Also fetch additional filing pages if company has > 40 recent filings
        # (EDGAR paginates at 40 for older filings via `filings.files`)
        older_events = self._fetch_older_filings(
            cik, company_name, ticker, start, end, data
        )
        if older_events:
            combined = {(e.date, e.title): e for e in events}
            for e in older_events:
                combined.setdefault((e.date, e.title), e)
            events = sorted(combined.values(), key=lambda e: e.date)

        if events:
            self._cache.save_news(ticker, events)

        logger.info("Fetched %d SEC 8-K events for %s", len(events), ticker)
        return events

    def _fetch_older_filings(
        self,
        cik: str,
        company_name: str,
        ticker: str,
        start: date,
        end: date,
        data: dict,
    ) -> list[MarketEvent]:
        """Fetch additional filing pages for companies with deep history."""
        filing_files = data.get("filings", {}).get("files", [])
        if not filing_files:
            return []

        events: list[MarketEvent] = []
        seen:   set[str]          = set()

        for file_info in filing_files:
            name = file_info.get("name", "")
            if not name:
                continue

            time.sleep(REQUEST_SLEEP)
            try:
                url  = f"https://data.sec.gov/submissions/{name}"
                resp = requests.get(url, headers=_HEADERS, timeout=20)
                resp.raise_for_status()
                page = resp.json()
            except Exception as e:
                logger.warning("Failed to fetch older filing page %s: %s", name, e)
                continue

            forms        = page.get("form",         [])
            report_dates = page.get("reportDate",   [])
            filing_dates = page.get("filingDate",   [])
            items_list   = page.get("items",        [])
            accessions   = page.get("accessionNumber", [])

            # Stop fetching pages once all filings are older than start
            page_dates = [
                date.fromisoformat(d) for d in report_dates if d
                if _safe_date(d)
            ]
            if page_dates and max(page_dates) < start:
                break

            for i, form in enumerate(forms):
                if form != "8-K":
                    continue

                raw_date = report_dates[i] if i < len(report_dates) and report_dates[i] else (
                    filing_dates[i] if i < len(filing_dates) else None
                )
                if not raw_date:
                    continue
                try:
                    event_date = date.fromisoformat(raw_date)
                except ValueError:
                    continue

                if not (start <= event_date <= end):
                    continue

                items_str = items_list[i] if i < len(items_list) else ""
                accession = accessions[i] if i < len(accessions) else ""

                if not items_str or all(
                    it.strip() in _SKIP_ITEMS
                    for it in items_str.replace(";", ",").split(",")
                    if it.strip()
                ):
                    continue

                event_type  = _item_to_event_type(items_str)
                title       = _item_to_title(items_str, company_name)
                description = (
                    f"SEC Form 8-K filed by {company_name} "
                    f"(Items: {items_str}). "
                    f"Accession: {accession.replace('-', '')}."
                )

                key = f"{event_date}|{items_str}|{accession}"
                if key in seen:
                    continue
                seen.add(key)

                events.append(MarketEvent(
                    date        = event_date,
                    title       = title,
                    description = description,
                    source      = "SEC EDGAR",
                    event_type  = event_type,
                ))

        return events


def _safe_date(s: str) -> bool:
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    ticker = sys.argv[1] if len(sys.argv) > 1 else "META"
    start  = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2021, 1, 1)
    end    = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date(2026, 4, 9)

    fetcher = SECFetcher()
    events  = fetcher.fetch_news(ticker, start, end)

    print(f"\n{len(events)} SEC 8-K events for {ticker} ({start} ~ {end})\n")
    by_type: dict[str, int] = {}
    for ev in events:
        by_type[ev.event_type.value] = by_type.get(ev.event_type.value, 0) + 1
        print(f"  {ev.date}  [{ev.event_type.value:12s}]  {ev.title}")
    print(f"\nBreakdown: {by_type}")
