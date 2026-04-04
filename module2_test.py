"""
module2_test.py — Unit tests for all Module 2 anomaly detector classes.

Covers:
  - ThresholdDetector
  - ZScoreDetector
  - BollingerDetector
  - IQRDetector
  - VolumeDetector
  - FunnelDetector (mock + full 4-layer)
  - _find_nearby_events (ticker filter, window, ranking)
  - _build_comment
  - _relevance_score
"""

from datetime import date, timedelta
from models import PricePoint, MarketEvent, EventType
from module2_anomaly_detector import (
    ThresholdDetector,
    ZScoreDetector,
    BollingerDetector,
    IQRDetector,
    VolumeDetector,
    FunnelDetector,
    _find_nearby_events,
    _build_comment,
    _relevance_score,
)

# ── helpers ────────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results: list[tuple[str, bool]] = []


def check(name: str, condition: bool) -> None:
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}")
    _results.append((name, condition))


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def make_price(d: date, open_: float, close: float, high: float = None,
               low: float = None, volume: int = 1_000_000) -> PricePoint:
    high = high or max(open_, close) * 1.01
    low  = low  or min(open_, close) * 0.99
    return PricePoint(date=d, open=open_, high=high, low=low,
                      close=close, volume=volume)


def make_flat_series(n: int = 30, base: float = 100.0,
                     start: date = date(2024, 1, 1)) -> list[PricePoint]:
    """Flat price series (0 % daily change) — anomaly = any non-flat day."""
    return [make_price(start + timedelta(days=i), base, base) for i in range(n)]


def make_event(d: date, title: str, etype: EventType = EventType.OTHER,
               description: str = "") -> MarketEvent:
    return MarketEvent(date=d, title=title, description=description,
                       source="TestSource", event_type=etype)


# ── ThresholdDetector ──────────────────────────────────────────────────────────

def test_threshold_detector() -> None:
    section("ThresholdDetector")
    det = ThresholdDetector(threshold=5.0)

    prices = make_flat_series()

    # Exactly at threshold — NOT anomaly (strict >=)
    at_threshold = make_price(date(2024, 3, 1), 100.0, 105.0)
    check("name contains threshold value", "5.0%" in det.name)
    check("exactly 5.0% change is anomaly (>=)", det.is_anomaly(at_threshold, prices))

    # Below threshold
    below = make_price(date(2024, 3, 2), 100.0, 104.9)
    check("4.9% change is NOT anomaly", not det.is_anomaly(below, prices))

    # Large drop
    big_drop = make_price(date(2024, 3, 3), 100.0, 90.0)
    check("-10% change is anomaly", det.is_anomaly(big_drop, prices))

    # Custom threshold
    tight = ThresholdDetector(threshold=1.0)
    small_move = make_price(date(2024, 3, 4), 100.0, 101.5)
    check("1.5% triggers tight threshold", tight.is_anomaly(small_move, prices))


# ── ZScoreDetector ─────────────────────────────────────────────────────────────

def test_zscore_detector() -> None:
    section("ZScoreDetector")
    det = ZScoreDetector(z_threshold=2.0)

    # Build a series with one obvious outlier at the end
    base = date(2024, 1, 1)
    prices = [make_price(base + timedelta(days=i), 100.0, 100.5)  # ~0.5% daily
              for i in range(29)]
    outlier = make_price(base + timedelta(days=29), 100.0, 130.0)  # +30%
    prices.append(outlier)

    check("name contains z_threshold", "2.0σ" in det.name)
    check("+30% outlier is anomaly (z >> 2)", det.is_anomaly(outlier, prices))

    normal = prices[0]
    check("normal day is NOT anomaly", not det.is_anomaly(normal, prices))

    # All-flat series → std=0, should return False (not crash)
    flat = make_flat_series(30)
    check("std=0 series returns False (no crash)", not det.is_anomaly(flat[0], flat))


# ── BollingerDetector ──────────────────────────────────────────────────────────

def test_bollinger_detector() -> None:
    section("BollingerDetector")
    det = BollingerDetector(window=20, k=2.0)

    base = date(2024, 1, 1)
    # 25 stable days around 100, then a spike
    prices = [make_price(base + timedelta(days=i), 99.0, 101.0) for i in range(25)]
    spike   = make_price(base + timedelta(days=25), 99.0, 130.0,
                         high=131.0, low=99.0)
    prices.append(spike)

    check("name reflects window and k", "w=20" in det.name and "k=2.0" in det.name)

    # Spike close (130) is well above upper band of ~101 + 2*σ
    check("spike above upper band is anomaly", det.is_anomaly(spike, prices))

    # Days before window — not enough history
    early = prices[5]
    check("day before window (idx<19) is NOT anomaly", not det.is_anomaly(early, prices))

    # Normal day inside bands
    normal = prices[22]
    check("normal day inside bands is NOT anomaly", not det.is_anomaly(normal, prices))


# ── IQRDetector ────────────────────────────────────────────────────────────────

def test_iqr_detector() -> None:
    section("IQRDetector")
    det = IQRDetector()

    base = date(2024, 1, 1)
    # 29 flat days + 1 extreme outlier
    prices = [make_price(base + timedelta(days=i), 100.0, 100.2) for i in range(29)]
    outlier = make_price(base + timedelta(days=29), 100.0, 150.0)
    prices.append(outlier)

    check("name is IQR(1.5)", det.name == "IQR(1.5)")
    check("+50% outlier is anomaly", det.is_anomaly(outlier, prices))
    check("flat day is NOT anomaly", not det.is_anomaly(prices[0], prices))


# ── VolumeDetector ─────────────────────────────────────────────────────────────

def test_volume_detector() -> None:
    section("VolumeDetector")
    det = VolumeDetector(window=20, multiplier=2.0)

    base = date(2024, 1, 1)
    # 25 days of normal volume (1M), then a volume spike (10M)
    prices = [make_price(base + timedelta(days=i), 100.0, 100.0,
                         volume=1_000_000) for i in range(25)]
    vol_spike = make_price(base + timedelta(days=25), 100.0, 100.0,
                           volume=10_000_000)
    prices.append(vol_spike)

    check("name reflects multiplier", "2.0x" in det.name)
    check("10M volume vs 1M avg (>2x) is anomaly", det.is_anomaly(vol_spike, prices))

    normal = prices[24]
    check("normal volume day is NOT anomaly", not det.is_anomaly(normal, prices))

    # Not enough history (idx < window)
    early = prices[5]
    check("early day (idx < window) is NOT anomaly", not det.is_anomaly(early, prices))


# ── FunnelDetector — mock (ThresholdDetector only) ────────────────────────────

def test_funnel_mock() -> None:
    section("FunnelDetector — mock (ThresholdDetector, min_triggers=1)")
    funnel = FunnelDetector([ThresholdDetector(threshold=5.0)], min_triggers=1)

    base = date(2024, 1, 1)
    prices = [make_price(base + timedelta(days=i), 100.0, 100.5) for i in range(30)]
    # Insert one anomalous day
    anomaly_date = base + timedelta(days=15)
    prices[15] = make_price(anomaly_date, 100.0, 93.0)  # -7%

    events: list[MarketEvent] = []
    results = funnel.detect(prices, events, ticker=None)

    check("detects exactly 1 anomaly", len(results) == 1)
    check("anomaly is on correct date", results[0].date == anomaly_date)
    check("percent_change is negative", results[0].percent_change < 0)
    check("comment mentions 'dropped'", "dropped" in results[0].comment)


# ── FunnelDetector — full 4-layer funnel ──────────────────────────────────────

def test_funnel_full() -> None:
    section("FunnelDetector — full 4-layer (min_triggers=2)")
    funnel = FunnelDetector(
        [ZScoreDetector(), BollingerDetector(), IQRDetector(), VolumeDetector()],
        min_triggers=2,
    )

    base = date(2024, 1, 1)
    # 29 quiet days, then 1 day with both price AND volume spike
    prices = [make_price(base + timedelta(days=i), 100.0, 100.2,
                         volume=1_000_000) for i in range(29)]
    # +30% price surge + 10× volume → should trigger ZScore + IQR + Volume
    big_day = make_price(base + timedelta(days=29), 100.0, 130.0,
                         high=131.0, low=100.0, volume=10_000_000)
    prices.append(big_day)

    results = funnel.detect(prices, [], ticker="NVDA")
    check("at least 1 anomaly detected with ≥2 triggers", len(results) >= 1)
    if results:
        check("anomaly date is the big day", results[-1].date == big_day.date)


# ── FunnelDetector — ticker keyword filter ────────────────────────────────────

def test_funnel_ticker_filter() -> None:
    section("FunnelDetector — ticker keyword filter in detect()")
    funnel = FunnelDetector([ThresholdDetector(threshold=5.0)], min_triggers=1)

    base = date(2024, 1, 1)
    anomaly_date = base + timedelta(days=10)
    prices = [make_price(base + timedelta(days=i), 100.0, 100.0) for i in range(20)]
    prices[10] = make_price(anomaly_date, 100.0, 108.0)  # +8%

    nvidia_event = make_event(anomaly_date, "NVIDIA beats earnings",
                              EventType.EARNINGS, "nvidia h100 revenue surges")
    apple_event  = make_event(anomaly_date, "Apple recall",
                              EventType.PRODUCT, "iphone battery issue")
    unrelated    = make_event(anomaly_date, "Fed rate decision", EventType.MACRO)

    results = funnel.detect(prices, [nvidia_event, apple_event, unrelated], ticker="NVDA")
    check("1 anomaly detected", len(results) == 1)
    related_titles = [e.title for e in results[0].related_events]
    check("NVIDIA event included", "NVIDIA beats earnings" in related_titles)
    check("Apple event excluded (different ticker)", "Apple recall" not in related_titles)


# ── _find_nearby_events ────────────────────────────────────────────────────────

def test_find_nearby_events() -> None:
    section("_find_nearby_events")
    anchor = date(2024, 6, 10)

    e_pre3  = make_event(anchor - timedelta(days=3), "Pre-3 earnings", EventType.EARNINGS)
    e_pre1  = make_event(anchor - timedelta(days=1), "Pre-1 product",  EventType.PRODUCT)
    e_same  = make_event(anchor,                     "Same-day macro", EventType.MACRO)
    e_post1 = make_event(anchor + timedelta(days=1), "Post-1 analyst", EventType.ANALYST)
    e_post3 = make_event(anchor + timedelta(days=3), "Post-3 other",   EventType.OTHER)

    all_events = [e_pre3, e_pre1, e_same, e_post1, e_post3]

    # Default window: pre_days=3, post_days=1
    nearby = _find_nearby_events(anchor, all_events, pre_days=3, post_days=1)
    titles = [e.title for e in nearby]
    check("pre-3 included", "Pre-3 earnings" in titles)
    check("post-1 included", "Post-1 analyst" in titles)
    check("post-3 excluded", "Post-3 other" not in titles)

    # Sorting: same-day EARNINGS (score=1.0) before MACRO (score=6.0+)
    nearby_sorted = _find_nearby_events(anchor, all_events, pre_days=3, post_days=1)
    check("EARNINGS before MACRO in sort",
          nearby_sorted.index(e_pre3) < next(i for i, e in enumerate(nearby_sorted)
                                              if e.event_type == EventType.MACRO))

    # top_n cap
    many = [make_event(anchor, f"Event {i}", EventType.OTHER) for i in range(20)]
    capped = _find_nearby_events(anchor, many, pre_days=3, post_days=1, top_n=5)
    check("top_n=5 returns at most 5 events", len(capped) <= 5)

    # Ticker filter: only nvidia events survive
    nv_event = make_event(anchor, "NVIDIA H100 update", EventType.PRODUCT,
                          "nvidia blackwell chips shipping")
    other_ev = make_event(anchor, "Boeing delay", EventType.OTHER, "supply chain issues")
    filtered = _find_nearby_events(anchor, [nv_event, other_ev],
                                   pre_days=1, post_days=1, ticker="NVDA")
    check("NVDA filter keeps nvidia event", nv_event in filtered)
    check("NVDA filter drops unrelated event", other_ev not in filtered)


# ── _relevance_score ───────────────────────────────────────────────────────────

def test_relevance_score() -> None:
    section("_relevance_score")
    anchor = date(2024, 6, 10)

    earnings_same  = make_event(anchor,                     "E0",  EventType.EARNINGS)
    earnings_pre1  = make_event(anchor - timedelta(days=1), "E-1", EventType.EARNINGS)
    earnings_pre3  = make_event(anchor - timedelta(days=3), "E-3", EventType.EARNINGS)
    other_post1    = make_event(anchor + timedelta(days=1), "O+1", EventType.OTHER)

    s_e0  = _relevance_score(earnings_same,  anchor)
    s_e1  = _relevance_score(earnings_pre1,  anchor)
    s_e3  = _relevance_score(earnings_pre3,  anchor)
    s_o1  = _relevance_score(other_post1,    anchor)

    check("EARNINGS day-0 score == 1.0",       abs(s_e0 - 1.0) < 1e-9)
    check("EARNINGS day-0 < EARNINGS day-1",   s_e0 < s_e1)
    check("EARNINGS day-1 < EARNINGS day-3",   s_e1 < s_e3)
    check("pre-event penalty < post-event penalty (same distance)",
          _relevance_score(earnings_pre1, anchor) <
          _relevance_score(make_event(anchor + timedelta(days=1), "X",
                                     EventType.EARNINGS), anchor))
    check("OTHER post-1 score == 7.8",         abs(s_o1 - 7.8) < 1e-9)


# ── _build_comment ─────────────────────────────────────────────────────────────

def test_build_comment() -> None:
    section("_build_comment")
    price_up   = make_price(date(2024, 6, 10), 100.0, 110.0)
    price_down = make_price(date(2024, 6, 11), 100.0,  90.0)
    det        = ThresholdDetector(5.0)
    event      = make_event(date(2024, 6, 10), "Fed rate cut", EventType.MACRO)

    comment_up = _build_comment(price_up,   [det], [event])
    comment_dn = _build_comment(price_down, [det], [])

    check("surge comment says 'surged'",           "surged"  in comment_up)
    check("drop  comment says 'dropped'",          "dropped" in comment_dn)
    check("comment includes event title",          "Fed rate cut" in comment_up)
    check("no-event comment says 'no related news'", "no related news" in comment_dn)
    check("percent appears in comment",            "10.00%" in comment_up)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 60)
    print("  Module 2 — Anomaly Detector Test Suite")
    print("═" * 60)

    test_threshold_detector()
    test_zscore_detector()
    test_bollinger_detector()
    test_iqr_detector()
    test_volume_detector()
    test_funnel_mock()
    test_funnel_full()
    test_funnel_ticker_filter()
    test_find_nearby_events()
    test_relevance_score()
    test_build_comment()

    passed = sum(1 for _, ok in _results if ok)
    failed = len(_results) - passed
    print(f"\n{'═' * 60}")
    print(f"  Results: {passed}/{len(_results)} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED:")
        for name, ok in _results:
            if not ok:
                print(f"    ✗ {name}")
    else:
        print("  — all good!")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
