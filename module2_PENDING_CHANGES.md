# Module 3 & 4 — Pending Changes

Triggered by keyword + EventType expansion in modules 1 and 2 (2026-04-03).
No breaking changes — both modules remain interface-compatible. Updates below
improve correctness and output quality.

---

## module3_sentiment_lstm.py

### Required fix — MockSentimentAnalyzer

A new `EventType.PERSONNEL` has been added to `models.py`.
`MockSentimentAnalyzer.analyze()` currently hard-codes two sets:

```python
bullish = {EventType.EARNINGS, EventType.ANALYST, EventType.PRODUCT}
bearish = {EventType.REGULATORY, EventType.MACRO}
```

`PERSONNEL` is not in either set, so it silently scores as **0.0 (neutral)**.
Executive changes (CEO resignation, CFO appointment) have directional signal
and should not be neutral by default.

**Action needed** — add `PERSONNEL` to one of the sets, or create a third
`neutral` set with explicit 0.0 weight. Suggested assignment:

```python
# Executive departures tend to be negative surprises; appointments are mixed.
# Treat as mildly bearish at the mock level until FinBERT replaces this.
bearish = {EventType.REGULATORY, EventType.MACRO, EventType.PERSONNEL}
```

If the team prefers neutral, just add it explicitly so the intent is clear:

```python
neutral = {EventType.OTHER, EventType.PERSONNEL}
```

### No change needed — FinBERTAnalyzer

FinBERT reads raw text (`e.description`) and does not use `EventType`.
The new `PERSONNEL` type will be classified correctly by the model once it
is integrated.

### No change needed — LSTMForecaster

`_build_features()` uses `is_anomaly` flag and `sentiment_score` as inputs —
neither depends on `EventType`.

---

## module4_claude_report.py

### No changes required

`StandardReportBuilder._anomalies()` emits event type as a string label:

```python
f"     [{e.event_type.value}] {e.title} ({e.source})"
```

Claude will now receive `[PERSONNEL]` instead of `[OTHER]` for executive
change articles. This is strictly better — more specific labels improve the
quality of the generated report with no code change required.

### Opportunistic improvement (not blocking)

`related_events` per anomaly is now capped at **top 10** (previously
unbounded — some anomaly days had 800+ events in the prompt). Events are
pre-sorted by a **composite relevance score** (see below) rather than
EventType alone.

If the prompt still feels long, the cap can be tightened by changing
`top_n` in `module2_anomaly_detector._find_nearby_events()`.

---

## Summary table

| Module | File | Action | Priority |
|--------|------|--------|----------|
| 3 | `module3_sentiment_lstm.py` | Add `PERSONNEL` to `bullish` or `bearish` set in `MockSentimentAnalyzer` | **Required** |
| 3 | `module3_sentiment_lstm.py` | `FinBERTAnalyzer` — no change needed | — |
| 3 | `module3_sentiment_lstm.py` | `LSTMForecaster` — no change needed | — |
| 4 | `module4_claude_report.py` | No changes required | — |
| 4 | `module4_claude_report.py` | Optionally lower `top_n` cap in module2 if prompt is too long | Optional |
