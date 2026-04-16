"""
Module 3 — Sentiment Analysis + Price Forecasting

        SentimentAnalyzer (abstract)
        ├── MockSentimentAnalyzer    ← heuristic mock
        └── FinBERTAnalyzer          ← real: ProsusAI/finbert

        PriceForecaster (abstract)
        ├── MockForecaster           ← heuristic mock
        ├── LSTMForecaster           ← baseline: 2 features, seq=10
        ├── TransformerForecaster    ← improved: 8 features, seq=20
        └── TFTForecaster            ← Temporal Fusion Transformer

pip install torch transformers scikit-learn pytorch-forecasting pytorch-lightning
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from models import MarketEvent, AnomalyPoint, PricePoint


# ── Shared result dataclass ───────────────────────────────────────────────────

@dataclass
class ForecastResult:
    model_name:   str
    day5_price:   float
    forecast_5d:  list[float]
    actual:       np.ndarray = field(default_factory=lambda: np.array([]))
    predicted:    np.ndarray = field(default_factory=lambda: np.array([]))
    test_dates:   list       = field(default_factory=list)
    dir_accuracy: float      = 0.0
    mae:          float      = 0.0
    sector_name:  str        = "Unknown"


# ── Sector mapping ────────────────────────────────────────────────────────────

SECTOR_MAP = {
    "NVDA": 0, "AAPL": 0, "MSFT": 0, "GOOGL": 0, "META": 0,
    "AMD":  0, "INTC": 0, "CRM":  0, "ORCL":  0, "ADBE": 0,
    "JPM":  1, "BAC":  1, "GS":   1, "MS":    1, "WFC":  1,
    "C":    1, "AXP":  1, "BLK":  1, "SCHW":  1, "USB":  1,
    "XOM":  2, "CVX":  2, "COP":  2, "SLB":   2, "EOG":  2,
    "JNJ":  3, "PFE":  3, "UNH":  3, "ABT":   3, "MRK":  3,
    "LLY":  3, "BMY":  3, "AMGN": 3, "GILD":  3, "CVS":  3,
    "AMZN": 4, "WMT":  4, "HD":   4, "MCD":   4, "NKE":  4,
    "SBUX": 4, "TGT":  4, "COST": 4, "LOW":   4, "TJX":  4,
}
SECTOR_NAMES = {
    0: "Technology", 1: "Financials", 2: "Energy",
    3: "Healthcare", 4: "Consumer",  -1: "Unknown",
}


# ── Abstract bases ────────────────────────────────────────────────────────────

class SentimentAnalyzer(ABC):
    @abstractmethod
    def analyze(self, events: list[MarketEvent]) -> tuple[float, str]: ...

class PriceForecaster(ABC):
    @abstractmethod
    def predict(
        self,
        prices:    list[PricePoint],
        anomalies: list[AnomalyPoint],
        ticker:    str = "UNKNOWN",
    ) -> ForecastResult: ...


# ── Mock implementations ──────────────────────────────────────────────────────

class MockSentimentAnalyzer(SentimentAnalyzer):
    def analyze(self, events: list[MarketEvent]) -> tuple[float, str]:
        from models import EventType
        if not events:
            return 0.0, "neutral"
        bullish = {EventType.EARNINGS, EventType.ANALYST, EventType.PRODUCT}
        bearish = {EventType.REGULATORY, EventType.MACRO}
        score = sum(
             0.3 if e.event_type in bullish else
            -0.3 if e.event_type in bearish else 0.0
            for e in events
        )
        score = max(-1.0, min(1.0, score))
        label = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
        return round(score, 2), label


class MockForecaster(PriceForecaster):
    def predict(self, prices, anomalies, ticker="UNKNOWN") -> ForecastResult:
        if not prices:
            return ForecastResult("Mock", 0.0, [0.0] * 5)
        last  = prices[-1].close
        nudge = 0.01 if (anomalies and anomalies[-1].is_gain()) else -0.01
        d1    = round(last * (1 + nudge), 2)
        forecast = [round(d1 * (1 + nudge * i * 0.5), 2) for i in range(5)]
        return ForecastResult("Mock", float(forecast[4]), forecast)


# ── Real Sentiment: FinBERT ───────────────────────────────────────────────────

class FinBERTAnalyzer(SentimentAnalyzer):
    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self):
        from transformers import pipeline as hf_pipeline
        print("[Module 3] Loading FinBERT...")
        self._pipe = hf_pipeline(
            "text-classification",
            model=self.MODEL_NAME,
            truncation=True,
            max_length=512,
        )
        print("[Module 3] FinBERT ready.")

    MAX_EVENTS = 150  # default cap for live API requests

    def analyze(self, events: list[MarketEvent],
                max_events: int | None = MAX_EVENTS) -> tuple[float, str]:
        """
        Score events with FinBERT.
        max_events: stride-sample cap (None = no cap, used by warm_up).
        """
        if not events:
            return 0.0, "neutral"
        # Stride-sample to stay representative across the full date range
        if max_events is not None and len(events) > max_events:
            step   = len(events) / max_events
            events = [events[int(i * step)] for i in range(max_events)]
        texts   = [(e.description or e.title) for e in events]
        results = self._pipe(texts, batch_size=16)
        scores  = [
             r["score"] if r["label"] == "positive" else
            -r["score"] if r["label"] == "negative" else 0.0
            for r in results
        ]
        avg   = float(np.clip(np.mean(scores), -1.0, 1.0))
        label = "bullish" if avg > 0.1 else "bearish" if avg < -0.1 else "neutral"
        print(f"[Module 3] Sentiment: {label} ({avg:+.3f}) over {len(events)} events.")
        return round(avg, 3), label


# ── Shared feature builders ───────────────────────────────────────────────────

def _build_base_features(prices, anomalies) -> np.ndarray:
    """2 features: close + anomaly flag. Baseline LSTM."""
    anom_dates = {a.date for a in anomalies}
    return np.array([
        [p.close, 1.0 if p.date in anom_dates else 0.0]
        for p in prices
    ], dtype=np.float64)


def _build_daily_sentiment(prices, events, window_days=7) -> np.ndarray:
    """
    第9个feature：每个交易日取前window_days天内的新闻，用FinBERT批量打分取均值。
    没有新闻的日期填0.0（neutral）。
    首次运行会下载FinBERT模型（~500MB），之后从本地缓存读取。
    如果transformers未安装或模型加载失败，自动降级为规则打分。
    """
    from collections import defaultdict
    from datetime import timedelta

    # 按日期建索引
    events_by_date = defaultdict(list)
    for e in events:
        events_by_date[e.date].append(e)

    # 收集每天的文本列表
    daily_texts: list[list[str]] = []
    for p in prices:
        texts = []
        for d in range(window_days):
            day = p.date - timedelta(days=d)
            for e in events_by_date.get(day, []):
                text = (e.description or e.title or "").strip()
                if text:
                    texts.append(text[:512])  # FinBERT最大512 tokens
        daily_texts.append(texts)

    # 收集所有需要打分的文本，记录位置
    all_texts: list[str] = []
    text_index: list[tuple[int, int]] = []  # (day_idx, text_idx_in_day)
    for day_idx, texts in enumerate(daily_texts):
        for t_idx, text in enumerate(texts):
            all_texts.append(text)
            text_index.append((day_idx, t_idx))

    sentiment_arr = np.zeros(len(prices), dtype=np.float32)

    if not all_texts:
        return sentiment_arr

    # FinBERT批量打分
    try:
        from transformers import pipeline as hf_pipeline
        print(f"[Sentiment] Loading FinBERT for {len(all_texts)} texts across "
              f"{len(prices)} days...")
        pipe = hf_pipeline(
            "text-classification",
            model    = "ProsusAI/finbert",
            truncation = True,
            max_length = 512,
            batch_size = 32,   # 批量处理，速度比逐条快很多
        )
        results = pipe(all_texts)

        # 把分数按天汇总
        day_scores: dict[int, list[float]] = defaultdict(list)
        for (day_idx, _), r in zip(text_index, results):
            score = (
                 r["score"] if r["label"] == "positive" else
                -r["score"] if r["label"] == "negative" else 0.0
            )
            day_scores[day_idx].append(score)

        for day_idx, scores in day_scores.items():
            sentiment_arr[day_idx] = float(np.clip(np.mean(scores), -1.0, 1.0))

        print(f"[Sentiment] FinBERT done. "
              f"Non-zero days: {np.count_nonzero(sentiment_arr)}/{len(prices)}")

    except Exception as e:
        # 降级：用规则打分
        print(f"[Sentiment] FinBERT failed ({e}), falling back to rule-based scoring.")
        from models import EventType
        bullish = {EventType.EARNINGS, EventType.ANALYST, EventType.PRODUCT}
        bearish = {EventType.REGULATORY, EventType.MACRO}
        for i, p in enumerate(prices):
            scores = []
            for d in range(window_days):
                day = p.date - timedelta(days=d)
                for e in events_by_date.get(day, []):
                    if e.event_type in bullish:
                        scores.append(0.3)
                    elif e.event_type in bearish:
                        scores.append(-0.3)
                    else:
                        scores.append(0.0)
            if scores:
                sentiment_arr[i] = float(np.clip(np.mean(scores), -1.0, 1.0))

    return sentiment_arr


def _build_rich_features(prices, anomalies, ticker, events=None) -> np.ndarray:
    """
    8 features (events=None): close + returns + volume + anomaly flag
                               + RSI + MACD + Bollinger position + sector encoding.
    9 features (events provided): above + daily rolling sentiment score.
    Same technical indicators as module2 detectors.
    """
    closes  = np.array([p.close        for p in prices], dtype=np.float32)
    volumes = np.array([p.volume / 1e8 for p in prices], dtype=np.float32)

    # Close-to-close returns (consistent with module2 ZScoreDetector)
    returns = np.zeros_like(closes)
    for i in range(1, len(closes)):
        returns[i] = (closes[i] - closes[i-1]) / (closes[i-1] + 1e-9)

    anom_dates  = {a.date for a in anomalies}
    flags       = np.array([1.0 if p.date in anom_dates else 0.0
                            for p in prices], dtype=np.float32)
    sector_id   = SECTOR_MAP.get(ticker.upper(), -1)
    sector_feat = np.full(len(prices),
                          (sector_id + 1) / (len(SECTOR_NAMES) - 1),
                          dtype=np.float32)

    def rolling_rsi(arr, w=14):
        rsi = np.full_like(arr, 0.5)
        for i in range(w, len(arr)):
            d    = np.diff(arr[i-w:i+1])
            gain = d[d > 0].mean() if (d > 0).any() else 0.0
            loss = -d[d < 0].mean() if (d < 0).any() else 1e-9
            rsi[i] = (100 - 100 / (1 + gain / loss)) / 100.0
        return rsi

    def rolling_macd(arr):
        ema = lambda n: np.array(
            [arr[:i+1][-n:].mean() if i >= n-1 else arr[i]
             for i in range(len(arr))], dtype=np.float32)
        return (ema(12) - ema(26)) / (arr + 1e-9)

    def bb_position(arr, w=20):
        pos = np.zeros_like(arr)
        for i in range(w, len(arr)):
            win = arr[i-w:i]
            std = win.std()
            if std > 0:
                pos[i] = (arr[i] - win.mean()) / std
        return pos

    base = [
        closes,
        returns,
        volumes,
        flags,
        rolling_rsi(closes).astype(np.float32),
        rolling_macd(closes).astype(np.float32),
        bb_position(closes).astype(np.float32),
        sector_feat,
    ]

    # 第9个feature：有新闻时加入每日情绪分数
    if events is not None:
        daily_sentiment = _build_daily_sentiment(prices, events)
        base.append(daily_sentiment)

    return np.stack(base, axis=1)


def _make_sequences(features: np.ndarray, seq_len: int, split: float = 0.85):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i, 0])
    X  = np.array(X, dtype=np.float32)
    y  = np.array(y, dtype=np.float32)
    sp = int(len(X) * split)
    return X[:sp], X[sp:], y[:sp], y[sp:], scaler


def _inverse_close(arr, scaler, n_features) -> np.ndarray:
    pad = np.zeros((len(arr), n_features))
    pad[:, 0] = arr
    return scaler.inverse_transform(pad)[:, 0]


def _roll_forecast(model, scaled, seq_len, scaler, n, days=5) -> list[float]:
    import torch
    seq = scaled[-seq_len:].copy()
    out = []
    model.eval()
    for _ in range(days):
        inp = torch.FloatTensor(seq).unsqueeze(0)
        with torch.no_grad():
            p = model(inp).item()
        pad = np.zeros((1, n)); pad[0, 0] = p
        out.append(round(float(scaler.inverse_transform(pad)[0, 0]), 2))
        row = seq[-1].copy(); row[0] = p
        seq = np.vstack([seq[1:], row])
    return out


def _print_forecast(prices, last_close, last_date, model_name):
    total = (prices[-1] - last_close) / last_close * 100
    arrow = "▲" if total >= 0 else "▼"
    print(f"\n[Module 3] {model_name} — 5-day forecast from {last_date}:")
    print(f"  {'Day':<6} {'Price':>10}  {'Change':>8}")
    print(f"  {'─'*28}")
    prev = last_close
    for i, p in enumerate(prices, 1):
        chg = (p - prev) / prev * 100
        print(f"  Day {i:<3}  ${p:>9.2f}  {'▲' if chg>=0 else '▼'} {chg:+.2f}%")
        prev = p
    print(f"  {'─'*28}")
    print(f"  Last close:   ${last_close:.2f}")
    print(f"  Day 5 target: ${prices[-1]:.2f}  ({arrow} {total:+.2f}% over 5 days)\n")


def _compute_metrics(actual, predicted):
    """Shared metric computation for all models."""
    dir_acc = float(np.mean(
        np.sign(np.diff(actual)) == np.sign(np.diff(predicted))))
    mae = float(np.mean(np.abs(actual - predicted)))
    return dir_acc, mae


# ── Baseline: LSTM ────────────────────────────────────────────────────────────

class LSTMForecaster(PriceForecaster):
    SEQ_LEN = 10
    HIDDEN  = 32
    EPOCHS  = 80
    LR      = 1e-3

    def predict(self, prices, anomalies, ticker="UNKNOWN") -> ForecastResult:
        import torch
        import torch.nn as nn

        if len(prices) < self.SEQ_LEN + 2:
            print("[Module 3] Not enough data for LSTM — using Mock.")
            return MockForecaster().predict(prices, anomalies, ticker)

        features = _build_base_features(prices, anomalies)
        Xtr, Xte, ytr, yte, scaler = _make_sequences(features, self.SEQ_LEN)
        n = scaler.n_features_in_

        class _LSTM(nn.Module):
            def __init__(s):
                super().__init__()
                s.lstm = nn.LSTM(2, self.HIDDEN, 1, batch_first=True)
                s.fc   = nn.Linear(self.HIDDEN, 1)
            def forward(s, x):
                return s.fc(s.lstm(x)[0][:, -1, :])

        model   = _LSTM()
        opt     = torch.optim.Adam(model.parameters(), lr=self.LR)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(Xtr)
        yt = torch.tensor(ytr).unsqueeze(1)

        model.train()
        for ep in range(self.EPOCHS):
            opt.zero_grad()
            loss_fn(model(Xt), yt).backward()
            opt.step()
            if (ep + 1) % 20 == 0:
                print(f"  [LSTM] epoch {ep+1}/{self.EPOCHS}")

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte)).numpy().flatten()

        actual    = _inverse_close(yte,  scaler, n)
        predicted = _inverse_close(pred, scaler, n)
        dir_acc, mae = _compute_metrics(actual, predicted)

        test_start = int(len(features) * 0.85)
        test_dates = [p.date for p in prices][test_start + self.SEQ_LEN:]

        scaled   = scaler.transform(features)
        forecast = _roll_forecast(model, scaled, self.SEQ_LEN, scaler, n)
        _print_forecast(forecast, prices[-1].close, prices[-1].date, "LSTM")
        print(f"[Module 3] LSTM — dir_acc={dir_acc:.1%}  MAE=${mae:.2f}")

        return ForecastResult(
            model_name   = "Baseline LSTM (2 features, seq=10)",
            day5_price   = float(forecast[4]),
            forecast_5d  = forecast,
            actual       = actual,
            predicted    = predicted,
            test_dates   = test_dates[:len(actual)],
            dir_accuracy = dir_acc,
            mae          = mae,
            sector_name  = SECTOR_NAMES.get(SECTOR_MAP.get(ticker.upper(), -1), "Unknown"),
        )


# ── Improved: Transformer ─────────────────────────────────────────────────────

class TransformerForecaster(PriceForecaster):
    """
    8-feature Transformer with self-attention.
    Features derived from the same anomaly detection signals as module2.
    """
    SEQ_LEN = 20
    D_MODEL = 64
    NHEAD   = 4
    LAYERS  = 2
    EPOCHS  = 300
    LR      = 5e-4

    def predict(self, prices, anomalies, ticker="UNKNOWN", events=None) -> ForecastResult:
        import torch
        import torch.nn as nn

        if len(prices) < self.SEQ_LEN + 2:
            print("[Module 3] Not enough data for Transformer — using LSTM.")
            return LSTMForecaster().predict(prices, anomalies, ticker)

        sector_id   = SECTOR_MAP.get(ticker.upper(), -1)
        sector_name = SECTOR_NAMES.get(sector_id, "Unknown")
        n_feats     = 9 if events is not None else 8
        print(f"[Module 3] Transformer — Sector: {ticker} → {sector_name}  "
              f"features={n_feats}")

        features = _build_rich_features(prices, anomalies, ticker, events=events)
        Xtr, Xte, ytr, yte, scaler = _make_sequences(features, self.SEQ_LEN)
        n = scaler.n_features_in_
        d, h, l = self.D_MODEL, self.NHEAD, self.LAYERS

        class _Transformer(nn.Module):
            def __init__(s):
                super().__init__()
                s.proj    = nn.Linear(n_feats, d)  # 8 or 9 depending on events
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d, nhead=h, dim_feedforward=128,
                    batch_first=True, dropout=0.1)
                s.encoder = nn.TransformerEncoder(enc_layer, num_layers=l)
                s.fc      = nn.Linear(d, 1)
            def forward(s, x):
                return s.fc(s.encoder(s.proj(x))[:, -1, :])

        model   = _Transformer()
        opt     = torch.optim.Adam(model.parameters(), lr=self.LR, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
        Xt = torch.tensor(Xtr)
        yt = torch.tensor(ytr).unsqueeze(1)

        model.train()
        for ep in range(self.EPOCHS):
            opt.zero_grad()
            loss = loss_fn(model(Xt), yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step(loss)
            if (ep + 1) % 60 == 0:
                print(f"  [Transformer] epoch {ep+1}/{self.EPOCHS}"
                      f"  loss={loss.item():.6f}")

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte)).numpy().flatten()

        actual    = _inverse_close(yte,  scaler, n)
        predicted = _inverse_close(pred, scaler, n)
        dir_acc, mae = _compute_metrics(actual, predicted)

        test_start = int(len(features) * 0.85)
        test_dates = [p.date for p in prices][test_start + self.SEQ_LEN:]

        scaled   = scaler.transform(features)
        forecast = _roll_forecast(model, scaled, self.SEQ_LEN, scaler, n)
        _print_forecast(forecast, prices[-1].close, prices[-1].date,
                        f"Transformer [{sector_name}]")
        print(f"[Module 3] Transformer — dir_acc={dir_acc:.1%}  MAE=${mae:.2f}")

        return ForecastResult(
            model_name   = f"Transformer [{sector_name}] ({n_feats} features, seq=20)",
            day5_price   = float(forecast[4]),
            forecast_5d  = forecast,
            actual       = actual,
            predicted    = predicted,
            test_dates   = test_dates[:len(actual)],
            dir_accuracy = dir_acc,
            mae          = mae,
            sector_name  = sector_name,
        )


# ── TFT: Temporal Fusion Transformer ─────────────────────────────────────────

class TFTForecaster(PriceForecaster):
    """
    Temporal Fusion Transformer — purpose-built for multi-variate time series.

    Key advantages over standard Transformer:
    - Separates static covariates (sector) from dynamic features (RSI, MACD)
    - Variable Selection Network: learns which features matter most
    - Multi-horizon output with uncertainty quantification
    - Gated Residual Network filters noise

    Same 8 features as TransformerForecaster for fair comparison:
    close, returns, volume, anomaly_flag, RSI, MACD, bollinger_pos, sector
    """
    SEQ_LEN    = 20   # encoder length (same as Transformer for fair comparison)
    PRED_LEN   = 5    # predict 5 days ahead
    EPOCHS     = 50
    LR         = 1e-3
    HIDDEN     = 64
    ATTN_HEADS = 4

    def predict(self, prices, anomalies, ticker="UNKNOWN", events=None) -> ForecastResult:
        import torch
        import torch.nn as nn
        import pandas as pd

        if len(prices) < self.SEQ_LEN + self.PRED_LEN + 2:
            print("[Module 3] Not enough data for TFT — using Transformer.")
            return TransformerForecaster().predict(prices, anomalies, ticker, events=events)

        sector_id   = SECTOR_MAP.get(ticker.upper(), -1)
        sector_name = SECTOR_NAMES.get(sector_id, "Unknown")
        n_feats     = 9 if events is not None else 8
        print(f"[Module 3] TFT — Sector: {ticker} → {sector_name}  "
              f"features={n_feats}")

        # ── Build features (same as Transformer for fair comparison) ──────────
        features = _build_rich_features(prices, anomalies, ticker, events=events)

        # ── Normalize ─────────────────────────────────────────────────────────
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)
        n      = scaler.n_features_in_

        # ── Build dataset with encoder + decoder windows ──────────────────────
        # TFT uses encoder (past SEQ_LEN days) → decoder (future PRED_LEN days)
        X_enc, X_dec, y_target = [], [], []
        for i in range(self.SEQ_LEN, len(scaled) - self.PRED_LEN):
            X_enc.append(scaled[i - self.SEQ_LEN: i])          # past 20 days
            X_dec.append(scaled[i: i + self.PRED_LEN, 1:])     # future known (no close)
            y_target.append(scaled[i: i + self.PRED_LEN, 0])   # future close prices

        X_enc    = torch.tensor(np.array(X_enc),    dtype=torch.float32)
        X_dec    = torch.tensor(np.array(X_dec),    dtype=torch.float32)
        y_target = torch.tensor(np.array(y_target), dtype=torch.float32)

        # Static feature: sector (same value repeated)
        static = torch.tensor(
            [[features[-1, -1]] for _ in range(len(X_enc))],
            dtype=torch.float32
        )

        # Train/test split (85/15)
        sp     = int(len(X_enc) * 0.85)
        tr_enc, te_enc = X_enc[:sp],    X_enc[sp:]
        tr_dec, te_dec = X_dec[:sp],    X_dec[sp:]
        tr_y,   te_y   = y_target[:sp], y_target[sp:]
        tr_st,  te_st  = static[:sp],   static[sp:]

        # ── TFT Model Definition ──────────────────────────────────────────────
        class _GRN(nn.Module):
            """Gated Residual Network — TFT's noise filtering component."""
            def __init__(s, inp, hid, out):
                super().__init__()
                s.fc1  = nn.Linear(inp, hid)
                s.fc2  = nn.Linear(hid, out)
                s.gate = nn.Linear(hid, out)
                s.ln   = nn.LayerNorm(out)
                s.proj = nn.Linear(inp, out) if inp != out else nn.Identity()
            def forward(s, x):
                h   = torch.relu(s.fc1(x))
                out = s.fc2(h) * torch.sigmoid(s.gate(h))
                return s.ln(out + s.proj(x))

        class _VarSelect(nn.Module):
            """Variable Selection Network — learns feature importance weights."""
            def __init__(s, n_feats, hidden):
                super().__init__()
                s.grns    = nn.ModuleList([_GRN(1, hidden, hidden) for _ in range(n_feats)])
                s.softmax = nn.Linear(n_feats * hidden, n_feats)
            def forward(s, x):
                # x: (batch, seq, n_feats)
                proc = [s.grns[i](x[..., i:i+1]) for i in range(x.shape[-1])]
                proc = torch.stack(proc, dim=-1)              # (B, T, H, F)
                flat = proc.reshape(proc.shape[0], proc.shape[1], -1)
                w    = torch.softmax(s.softmax(flat), dim=-1) # (B, T, F)
                out  = (proc * w.unsqueeze(2)).sum(-1)        # (B, T, H)
                return out, w

        class _TFT(nn.Module):
            def __init__(s):
                super().__init__()
                h = self.HIDDEN
                # Static encoder
                s.static_enc  = _GRN(1, h, h)
                # Variable selection for encoder (n_feats: 8 or 9)
                s.enc_varsel  = _VarSelect(n_feats, h)
                # LSTM encoder (short-term patterns)
                s.lstm_enc    = nn.LSTM(h, h, batch_first=True)
                # Variable selection for decoder (no close price — n_feats-1)
                s.dec_varsel  = _VarSelect(n_feats - 1, h)
                # LSTM decoder
                s.lstm_dec    = nn.LSTM(h, h, batch_first=True)
                # Multi-head attention (long-range patterns)
                s.attn        = nn.MultiheadAttention(h, self.ATTN_HEADS,
                                                       batch_first=True)
                # GRN post-attention
                s.post_attn   = _GRN(h, h, h)
                # Output projection per future step
                s.out         = nn.Linear(h, 1)

            def forward(s, enc, dec, stat):
                # Static context
                ctx = s.static_enc(stat)                        # (B, H)

                # Encoder variable selection + LSTM
                enc_feat, enc_w = s.enc_varsel(enc)             # (B, T_enc, H)
                enc_out, (hc, cc) = s.lstm_enc(enc_feat)

                # Decoder variable selection + LSTM (seeded with encoder state)
                dec_feat, _ = s.dec_varsel(dec)
                dec_out, _  = s.lstm_dec(dec_feat, (hc, cc))

                # Cross-attention: decoder queries encoder
                attn_out, _ = s.attn(dec_out, enc_out, enc_out)
                attn_out    = s.post_attn(attn_out + dec_out)

                # Output: predict close price for each future step
                return s.out(attn_out).squeeze(-1)              # (B, PRED_LEN)

        # ── Training ──────────────────────────────────────────────────────────
        model   = _TFT()
        opt     = torch.optim.Adam(model.parameters(), lr=self.LR, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

        model.train()
        for ep in range(self.EPOCHS):
            opt.zero_grad()
            pred = model(tr_enc, tr_dec, tr_st)
            loss = loss_fn(pred, tr_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step(loss)
            if (ep + 1) % 10 == 0:
                print(f"  [TFT] epoch {ep+1}/{self.EPOCHS}"
                      f"  loss={loss.item():.6f}")

        # ── Evaluation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            pred_te = model(te_enc, te_dec, te_st).numpy()  # (N_test, PRED_LEN)

        # Use Day-1 prediction for each test point to build the prediction curve
        pred_d1  = pred_te[:, 0]   # first predicted day for each window
        true_d1  = te_y[:, 0].numpy()

        # Inverse transform
        def inv(arr):
            pad = np.zeros((len(arr), n)); pad[:, 0] = arr
            return scaler.inverse_transform(pad)[:, 0]

        actual    = inv(true_d1)
        predicted = inv(pred_d1)
        dir_acc, mae = _compute_metrics(actual, predicted)

        # Test dates align with the start of each test decoder window
        test_start_idx = sp + self.SEQ_LEN
        test_dates     = [p.date for p in prices][test_start_idx:
                                                   test_start_idx + len(actual)]

        # ── 5-day rolling forecast from last available data ───────────────────
        last_enc = torch.tensor(scaled[-self.SEQ_LEN:][np.newaxis], dtype=torch.float32)
        last_dec = torch.tensor(scaled[-self.PRED_LEN:, 1:][np.newaxis], dtype=torch.float32)
        last_st  = torch.tensor([[features[-1, -1]]], dtype=torch.float32)

        with torch.no_grad():
            raw_forecast = model(last_enc, last_dec, last_st).numpy()[0]

        forecast = []
        for v in raw_forecast:
            pad = np.zeros((1, n)); pad[0, 0] = v
            forecast.append(round(float(scaler.inverse_transform(pad)[0, 0]), 2))

        _print_forecast(forecast, prices[-1].close, prices[-1].date,
                        f"TFT [{sector_name}]")
        print(f"[Module 3] TFT — dir_acc={dir_acc:.1%}  MAE=${mae:.2f}")

        return ForecastResult(
            model_name   = f"TFT [{sector_name}] ({n_feats} features, seq=20)",
            day5_price   = float(forecast[4]),
            forecast_5d  = forecast,
            actual       = actual,
            predicted    = predicted,
            test_dates   = test_dates[:len(actual)],
            dir_accuracy = dir_acc,
            mae          = mae,
            sector_name  = sector_name,
        )