"""
Module 3 — Sentiment Analysis + Price Forecasting

        SentimentAnalyzer (abstract)
        ├── MockSentimentAnalyzer    ← heuristic mock
        └── FinBERTAnalyzer          ← real: ProsusAI/finbert

        PriceForecaster (abstract)
        ├── MockForecaster           ← heuristic mock
        ├── LSTMForecaster           ← baseline: 2 features, seq=10
        └── TransformerForecaster    ← improved: 8 features, seq=20

pip install torch transformers scikit-learn
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
        return ForecastResult("Mock", forecast[4], forecast)


# ── Real Sentiment: FinBERT ───────────────────────────────────────────────────

class FinBERTAnalyzer(SentimentAnalyzer):
    """
    pip install transformers torch
    """
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

    def analyze(self, events: list[MarketEvent]) -> tuple[float, str]:
        if not events:
            return 0.0, "neutral"
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


def _build_rich_features(prices, anomalies, ticker) -> np.ndarray:
    """8 features: close + returns + volume + anomaly + RSI + MACD + BB + sector."""
    closes  = np.array([p.close        for p in prices], dtype=np.float32)
    volumes = np.array([p.volume / 1e8 for p in prices], dtype=np.float32)
    returns = np.diff(closes, prepend=closes[0]) / (closes[0] + 1e-9)

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

    return np.stack([
        closes,
        returns.astype(np.float32),
        volumes,
        flags,
        rolling_rsi(closes).astype(np.float32),
        rolling_macd(closes).astype(np.float32),
        bb_position(closes).astype(np.float32),
        sector_feat,
    ], axis=1)


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
        dir_acc   = float(np.mean(
            np.sign(np.diff(actual)) == np.sign(np.diff(predicted))))
        mae = float(np.mean(np.abs(actual - predicted)))

        test_start = int(len(features) * 0.85)
        test_dates = [p.date for p in prices][test_start + self.SEQ_LEN:]

        scaled   = scaler.transform(features)
        forecast = _roll_forecast(model, scaled, self.SEQ_LEN, scaler, n)
        _print_forecast(forecast, prices[-1].close, prices[-1].date, "LSTM")
        print(f"[Module 3] LSTM — dir_acc={dir_acc:.1%}  MAE=${mae:.2f}")

        return ForecastResult(
            model_name   = "Baseline LSTM (2 features, seq=10)",
            day5_price   = forecast[4],
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
    SEQ_LEN = 20
    D_MODEL = 64
    NHEAD   = 4
    LAYERS  = 2
    EPOCHS  = 300
    LR      = 5e-4

    def predict(self, prices, anomalies, ticker="UNKNOWN") -> ForecastResult:
        import torch
        import torch.nn as nn

        if len(prices) < self.SEQ_LEN + 2:
            print("[Module 3] Not enough data for Transformer — using LSTM.")
            return LSTMForecaster().predict(prices, anomalies, ticker)

        sector_id   = SECTOR_MAP.get(ticker.upper(), -1)
        sector_name = SECTOR_NAMES.get(sector_id, "Unknown")
        print(f"[Module 3] Sector: {ticker} → {sector_name}")

        features = _build_rich_features(prices, anomalies, ticker)
        Xtr, Xte, ytr, yte, scaler = _make_sequences(features, self.SEQ_LEN)
        n = scaler.n_features_in_
        d, h, l = self.D_MODEL, self.NHEAD, self.LAYERS

        class _Transformer(nn.Module):
            def __init__(s):
                super().__init__()
                s.proj    = nn.Linear(8, d)
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
                print(f"  [Transformer] epoch {ep+1}/{self.EPOCHS}  loss={loss.item():.6f}")

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte)).numpy().flatten()

        actual    = _inverse_close(yte,  scaler, n)
        predicted = _inverse_close(pred, scaler, n)
        dir_acc   = float(np.mean(
            np.sign(np.diff(actual)) == np.sign(np.diff(predicted))))
        mae = float(np.mean(np.abs(actual - predicted)))

        test_start = int(len(features) * 0.85)
        test_dates = [p.date for p in prices][test_start + self.SEQ_LEN:]

        scaled   = scaler.transform(features)
        forecast = _roll_forecast(model, scaled, self.SEQ_LEN, scaler, n)
        _print_forecast(forecast, prices[-1].close, prices[-1].date,
                        f"Transformer [{sector_name}]")
        print(f"[Module 3] Transformer — dir_acc={dir_acc:.1%}  MAE=${mae:.2f}")

        return ForecastResult(
            model_name   = f"Transformer [{sector_name}] (8 features, seq=20)",
            day5_price   = forecast[4],
            forecast_5d  = forecast,
            actual       = actual,
            predicted    = predicted,
            test_dates   = test_dates[:len(actual)],
            dir_accuracy = dir_acc,
            mae          = mae,
            sector_name  = sector_name,
        )