"""
walk_forward_validation.py — Out-of-sample Walk-Forward Validation

训练数据：2021-01-01 ~ 2026-02-28
验证数据：2026-03-01 ~ 2026-04-15（模型从未见过）

验证模型：Transformer vs TFT vs MA baselines
验证逻辑：每天滚动预测下一天收盘价，与实际价格对比
输出：预测 vs 实际 对比图 + MAE / 方向准确率

运行：python3 walk_forward_validation.py
"""


from dotenv import load_dotenv
load_dotenv()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

from module1_data_fetcher     import YFinancePriceFetcher
from module1_data_fetcher     import FinnhubNewsFetcher
from module2_anomaly_detector import (
    FunnelDetector, ZScoreDetector, BollingerDetector,
    VolumeDetector, RSIDetector, MACDDetector,
    GapDetector, IntradayRangeDetector, ConsecutiveMoveDetector,
)
from module3_sentiment_lstm import (
    _build_rich_features, _make_sequences, _inverse_close,
    SECTOR_MAP, SECTOR_NAMES,
)

# ── Config ────────────────────────────────────────────────────────────────────

TICKER      = "META"

# ── 模式切换 ──────────────────────────────────────────────────────────────────
# MODE = "5Y_PRICE"  : 5年股价 + 1年新闻(前4年情绪填0)  → 样本量多，新闻覆盖部分
# MODE = "1Y_ALIGNED": 1年股价 + 1年新闻(完全对齐)      → 样本量少，数据干净
MODE = "5Y_PRICE"   # ← 改这里切换模式

if MODE == "5Y_PRICE":
    TRAIN_START  = date(2021, 1, 1)
    TRAIN_END    = date(2026, 2, 28)
    NEWS_START   = date(2025, 3, 1)   # 新闻API只能抓最近1年
else:  # 1Y_ALIGNED
    TRAIN_START  = date(2025, 3, 1)
    TRAIN_END    = date(2026, 2, 28)
    NEWS_START   = date(2025, 3, 1)

VAL_START   = date(2026, 3, 1)
VAL_END     = date(2026, 4, 15)

# Feature count: 8 technical + 1 daily sentiment
n_feats = 9

# Shared hyperparameters
SEQ_LEN  = 20
D_MODEL  = 64
NHEAD    = 4
LAYERS   = 2
PRED_LEN = 5   # TFT decoder length

# Transformer
TF_EPOCHS = 300
TF_LR     = 5e-4

# TFT
TFT_EPOCHS = 50
TFT_LR     = 1e-3
TFT_HIDDEN = 64

BLUE   = "#1d4ed8"
ORANGE = "#f97316"
PURPLE = "#7c3aed"
GREEN  = "#16a34a"
RED    = "#dc2626"
GRAY   = "#94a3b8"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "text.color": "#1e293b",
    "axes.labelcolor": "#1e293b", "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8", "axes.edgecolor": "#e2e8f0",
})


# ── Model definitions ─────────────────────────────────────────────────────────

class _Transformer(nn.Module):
    def __init__(self, d=D_MODEL, h=NHEAD, l=LAYERS):
        super().__init__()
        self.proj    = nn.Linear(n_feats, d)  # 9 features (8 tech + sentiment)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d, nhead=h, dim_feedforward=128,
            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=l)
        self.fc      = nn.Linear(d, 1)
    def forward(self, x):
        return self.fc(self.encoder(self.proj(x))[:, -1, :])


class _GRN(nn.Module):
    """Gated Residual Network — TFT noise filtering."""
    def __init__(self, inp, hid, out):
        super().__init__()
        self.fc1  = nn.Linear(inp, hid)
        self.fc2  = nn.Linear(hid, out)
        self.gate = nn.Linear(hid, out)
        self.ln   = nn.LayerNorm(out)
        self.proj = nn.Linear(inp, out) if inp != out else nn.Identity()
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.ln(self.fc2(h) * torch.sigmoid(self.gate(h)) + self.proj(x))


class _VarSelect(nn.Module):
    """Variable Selection Network — learns feature importance."""
    def __init__(self, n_feats, hidden):
        super().__init__()
        self.grns    = nn.ModuleList([_GRN(1, hidden, hidden)
                                      for _ in range(n_feats)])
        self.softmax = nn.Linear(n_feats * hidden, n_feats)
    def forward(self, x):
        proc = torch.stack([self.grns[i](x[..., i:i+1])
                            for i in range(x.shape[-1])], dim=-1)
        flat = proc.reshape(proc.shape[0], proc.shape[1], -1)
        w    = torch.softmax(self.softmax(flat), dim=-1)
        return (proc * w.unsqueeze(2)).sum(-1), w


class _TFT(nn.Module):
    def __init__(self, h=TFT_HIDDEN, heads=NHEAD):
        super().__init__()
        self.static_enc = _GRN(1, h, h)
        self.enc_varsel = _VarSelect(n_feats,     h)   # 9 features
        self.lstm_enc   = nn.LSTM(h, h, batch_first=True)
        self.dec_varsel = _VarSelect(n_feats - 1, h)   # 8 (no close)
        self.lstm_dec   = nn.LSTM(h, h, batch_first=True)
        self.attn       = nn.MultiheadAttention(h, heads, batch_first=True)
        self.post_attn  = _GRN(h, h, h)
        self.out        = nn.Linear(h, 1)
    def forward(self, enc, dec, stat):
        ctx = self.static_enc(stat)
        enc_feat, _ = self.enc_varsel(enc)
        enc_out, (hc, cc) = self.lstm_enc(enc_feat)
        dec_feat, _ = self.dec_varsel(dec)
        dec_out, _  = self.lstm_dec(dec_feat, (hc, cc))
        attn_out, _ = self.attn(dec_out, enc_out, enc_out)
        attn_out    = self.post_attn(attn_out + dec_out)
        return self.out(attn_out).squeeze(-1)


# ── Helper ────────────────────────────────────────────────────────────────────

def compute_metrics(pred, actual):
    mae     = float(np.mean(np.abs(actual - pred)))
    dir_acc = float(np.mean(
        np.sign(np.diff(actual)) == np.sign(np.diff(pred))))
    mape    = float(np.mean(np.abs(actual - pred) / actual * 100))
    return mae, dir_acc, mape


# ── Main validation function ──────────────────────────────────────────────────

def run_validation():
    fetcher      = YFinancePriceFetcher()
    news_fetcher = FinnhubNewsFetcher()
    detector = FunnelDetector([
        ZScoreDetector(), BollingerDetector(), VolumeDetector(),
        RSIDetector(), MACDDetector(), GapDetector(),
        IntradayRangeDetector(), ConsecutiveMoveDetector(),
    ], min_triggers=2)

    # ── Step 1: Fetch data ────────────────────────────────────────────────────
    print(f"[0] Mode: {MODE}")
    print(f"    Train: {TRAIN_START} ~ {TRAIN_END}  |  "
          f"News: {NEWS_START} ~ {VAL_END}  |  "
          f"Val: {VAL_START} ~ {VAL_END}")

    print(f"\n[1] Fetching training data {TRAIN_START} ~ {TRAIN_END}...")
    train_prices = fetcher.fetch_prices(TICKER, TRAIN_START, TRAIN_END)
    print(f"    {len(train_prices)} training days loaded.")

    train_anomalies = detector.detect(train_prices, [], ticker=TICKER)
    print(f"    {len(train_anomalies)} anomalies in training period.")

    print(f"\n[1] Fetching news {NEWS_START} ~ {VAL_END}...")
    all_events = news_fetcher.fetch_news(TICKER, NEWS_START, VAL_END)
    print(f"    {len(all_events)} news events loaded.")

    print(f"\n[1] Fetching validation data {VAL_START} ~ {VAL_END}...")
    val_prices = fetcher.fetch_prices(TICKER, VAL_START, VAL_END)
    print(f"    {len(val_prices)} validation days loaded.")

    if not val_prices:
        print("    No validation data. Check date range.")
        return

    # ── Step 2: Build shared features ────────────────────────────────────────
    print(f"\n[2] Building features ({n_feats} features = 8 technical + 1 sentiment)...")
    all_prices   = train_prices + val_prices
    features     = _build_rich_features(all_prices, train_anomalies, TICKER,
                                         events=all_events)   # ← 传入新闻
    scaler       = MinMaxScaler()
    # Fit scaler on TRAINING data only — never peek at validation
    scaler.fit(features[:len(train_prices)])
    all_scaled   = scaler.transform(features)
    n            = features.shape[1]
    train_len    = len(train_prices)
    val_actual   = np.array([p.close for p in val_prices])
    val_dates    = [p.date for p in val_prices]

    # ── Step 3: Train Transformer ─────────────────────────────────────────────
    print(f"\n[3] Training Transformer ({TF_EPOCHS} epochs)...")
    train_scaled = all_scaled[:train_len]
    # Build sequences using pre-fit scaler
    train_seq_X, train_seq_y = [], []
    for i in range(SEQ_LEN, train_len):
        train_seq_X.append(all_scaled[i-SEQ_LEN:i])
        train_seq_y.append(all_scaled[i, 0])
    Xtr_t = torch.tensor(np.array(train_seq_X), dtype=torch.float32)
    ytr_t = torch.tensor(np.array(train_seq_y), dtype=torch.float32).unsqueeze(1)

    tf_model = _Transformer()
    tf_opt   = torch.optim.Adam(tf_model.parameters(), lr=TF_LR, weight_decay=1e-4)
    tf_loss  = nn.MSELoss()
    tf_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                 tf_opt, patience=20, factor=0.5)

    tf_model.train()
    for ep in range(TF_EPOCHS):
        tf_opt.zero_grad()
        loss = tf_loss(tf_model(Xtr_t), ytr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tf_model.parameters(), 1.0)
        tf_opt.step()
        tf_sched.step(loss)
        if (ep + 1) % 60 == 0:
            print(f"  [Transformer] epoch {ep+1}/{TF_EPOCHS}"
                  f"  loss={loss.item():.6f}")

    # ── Step 4: Train TFT ─────────────────────────────────────────────────────
    print(f"\n[4] Training TFT ({TFT_EPOCHS} epochs)...")
    X_enc, X_dec, y_tft = [], [], []
    for i in range(SEQ_LEN, train_len - PRED_LEN):
        X_enc.append(all_scaled[i-SEQ_LEN:i])
        X_dec.append(all_scaled[i:i+PRED_LEN, 1:])
        y_tft.append(all_scaled[i:i+PRED_LEN, 0])

    X_enc_t = torch.tensor(np.array(X_enc), dtype=torch.float32)
    X_dec_t = torch.tensor(np.array(X_dec), dtype=torch.float32)
    y_tft_t = torch.tensor(np.array(y_tft), dtype=torch.float32)
    stat_t  = torch.tensor(
        [[features[train_len-1, -1]]] * len(X_enc), dtype=torch.float32)

    tft_model = _TFT()
    tft_opt   = torch.optim.Adam(tft_model.parameters(),
                                  lr=TFT_LR, weight_decay=1e-4)
    tft_loss  = nn.MSELoss()
    tft_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  tft_opt, patience=10, factor=0.5)

    tft_model.train()
    for ep in range(TFT_EPOCHS):
        tft_opt.zero_grad()
        loss = tft_loss(tft_model(X_enc_t, X_dec_t, stat_t), y_tft_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tft_model.parameters(), 1.0)
        tft_opt.step()
        tft_sched.step(loss)
        if (ep + 1) % 10 == 0:
            print(f"  [TFT] epoch {ep+1}/{TFT_EPOCHS}"
                  f"  loss={loss.item():.6f}")

    # ── Step 5: Walk-forward validation ───────────────────────────────────────
    print(f"\n[5] Running walk-forward validation on {VAL_START}~{VAL_END}...")

    tf_predicted  = []
    tft_predicted = []
    all_closes    = [p.close for p in all_prices]

    tf_model.eval()
    tft_model.eval()

    for i in range(len(val_prices)):
        idx = train_len + i

        # ── Transformer prediction ────────────────────────────────────────────
        seq  = all_scaled[idx-SEQ_LEN:idx]
        inp  = torch.tensor(seq[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            p_scaled = tf_model(inp).item()
        pad = np.zeros((1, n)); pad[0, 0] = p_scaled
        tf_predicted.append(float(scaler.inverse_transform(pad)[0, 0]))

        # ── TFT prediction ────────────────────────────────────────────────────
        enc  = torch.tensor(all_scaled[idx-SEQ_LEN:idx][np.newaxis],
                             dtype=torch.float32)
        # For decoder: use last PRED_LEN known rows (minus close col)
        dec_rows = all_scaled[max(0, idx-PRED_LEN):idx, 1:]
        if len(dec_rows) < PRED_LEN:
            dec_rows = np.pad(dec_rows,
                              ((PRED_LEN-len(dec_rows), 0), (0, 0)))
        dec  = torch.tensor(dec_rows[np.newaxis], dtype=torch.float32)
        stat = torch.tensor([[features[idx-1, -1]]], dtype=torch.float32)
        with torch.no_grad():
            tft_out = tft_model(enc, dec, stat).numpy()[0]
        pad = np.zeros((1, n)); pad[0, 0] = tft_out[0]
        tft_predicted.append(float(scaler.inverse_transform(pad)[0, 0]))

    tf_predicted  = np.array(tf_predicted)
    tft_predicted = np.array(tft_predicted)

    # ── Step 6: MA baselines ──────────────────────────────────────────────────
    sma5_pred  = np.array([np.mean(all_closes[train_len+i-5:train_len+i])
                            for i in range(len(val_prices))])
    sma20_pred = np.array([np.mean(all_closes[train_len+i-20:train_len+i])
                            for i in range(len(val_prices))])
    naive_pred = np.array([all_closes[train_len+i-1]
                            for i in range(len(val_prices))])

    # ── Step 7: Print comparison ──────────────────────────────────────────────
    results = {
        "Naive":       compute_metrics(naive_pred,  val_actual),
        "SMA-5":       compute_metrics(sma5_pred,   val_actual),
        "SMA-20":      compute_metrics(sma20_pred,  val_actual),
        "Transformer": compute_metrics(tf_predicted, val_actual),
        "TFT":         compute_metrics(tft_predicted,val_actual),
    }

    print(f"\n{'─'*62}")
    print(f"  Walk-Forward Validation — {TICKER}  "
          f"({VAL_START} ~ {VAL_END})")
    print(f"{'─'*62}")
    print(f"  {'Model':<20} {'MAE ($)':>10}  {'Dir Acc':>9}  {'MAPE':>8}")
    print(f"  {'─'*58}")
    for name, (mae, dir_acc, mape) in results.items():
        tag = " ← baseline" if name in ["Naive","SMA-5","SMA-20"] else ""
        print(f"  {name:<20} ${mae:>9.2f}  {dir_acc:>8.1%}  {mape:>7.2f}%{tag}")
    print(f"{'─'*62}")

    # Best baseline vs our models
    best_bl  = min(results["Naive"][0], results["SMA-5"][0],
                   results["SMA-20"][0])
    tf_imp   = (best_bl - results["Transformer"][0]) / best_bl * 100
    tft_imp  = (best_bl - results["TFT"][0])         / best_bl * 100
    winner   = "TFT" if results["TFT"][0] < results["Transformer"][0] \
               else "Transformer"
    gap      = abs(results["TFT"][0] - results["Transformer"][0])

    print(f"\n  Transformer vs best baseline: {tf_imp:+.1f}% MAE improvement")
    print(f"  TFT        vs best baseline: {tft_imp:+.1f}% MAE improvement")
    print(f"  Winner: {winner} (MAE gap: ${gap:.2f})\n")

    # ── Step 8: Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 12), facecolor="white")
    fig.suptitle(
        f"{TICKER} — Walk-Forward Validation  [{MODE}]  "
        f"(trained {TRAIN_START}~{TRAIN_END}, "
        f"validated {VAL_START}~{VAL_END})",
        fontsize=10, color=GRAY, x=0.01, ha="left")

    gs  = gridspec.GridSpec(3, 1, figure=fig,
                            height_ratios=[3, 1, 1], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Panel 1: Predicted vs Actual
    ctx_prices = train_prices[-40:]
    ctx_dates  = [p.date for p in ctx_prices]
    ctx_closes = [p.close for p in ctx_prices]

    ax1.plot(ctx_dates,  ctx_closes,   color=GRAY,   lw=1.2,
             alpha=0.4, linestyle="--", label="Training (last 40d)")
    ax1.plot(val_dates, val_actual,    color=BLUE,   lw=2.0,
             label="Actual price")
    ax1.plot(val_dates, tf_predicted,  color=ORANGE, lw=1.5,
             linestyle="--",
             label=f"Transformer  MAE=${results['Transformer'][0]:.2f}")
    ax1.plot(val_dates, tft_predicted, color=PURPLE, lw=1.5,
             linestyle=":",
             label=f"TFT          MAE=${results['TFT'][0]:.2f}")
    ax1.plot(val_dates, sma5_pred,     color=GRAY,   lw=1.0,
             linestyle="-.", alpha=0.6,
             label=f"SMA-5        MAE=${results['SMA-5'][0]:.2f}")

    ax1.axvspan(val_dates[0], val_dates[-1],
                alpha=0.04, color=ORANGE)
    ax1.axvline(val_dates[0], color=ORANGE, lw=0.8, linestyle=":")
    ax1.set_title(f"{TICKER} — Transformer vs TFT vs SMA baseline",
                  fontsize=11, fontweight="normal", pad=8, loc="left")
    ax1.set_ylabel("Price (USD)", fontsize=9, color=GRAY)
    ax1.tick_params(colors=GRAY, labelsize=8)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.spines[["left","bottom"]].set_color("#e2e8f0")
    ax1.grid(axis="y", color="#f1f5f9", linewidth=0.8)
    ax1.legend(fontsize=8, framealpha=0, loc="upper left")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v:.0f}"))

    # Panel 2: Absolute error per day
    tf_errors  = np.abs(val_actual - tf_predicted)
    tft_errors = np.abs(val_actual - tft_predicted)
    x_idx      = np.arange(len(val_dates))
    w          = 0.35
    ax2.bar(x_idx - w/2, tf_errors,  width=w, color=ORANGE,
            alpha=0.7, label="Transformer error")
    ax2.bar(x_idx + w/2, tft_errors, width=w, color=PURPLE,
            alpha=0.7, label="TFT error")
    ax2.axhline(results["Transformer"][0], color=ORANGE,
                lw=1.0, linestyle="--", alpha=0.8)
    ax2.axhline(results["TFT"][0],         color=PURPLE,
                lw=1.0, linestyle="--", alpha=0.8)
    ax2.set_xticks(x_idx[::3])
    ax2.set_xticklabels([str(val_dates[i]) for i in range(0,len(val_dates),3)],
                         rotation=30, fontsize=7)
    ax2.set_ylabel("Abs. Error ($)", fontsize=8, color=GRAY)
    ax2.set_title("Daily absolute error — Transformer vs TFT",
                  fontsize=9, fontweight="normal", loc="left")
    ax2.tick_params(colors=GRAY, labelsize=7)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.spines[["left","bottom"]].set_color("#e2e8f0")
    ax2.grid(axis="y", color="#f1f5f9", linewidth=0.8)
    ax2.legend(fontsize=8, framealpha=0)

    # Panel 3: MAE bar chart summary
    model_names = ["Naive", "SMA-5", "SMA-20", "Transformer", "TFT"]
    maes        = [results[m][0] for m in model_names]
    colors_bar  = [GRAY, GRAY, GRAY, ORANGE, PURPLE]
    bars = ax3.bar(model_names, maes, color=colors_bar, alpha=0.8, width=0.5)
    for bar, mae in zip(bars, maes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"${mae:.2f}", ha="center", fontsize=8, color="#334155")
    ax3.set_ylabel("MAE ($)", fontsize=8, color=GRAY)
    ax3.set_title("MAE comparison — all models",
                  fontsize=9, fontweight="normal", loc="left")
    ax3.tick_params(colors=GRAY, labelsize=8)
    ax3.spines[["top","right"]].set_visible(False)
    ax3.spines[["left","bottom"]].set_color("#e2e8f0")
    ax3.grid(axis="y", color="#f1f5f9", linewidth=0.8)

    plt.tight_layout()
    path = f"{TICKER}_validation_{MODE}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Validation] Chart saved → {path}")

    return results


if __name__ == "__main__":
    run_validation()