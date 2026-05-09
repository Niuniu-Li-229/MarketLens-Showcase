"""MarketLens — five-module ML pipeline for equity analysis.

Public submodules:
    models                    — shared dataclasses (PricePoint, AnomalyPoint, …)
    module1_data_fetcher      — yFinance + Finnhub price/news fetchers
    module2_anomaly_detector  — 8-detector funnel anomaly detection
    module3_sentiment_lstm    — FinBERT sentiment + Transformer/TFT forecasters
    module4_claude_report     — GPT-4o analyst report generator
    module5_visualizer        — Matplotlib chart generator (CLI use)
"""
