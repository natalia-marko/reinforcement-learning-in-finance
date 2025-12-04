# path: src/utile_gdelt.py
"""Utilities for correlation-first GDELT sentiment analysis (data-science friendly).

Focus: co-movement, not prediction. Monthly frequency, 2015–2025.

Main tasks:
- Load v2tone_last.csv and standardize sentiment per ticker (winsorize + z-score).
- Download monthly prices (yfinance) for target tickers and compute same-month log returns.
- Merge sentiment with returns and market return (SPY by default).
- Compute pooled & per-ticker Pearson/Spearman correlations + BH/FDR.
- Rolling (12m) correlation of avg sentiment vs market.
- Simple plots and CSV/JSON outputs.

Design: minimal API, clear docstrings, few knobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import os

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

# ---- Defaults for your project ----
DEFAULT_TICKERS: List[str] = [
    "AEM", "AI", "AMD", "APP", "ARBE", "ASML", "GOOG", "IONQ", "MRVL",
    "MSFT", "MU", "NVDA", "PLTR", "QBTS", "QQQ", "RDDT", "RGTI", "SMR", "VERU",
]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_MARKET = "SPY"


@dataclass
class GdeltCorrConfig:
    """Lightweight configuration for correlation analysis."""
    csv_path: str = "data/v2tone_last.csv"
    sentiment_col: str = "weighted_tone"  # or "mean_tone"
    tickers: Sequence[str] = tuple(DEFAULT_TICKERS)
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    market_ticker: str = DEFAULT_MARKET
    winsor: Tuple[float, float] = (0.01, 0.99)
    rolling_z: bool = False
    rolling_window: int = 12
    out_dir: str = "out/gdelt"


# -------------------- IO & transforms --------------------

def _winsorize(s: pd.Series, p: Tuple[float, float]) -> pd.Series:
    lo, hi = s.quantile(p[0]), s.quantile(p[1])
    return s.clip(lo, hi)


def load_gdelt(cfg: GdeltCorrConfig) -> pd.DataFrame:
    """Load CSV and prepare monthly sentiment.

    Expects columns: ticker, year_month, weighted_tone/mean_tone, article_count, etc.
    """
    df = pd.read_csv(cfg.csv_path)
    keep = [
        "ticker", "year_month", "weighted_tone", "mean_tone",
        "mean_positive", "mean_negative", "article_count",
    ]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["year_month"] = pd.to_datetime(df["year_month"]).dt.to_period("M").dt.to_timestamp("M")

    for c in ("weighted_tone", "mean_tone", "mean_positive", "mean_negative"):
        if c in df.columns:
            df[c] = _winsorize(df[c], cfg.winsor)

    if cfg.tickers:
        df = df[df["ticker"].isin([t.upper() for t in cfg.tickers])]

    df = df[(df["year_month"] >= pd.to_datetime(cfg.start).to_period("M").to_timestamp("M")) &
            (df["year_month"] <= pd.to_datetime(cfg.end).to_period("M").to_timestamp("M"))]
    return df


def standardize_sentiment(df: pd.DataFrame, cfg: GdeltCorrConfig) -> pd.DataFrame:
    """Within-ticker z-score; optionally rolling z."""
    col = cfg.sentiment_col
    if col not in df.columns:
        raise KeyError(f"Sentiment column '{col}' not found in CSV")

    df = df.sort_values(["ticker", "year_month"]).copy()

    def _z(g: pd.DataFrame) -> pd.Series:
        x = g[col]
        if cfg.rolling_z:
            mean = x.rolling(cfg.rolling_window, min_periods=max(3, cfg.rolling_window // 3)).mean()
            std = x.rolling(cfg.rolling_window, min_periods=max(3, cfg.rolling_window // 3)).std()
            return (x - mean) / std
        return (x - x.mean()) / x.std(ddof=0)

    df["sentiment_z"] = df.groupby("ticker", group_keys=False).apply(_z)
    return df


# -------------------- Prices & returns --------------------

def _to_month_end(ts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.to_datetime(ts).to_period("M").to_timestamp("M")


def get_monthly_prices(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    """Download monthly adjusted close. Simple and tolerant to missing tickers."""
    tickers = [t.upper() for t in tickers]
    px = yf.download(tickers, start=start, end=end, interval="1mo", auto_adjust=True, progress=False)["Adj Close"]
    if isinstance(px, pd.Series):  # single ticker edge case
        px = px.to_frame()
    px = px.dropna(how="all")
    px.index = _to_month_end(px.index)
    return px


def same_month_log_returns(px: pd.DataFrame) -> pd.DataFrame:
    rets = np.log(px / px.shift(1))
    out = (
        rets.stack()
        .rename("ret")
        .reset_index()
        .rename(columns={"level_0": "year_month", "level_1": "ticker"})
        .dropna()
    )
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def market_same_month_return(ticker: str, start: str, end: str) -> pd.DataFrame:
    px = get_monthly_prices([ticker], start=start, end=end)
    ret = same_month_log_returns(px).rename(columns={"ret": "mkt_ret", "ticker": "mkt_ticker"})
    return ret[["year_month", "mkt_ret"]]


# -------------------- Merge & metrics --------------------

def merge_sentiment_returns(df_sent: pd.DataFrame, tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    px = get_monthly_prices(tickers, start, end)
    ret = same_month_log_returns(px)
    merged = df_sent.merge(ret, on=["ticker", "year_month"], how="inner")
    return merged.dropna(subset=["sentiment_z", "ret"])  # contemporaneous set


def add_market(df: pd.DataFrame, market_ticker: str, start: str, end: str) -> pd.DataFrame:
    mkt = market_same_month_return(market_ticker, start, end)
    return df.merge(mkt, on="year_month", how="left")


def pooled_correlations(df: pd.DataFrame) -> Dict[str, float]:
    """Pooled Pearson/Spearman for sentiment vs stock return and market."""
    s = df["sentiment_z"].astype(float)
    r = df["ret"].astype(float)
    m = df["mkt_ret"].astype(float)

    out = {
        "pearson_sent_vs_ret": float(pearsonr(s, r)[0]),
        "spearman_sent_vs_ret": float(spearmanr(s, r)[0]),
        "pearson_sent_vs_mkt": float(pearsonr(s, m)[0]),
        "spearman_sent_vs_mkt": float(spearmanr(s, m)[0]),
    }

    # Month-average sentiment vs market (why: portfolio-level co-movement)
    avg_s = df.groupby("year_month")["sentiment_z"].mean()
    mkt = df.drop_duplicates("year_month").set_index("year_month")["mkt_ret"]
    out["pearson_avgSent_vs_mkt"] = float(pearsonr(avg_s, mkt.reindex(avg_s.index))[0])
    return out


def per_ticker_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tkr, g in df.groupby("ticker"):
        if g["sentiment_z"].notna().sum() < 6:
            continue
        pr, pp = pearsonr(g["sentiment_z"], g["ret"])  # r, p
        sr, sp = spearmanr(g["sentiment_z"], g["ret"])  # rho, p
        rows.append({"ticker": tkr, "n": int(len(g)), "pearson_r": float(pr), "pearson_p": float(pp), "spearman_rho": float(sr), "spearman_p": float(sp)})
    out = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
    return out


def bh_fdr(p: pd.Series, alpha: float = 0.10) -> pd.Series:
    """Benjamini–Hochberg FDR control (vectorized)."""
    p = p.copy()
    m = p.notna().sum()
    rank = p.rank(method="first")
    thresh = alpha * rank / m
    return p <= thresh


def partial_corr_via_ols(df: pd.DataFrame) -> Tuple[float, float]:
    """OLS with HAC errors: ret ~ sentiment + market. Beta,t for sentiment."""
    X = df[["sentiment_z", "mkt_ret"]].copy()
    X = sm.add_constant(X)
    y = df["ret"].values
    mod = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    beta = float(mod.params.get("sentiment_z", np.nan))
    tval = float(mod.tvalues.get("sentiment_z", np.nan))
    return beta, tval


def rolling_corr_avg_sent_vs_mkt(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    avg_s = df.groupby("year_month")["sentiment_z"].mean().rename("avg_sent")
    mkt = df.drop_duplicates("year_month").set_index("year_month")["mkt_ret"].rename("mkt")
    tmp = pd.concat([avg_s, mkt], axis=1).dropna()
    out = tmp.assign(roll_corr=lambda x: x["avg_sent"].rolling(window).corr(x["mkt"]))
    return out.reset_index()


# -------------------- Plots --------------------

def save_plots(df: pd.DataFrame, roll: pd.DataFrame, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    # Scatter: avg sentiment vs market
    avg_sent = df.groupby("year_month")["sentiment_z"].mean()
    mkt = df.drop_duplicates("year_month").set_index("year_month")["mkt_ret"]
    x = avg_sent.reindex(mkt.index).dropna()
    y = mkt.reindex(x.index)

    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    if len(x) > 2:
        k, b = np.polyfit(x.values, y.values, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        plt.plot(xs, k * xs + b)
    plt.title("Avg monthly sentiment (z) vs market return")
    plt.xlabel("Avg sentiment z")
    plt.ylabel("Market return")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_avg_sent_vs_mkt.png"))
    plt.close()

    # Rolling correlation
    plt.figure()
    plt.plot(roll["year_month"], roll["roll_corr"])
    plt.title("Rolling correlation (12m) avg sentiment vs market")
    plt.xlabel("Month")
    plt.ylabel("Rolling corr")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling_corr_avg_sent_vs_mkt.png"))
    plt.close()


# -------------------- High-level runner --------------------

def run_correlation_analysis(cfg: Optional[GdeltCorrConfig] = None):
    """End-to-end: load, standardize, merge, metrics, plots, save outputs.

    Returns
    -------
    merged : pd.DataFrame
        Per (ticker, month) rows with sentiment_z, ret, mkt_ret.
    per_ticker : pd.DataFrame
        Correlations per ticker with BH/FDR flags.
    pooled : Dict[str, float]
        Pooled Pearson/Spearman metrics.
    roll : pd.DataFrame
        Rolling (12m) correlation of avg sentiment vs market.
    reg : Dict[str, float]
        OLS partial-corr beta & t-stat for sentiment.
    """
    cfg = cfg or GdeltCorrConfig()

    os.makedirs(cfg.out_dir, exist_ok=True)

    sent = load_gdelt(cfg)
    sent = standardize_sentiment(sent, cfg)

    merged = merge_sentiment_returns(sent, cfg.tickers, cfg.start, cfg.end)
    merged = add_market(merged, cfg.market_ticker, cfg.start, cfg.end)

    pooled = pooled_correlations(merged)
    per = per_ticker_correlations(merged)
    if not per.empty:
        per["pearson_fdr10"] = bh_fdr(per["pearson_p"], alpha=0.10)
        per["spearman_fdr10"] = bh_fdr(per["spearman_p"], alpha=0.10)

    beta, t = partial_corr_via_ols(merged)
    roll = rolling_corr_avg_sent_vs_mkt(merged, window=12)

    per.to_csv(os.path.join(cfg.out_dir, "gdelt_per_ticker_correlations.csv"), index=False)
    with open(os.path.join(cfg.out_dir, "gdelt_pooled_correlations.json"), "w") as f:
        json.dump({**pooled, "ols_beta_sent": beta, "ols_t_sent": t}, f, indent=2)

    save_plots(merged, roll, cfg.out_dir)

    return merged, per, pooled, roll, {"ols_beta": beta, "ols_t": t}


if __name__ == "__main__":  # manual run helper
    cfg = GdeltCorrConfig(
        csv_path="data/v2tone_last.csv",
        tickers=DEFAULT_TICKERS,
        start=DEFAULT_START,
        end=DEFAULT_END,
        sentiment_col="weighted_tone",
        rolling_z=False,
        out_dir="out/gdelt",
    )
    merged, per, pooled, roll, reg = run_correlation_analysis(cfg)
    print({"pooled": pooled, "ols_beta": reg["ols_beta"], "ols_t": reg["ols_t"]})
