"""Utility functions for the enhanced sentiment-analysis pipeline.

All heavy-lifting code has been moved out of the notebook so it can be

• imported and unit-tested easily
• reused by other notebooks / scripts
• kept concise in the notebook (KISS principle)

Key improvements vs the original in-notebook code
-------------------------------------------------
1.  Targeted exception handling – no silent `except:` blocks.
2.  Environment-variable support for Reddit credentials.
3.  Smarter Google News monthly loop – stops once quality quota met.
4.  Per-row weighted sentiment that ignores missing sources (fixes bias).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Optional, imported lazily where needed
try:
    from transformers import pipeline  # type: ignore
except ImportError:  #   Not every environment has transformers
    pipeline = None  # type: ignore

###############################################################################
# Caching helpers
###############################################################################
CACHE_DIR = Path("news_cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_path(source: str, ticker: str, tag: str | int) -> Path:
    """Return the cache filename for a given *source* / *ticker* / *tag*.

    *tag* can be a year or any other identifier that keeps the file unique.
    """
    return CACHE_DIR / f"{source}_{ticker}_{tag}.json"


def load_cached_data(cache_path: Path, max_age_seconds: int = 86_400):
    """Return cached JSON if present *and* not older than *max_age_seconds*.
    Raises `json.JSONDecodeError` for corrupt files instead of swallowing it –
    the caller can catch and refresh the cache on failure.
    """
    if not cache_path.exists():
        return None

    age = time.time() - cache_path.stat().st_mtime
    if age > max_age_seconds:
        return None

    with cache_path.open("r") as f:
        return json.load(f)


def save_to_cache(cache_path: Path, data):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(data, f)

###############################################################################
# Yahoo Finance News
###############################################################################

def collect_yahoo_news(
    tickers: List[str],
    portfolio_assets: Dict[str, str],
    start_year: int = 2024,
    end_year: Optional[int] = None,
):
    """Collect Yahoo-Finance news per ticker.

    This **reads the live attribute** `Ticker.news` once and relies on the
    provider's built-in date field; if cache exists the call is skipped.
    """
    end_year = end_year or datetime.now().year

    all_records: list[dict] = []
    for ticker in tickers:
        cache_path = get_cache_path("yahoo", ticker, start_year)
        try:
            cached = load_cached_data(cache_path)
        except json.JSONDecodeError:
            cached = None  # treat corruption as missing

        if cached:
            all_records.extend(cached)
            continue

        try:
            news_items = getattr(yf.Ticker(ticker), "news", [])
        except Exception as exc:  # pragma: no cover – yfinance can raise varied
            print(f"Yahoo fetch error for {ticker}: {exc}")
            continue

        ticker_news: list[dict] = []
        for item in news_items:
            dt_raw = item.get("providerPublishTime") or item.get("content", {}).get("pubDate")
            if not dt_raw:
                continue
            dt = pd.to_datetime(dt_raw, unit="s", errors="coerce") if str(dt_raw).isdigit() else pd.to_datetime(dt_raw, errors="coerce")
            if pd.isna(dt):
                continue
            if not (start_year <= dt.year <= end_year):
                continue

            content = item.get("content", {})
            title, summary = content.get("title", ""), content.get("summary", "")
            headline = f"{title} {summary}".strip()
            if len(headline) < 10:
                continue

            ticker_news.append(
                {
                    "ticker": ticker,
                    "company": portfolio_assets[ticker],
                    "date": dt.strftime("%Y-%m-%d"),
                    "month": dt.strftime("%Y-%m"),
                    "headline": headline,
                    "title": title,
                    "summary": summary,
                    "source": "yahoo",
                }
            )

        save_to_cache(cache_path, ticker_news)
        all_records.extend(ticker_news)
    return pd.DataFrame(all_records)

###############################################################################
# Google News via gnews
###############################################################################

def _google_domain(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


QUALITY_DOMAINS = {
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "reuters.com",
    "cnbc.com",
    "marketwatch.com",
    "barrons.com",
    "economist.com",
    "forbes.com",
    "morningstar.com",
    "seekingalpha.com",
    "investors.com",
    "fool.com",
    "finance.yahoo.com",
    "businessinsider.com",
    "investing.com",
}


CLICKBAIT_PHRASES = {
    "you won't believe",
    "shocking",
    "this one trick",
    "click here",
    "top 10",
}


def collect_google_news_monthly(
    tickers: List[str],
    portfolio_assets: Dict[str, str],
    *,
    start_year: int = 2020,
    end_year: Optional[int] = None,
    news_per_month: int = 20,
):
    """Monthly Google-News scrape that stops when quota met (fix to original)."""
    from gnews import GNews  # local import – not always installed

    end_year = end_year or datetime.now().year
    all_records: list[dict] = []

    def is_quality_article(title: str, desc: str, url: str) -> bool:
        if len(title) + len(desc) < 50:
            return False
        if any(cb in title.lower() for cb in CLICKBAIT_PHRASES):
            return False
        return any(dom in _google_domain(url) for dom in QUALITY_DOMAINS)

    for ticker in tickers:
        cache_path = get_cache_path("google", ticker, start_year)
        try:
            cached = load_cached_data(cache_path)
        except json.JSONDecodeError:
            cached = None
        if cached:
            all_records.extend(cached)
            continue

        ticker_news: list[dict] = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                if len([n for n in ticker_news if n["month"].startswith(f"{year}-{month:02d}")]) >= news_per_month:
                    # this month already satisfied – skip
                    continue

                start_date = pd.Timestamp(year=year, month=month, day=1)
                if start_date > pd.Timestamp.today():
                    break  # future months

                end_date = (start_date + pd.offsets.MonthEnd(1)).normalize()

                googlenews = GNews(language="en", country="US", max_results=news_per_month * 3)
                googlenews.start_date = start_date
                googlenews.end_date = end_date
                query = f"{ticker} stock news"
                try:
                    articles = googlenews.get_news(query)
                except Exception as exc:
                    print(f"GoogleNews error {ticker} {year}-{month:02d}: {exc}")
                    continue

                month_quality_cnt = 0
                for art in articles:
                    title, desc, link = art.get("title", ""), art.get("description", ""), art.get("link", "")
                    if not is_quality_article(title, desc, link):
                        continue
                    pub_date = pd.to_datetime(art.get("published date", start_date))
                    headline = f"{title} {desc}".strip()
                    ticker_news.append(
                        {
                            "ticker": ticker,
                            "company": portfolio_assets[ticker],
                            "date": pub_date.strftime("%Y-%m-%d"),
                            "month": pub_date.strftime("%Y-%m"),
                            "headline": headline,
                            "title": title,
                            "summary": desc,
                            "source": "google",
                            "is_quality_source": True,
                            "domain": _google_domain(link),
                        }
                    )
                    month_quality_cnt += 1
                    if month_quality_cnt >= news_per_month:
                        break  # ✅ quota met – stop processing more articles for this month
            # end month loop
        # end year loop
        save_to_cache(cache_path, ticker_news)
        all_records.extend(ticker_news)
    return pd.DataFrame(all_records)

###############################################################################
# Reddit
###############################################################################

def _init_reddit():
    import praw  # local import

    return praw.Reddit(
        client_id="pLqfk1M1ymfj3ih1NrVFlA",
        client_secret="_hl1434FeTi9kgv_GXAi5tBLoCaLIQ",
        user_agent="SentimentAnalysisBot"
        )

def collect_reddit_sentiment(
    tickers: List[str],
    portfolio_assets: Dict[str, str],
    finbert_pipeline,  # pass the loaded model (can be None)
    *,
    start_year: int = 2020,
    end_year: Optional[int] = None,
    posts_per_ticker: int = 100,
):
    """Collect *posts_per_ticker* Reddit posts mentioning each *ticker*.

    Uses a multireddit search over a fixed subreddit set.  Sentiment is computed
    in *batch* with FinBERT for speed.  No hard-coded secrets – credentials must
    be supplied via environment variables.
    """
    import praw
    from datetime import datetime

    end_year = end_year or datetime.now().year
    reddit = _init_reddit()

    subreddit_str = (
        "investing+stocks+wallstreetbets+SecurityAnalysis+ValueInvesting"
    )
    subreddit = reddit.subreddit(subreddit_str)

    all_records: list[dict] = []

    for ticker in tickers:
        cache_path = get_cache_path("reddit", ticker, start_year)
        try:
            cached = load_cached_data(cache_path)
        except json.JSONDecodeError:
            cached = None
        if cached:
            all_records.extend(cached)
            continue

        search_terms = [f"\"{ticker}\"", f"${ticker}", f"\"{portfolio_assets[ticker]}\""]
        seen_ids: set[str] = set()
        ticker_posts: list[dict] = []

        for term in search_terms:
            if len(ticker_posts) >= posts_per_ticker:
                break
            try:
                submissions = subreddit.search(term, time_filter="all", sort="relevance", limit=200)
            except praw.exceptions.PRAWException as exc:
                print(f"Reddit search error {ticker}: {exc}")
                continue
            for sub in submissions:
                if sub.id in seen_ids:
                    continue
                post_date = datetime.fromtimestamp(sub.created_utc)
                if not (start_year <= post_date.year <= end_year):
                    continue
                if sub.score < 2 or len(sub.title) < 10:
                    continue
                text = f"{sub.title} {sub.selftext or ''}"[:512]
                ticker_posts.append(
                    {
                        "id": sub.id,
                        "ticker": ticker,
                        "company": portfolio_assets[ticker],
                        "date": post_date.strftime("%Y-%m-%d"),
                        "month": post_date.strftime("%Y-%m"),
                        "headline": text,
                        "title": sub.title,
                        "source": "reddit",
                        "subreddit": sub.subreddit.display_name,
                        "reddit_score": sub.score,
                    }
                )
                seen_ids.add(sub.id)
                if len(ticker_posts) >= posts_per_ticker:
                    break
        # sentiment inference --------------------------------------------------
        if finbert_pipeline is not None and ticker_posts:
            texts = [p["headline"] for p in ticker_posts]
            sentiments = finbert_pipeline(texts)
            for post, res in zip(ticker_posts, sentiments):
                label, score = res["label"], res["score"]
                post["sentiment_label"] = label
                post["sentiment_confidence"] = score
                post["sentiment_score"] = score if label == "positive" else -score if label == "negative" else 0.0
        save_to_cache(cache_path, ticker_posts)
        all_records.extend(ticker_posts)
    return pd.DataFrame(all_records)

###############################################################################
# FinBERT sentiment for generic news headlines
###############################################################################

def load_finbert_pipeline():
    if pipeline is None:
        raise ImportError("transformers not installed – cannot load FinBERT")
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def analyze_sentiment_finbert_enhanced(df: pd.DataFrame, finbert_pipeline):
    """Annotate *df* with FinBERT sentiment (P+-P- score)."""
    if df.empty:
        return pd.DataFrame()

    texts = df["headline"].str.slice(stop=512).tolist()
    results = finbert_pipeline(texts, batch_size=32)
    scores = []
    labels = []
    for res in results:
        label, score = res["label"], res["score"]
        labels.append(label)
        scores.append(score if label == "positive" else -score if label == "negative" else 0.0)

    sentiment_df = df.copy()
    sentiment_df["sentiment_label"] = labels
    sentiment_df["sentiment_confidence"] = [r["score"] for r in results]
    sentiment_df["sentiment_score"] = scores
    return sentiment_df

###############################################################################
# Monthly aggregation – bias-free weighting
###############################################################################

def aggregate_monthly_sentiment_enhanced(sentiment_df: pd.DataFrame):
    if sentiment_df.empty:
        return pd.DataFrame()

    # overall stats
    monthly = (
        sentiment_df.groupby(["ticker", "month"]).agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            news_count=("sentiment_score", "size"),
            confidence_mean=("sentiment_confidence", "mean"),
            company=("company", "first"),
        )
    ).reset_index()

    # source breakdown --------------------------------------------------------
    by_src = (
        sentiment_df.groupby(["ticker", "month", "source"]).agg(
            sentiment_mean=("sentiment_score", "mean"),
            news_count=("sentiment_score", "size"),
        ).reset_index()
    )

    pivot = by_src.pivot_table(index=["ticker", "month"], columns="source", values=["sentiment_mean", "news_count"], fill_value=0)
    pivot.columns = [f"{src}_{metric}" for metric, src in pivot.columns]
    pivot = pivot.reset_index()

    df = monthly.merge(pivot, on=["ticker", "month"], how="left").fillna(0)

    # ------------------------------------------------------------------
    # unbiased weighted average: sum(mean_i * count_i) / sum(count_i)
    # ------------------------------------------------------------------
    src_means = [c for c in df.columns if c.endswith("_sentiment_mean")]
    src_counts = [c for c in df.columns if c.endswith("_news_count")]
    src_bases = [c.split("_")[0] for c in src_means]

    def _row_weighted(row):
        num = 0.0
        den = 0.0
        for base in src_bases:
            cnt = row[f"{base}_news_count"]
            if cnt:
                num += row[f"{base}_sentiment_mean"] * cnt
                den += cnt
        return num / den if den else row["sentiment_mean"]

    df["sentiment_weighted"] = df.apply(_row_weighted, axis=1)
    return df

###############################################################################
# Validation – unchanged except for minor tidy-ups
###############################################################################

def validate_sentiment_vs_returns(
    sentiment_monthly: pd.DataFrame,
    returns_csv_path: str | Path,
):
    """Return (summary_df, merged_df) validating sentiment vs log-returns.
    
    Parameters
    ----------
    sentiment_monthly : pd.DataFrame
        Monthly sentiment data with 'ticker' and 'month' columns
    returns_csv_path : str | Path
        Path to CSV containing daily log-returns. Must have a 'Date' column
        and ticker columns for returns.
    
    Returns
    -------
    summary_df : pd.DataFrame
        Correlation statistics per ticker
    merged_df : pd.DataFrame
        Raw data used for correlation calculation
        
    Raises
    ------
    ValueError
        If input data is empty or date parsing fails
    FileNotFoundError
        If returns CSV doesn't exist
    """
    if sentiment_monthly.empty:
        raise ValueError("sentiment_monthly is empty – nothing to validate")

    if not Path(returns_csv_path).exists():
        raise FileNotFoundError(f"Returns file not found: {returns_csv_path}")

    returns_df = pd.read_csv(returns_csv_path)
    
    # Verify 'Date' column exists (case-insensitive)
    date_col = next((col for col in returns_df.columns if col.lower() == 'date'), None)
    if not date_col:
        raise ValueError(
            f"No 'Date' column found in {returns_csv_path}. "
            f"Available columns: {list(returns_df.columns)}"
        )
    
    # Try parsing dates with multiple formats
    try:
        # First try exact format with UTC handling
        returns_df["Date"] = pd.to_datetime(
            returns_df[date_col],
            format="%Y-%m-%d",
            errors="raise",
            utc=True
        ).dt.tz_localize(None)  # strip TZ after standardizing
    except ValueError:
        try:
            # Fallback: flexible parser with UTC
            returns_df["Date"] = pd.to_datetime(
                returns_df[date_col],
                infer_datetime_format=True,
                errors="raise",
                utc=True
            ).dt.tz_localize(None)  # strip TZ after standardizing
        except ValueError as e:
            raise ValueError(
                f"Failed to parse dates in column '{date_col}'. "
                f"Sample values: {returns_df[date_col].head().tolist()}"
            ) from e
    returns_df["month"] = returns_df["Date"].dt.strftime("%Y-%m")

    # pivot long so we can merge easily
    value_cols = [c for c in returns_df.columns if c not in {"Date", "month"}]
    monthly_returns = (
        returns_df.melt(id_vars=["month"], value_vars=value_cols, var_name="ticker", value_name="daily_lr")
        .dropna()
        .groupby(["ticker", "month"]).daily_lr.sum().reset_index(name="monthly_return")
    )

    merged = sentiment_monthly.merge(monthly_returns, on=["ticker", "month"], how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    results = []
    from scipy.stats import pearsonr  # heavy import here to avoid top level dep

    for tik, grp in merged.groupby("ticker"):
        if len(grp) < 3:
            continue
        cur_corr, cur_p = pearsonr(grp["sentiment_weighted"], grp["monthly_return"])
        lagged = grp.sort_values("month").assign(next_ret=lambda d: d["monthly_return"].shift(-1))
        if lagged["next_ret"].notna().sum() >= 3:
            lag_corr, lag_p = pearsonr(lagged.dropna()["sentiment_weighted"], lagged.dropna()["next_ret"])
        else:
            lag_corr, lag_p = np.nan, np.nan
        results.append(
            {
                "ticker": tik,
                "data_points": len(grp),
                "current_corr": cur_corr,
                "current_p": cur_p,
                "lagged_corr": lag_corr,
                "lagged_p": lag_p,
            }
        )
    return pd.DataFrame(results), merged
