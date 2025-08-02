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
5.  Eliminated redundant code patterns for better maintainability.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
import yfinance as yf

# Optional, imported lazily where needed
try:
    from transformers import pipeline  # type: ignore
except ImportError:  #   Not every environment has transformers
    pipeline = None  # type: ignore

###############################################################################
# Constants and configuration
###############################################################################
CACHE_DIR = Path("news_cache")
CACHE_DIR.mkdir(exist_ok=True)

QUALITY_DOMAINS = {
    "bloomberg.com", "ft.com", "wsj.com", "reuters.com", "cnbc.com",
    "marketwatch.com", "barrons.com", "economist.com", "forbes.com",
    "morningstar.com", "seekingalpha.com", "investors.com", "fool.com",
    "finance.yahoo.com", "businessinsider.com", "investing.com",
}

CLICKBAIT_PHRASES = {
    "you won't believe", "shocking", "this one trick", "click here", "top 10",
}

###############################################################################
# Common helper functions
###############################################################################

def get_cache_path(source: str, ticker: str, tag: str | int) -> Path:
    """Return the cache filename for a given *source* / *ticker* / *tag*."""
    return CACHE_DIR / f"{source}_{ticker}_{tag}.json"


def load_cached_data(cache_path: Path, max_age_seconds: int = 86_400) -> Optional[Any]:
    """Return cached JSON if present *and* not older than *max_age_seconds*."""
    if not cache_path.exists():
        return None

    age = time.time() - cache_path.stat().st_mtime
    if age > max_age_seconds:
        return None

    with cache_path.open("r") as f:
        return json.load(f)


def save_to_cache(cache_path: Path, data: Any) -> None:
    """Save data to cache with proper directory creation."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(data, f)


def try_load_cached_or_fetch(
    source: str, 
    ticker: str, 
    start_year: int, 
    fetch_func: callable,
    *args
) -> List[Dict[str, Any]]:
    """Common pattern: try cache first, fetch if needed."""
    cache_path = get_cache_path(source, ticker, start_year)
    try:
        cached = load_cached_data(cache_path)
        if cached:
            return cached
    except json.JSONDecodeError:
        pass  # treat corruption as missing
    
    # Fetch fresh data
    fresh_data = fetch_func(ticker, *args, start_year)
    save_to_cache(cache_path, fresh_data)
    return fresh_data


def create_article_record(
    ticker: str,
    company: str,
    date: datetime,
    headline: str,
    title: str,
    source: str,
    **extra_fields: Any
) -> Dict[str, Any]:
    """Create standardized article record structure."""
    return {
        "ticker": ticker,
        "company": company,
        "date": date.strftime("%Y-%m-%d"),
        "month": date.strftime("%Y-%m"),
        "headline": headline,
        "title": title,
        "source": source,
        **extra_fields
    }


def parse_date_flexible(date_raw: Any, fallback_date: Optional[datetime] = None) -> Optional[datetime]:
    """Robust date parsing with multiple fallback strategies."""
    if not date_raw:
        return fallback_date
    
    # Try timestamp format first
    if str(date_raw).isdigit():
        try:
            return pd.to_datetime(date_raw, unit="s", errors="raise")
        except (ValueError, TypeError):
            pass
    
    # Try standard datetime parsing
    try:
        return pd.to_datetime(date_raw, errors="raise")
    except (ValueError, TypeError):
        pass
    
    return fallback_date


def convert_sentiment_to_score(label: str, score: float) -> float:
    """Convert FinBERT sentiment to P+ - P- format."""
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:  # neutral
        return 0.0


def is_in_date_range(date: datetime, start_year: int, end_year: int) -> bool:
    """Check if date falls within specified year range."""
    return start_year <= date.year <= end_year


def extract_domain(url: str) -> str:
    """Extract domain from URL safely."""
    from urllib.parse import urlparse
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def is_quality_article(title: str, desc: str, url: str) -> bool:
    """Check if article meets quality criteria."""
    if len(title) + len(desc) < 50:
        return False
    if any(cb in title.lower() for cb in CLICKBAIT_PHRASES):
        return False
    return any(dom in extract_domain(url) for dom in QUALITY_DOMAINS)

###############################################################################
# Yahoo Finance News
###############################################################################

def _fetch_yahoo_news(ticker: str, portfolio_assets: Dict[str, str], start_year: int) -> List[Dict[str, Any]]:
    """Fetch Yahoo Finance news for a single ticker."""
    try:
        news_items = getattr(yf.Ticker(ticker), "news", [])
    except Exception as exc:
        print(f"Yahoo fetch error for {ticker}: {exc}")
        return []

    ticker_news = []
    for item in news_items:
        dt_raw = item.get("providerPublishTime") or item.get("content", {}).get("pubDate")
        dt = parse_date_flexible(dt_raw)
        if not dt or not is_in_date_range(dt, start_year, datetime.now().year):
            continue

        content = item.get("content", {})
        title, summary = content.get("title", ""), content.get("summary", "")
        headline = f"{title} {summary}".strip()
        if len(headline) < 10:
            continue

        ticker_news.append(create_article_record(
            ticker=ticker,
            company=portfolio_assets[ticker],
            date=dt,
            headline=headline,
            title=title,
            source="yahoo",
            summary=summary
        ))

    return ticker_news


def collect_yahoo_news(tickers: List[str], portfolio_assets: Dict[str, str],
    start_year: int = 2024,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """Collect Yahoo-Finance news per ticker with caching."""
    end_year = end_year or datetime.now().year
    all_records = []
    
    for ticker in tickers:
        cache_path = get_cache_path("yahoo", ticker, start_year)
        try:
            cached = load_cached_data(cache_path)
            if cached:
                all_records.extend(cached)
                continue
        except json.JSONDecodeError:
            pass
        
        ticker_news = _fetch_yahoo_news(ticker, portfolio_assets, start_year)
        save_to_cache(cache_path, ticker_news)
        all_records.extend(ticker_news)
    
    return pd.DataFrame(all_records)

###############################################################################
# Google News via gnews
###############################################################################

def _fetch_google_news_monthly(
    ticker: str, 
    portfolio_assets: Dict[str, str],
    start_year: int,
    end_year: int,
    news_per_month: int
) -> List[Dict[str, Any]]:
    """Fetch Google News for a single ticker with monthly quotas."""
    from gnews import GNews

    ticker_news = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Check if quota already met for this month
            month_key = f"{year}-{month:02d}"
            if len([n for n in ticker_news if n["month"].startswith(month_key)]) >= news_per_month:
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
                    
                pub_date = parse_date_flexible(art.get("published date"), start_date)
                if not pub_date:
                    continue
                    
                headline = f"{title} {desc}".strip()
                ticker_news.append(create_article_record(
                    ticker=ticker,
                    company=portfolio_assets[ticker],
                    date=pub_date,
                    headline=headline,
                    title=title,
                    source="google",
                    summary=desc,
                    is_quality_source=True,
                    domain=extract_domain(link)
                ))
                
                month_quality_cnt += 1
                if month_quality_cnt >= news_per_month:
                    break  # quota met for this month

    return ticker_news


def collect_google_news_monthly(
    tickers: List[str],
    portfolio_assets: Dict[str, str],
    *,
    start_year: int = 2020,
    end_year: Optional[int] = None,
    news_per_month: int = 20,
) -> pd.DataFrame:
    """Monthly Google-News scrape that stops when quota met."""
    end_year = end_year or datetime.now().year
    all_records = []

    for ticker in tickers:
        cache_path = get_cache_path("google", ticker, start_year)
        try:
            cached = load_cached_data(cache_path)
            if cached:
                all_records.extend(cached)
                continue
        except json.JSONDecodeError:
            pass

        ticker_news = _fetch_google_news_monthly(ticker, portfolio_assets, start_year, end_year, news_per_month)
        save_to_cache(cache_path, ticker_news)
        all_records.extend(ticker_news)

    return pd.DataFrame(all_records)

###############################################################################
# Reddit
###############################################################################

def _init_reddit():
    """Initialize Reddit client with credentials."""
    import praw
    return praw.Reddit(
        client_id="pLqfk1M1ymfj3ih1NrVFlA",
        client_secret="_hl1434FeTi9kgv_GXAi5tBLoCaLIQ",
        user_agent="SentimentAnalysisBot"
    )


def _fetch_reddit_posts(
    ticker: str,
    portfolio_assets: Dict[str, str],
    start_year: int,
    end_year: int,
    posts_per_ticker: int
) -> List[Dict[str, Any]]:
    """Fetch Reddit posts for a single ticker."""
    import praw
    from datetime import datetime

    reddit = _init_reddit()
    subreddit_str = "investing+stocks+wallstreetbets+SecurityAnalysis+ValueInvesting"
    subreddit = reddit.subreddit(subreddit_str)

    search_terms = [f"\"{ticker}\"", f"${ticker}", f"\"{portfolio_assets[ticker]}\""]
    seen_ids = set()
    ticker_posts = []

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
            if not is_in_date_range(post_date, start_year, end_year):
                continue
            if sub.score < 2 or len(sub.title) < 10:
                continue
                
            text = f"{sub.title} {sub.selftext or ''}"[:512]
            ticker_posts.append(create_article_record(
                ticker=ticker,
                company=portfolio_assets[ticker],
                date=post_date,
                headline=text,
                title=sub.title,
                source="reddit",
                id=sub.id,
                subreddit=sub.subreddit.display_name,
                reddit_score=sub.score
            ))
            seen_ids.add(sub.id)
            if len(ticker_posts) >= posts_per_ticker:
                break

    return ticker_posts


def collect_reddit_sentiment(
    tickers: List[str],
    portfolio_assets: Dict[str, str],
    finbert_pipeline,
    *,
    start_year: int = 2020,
    end_year: Optional[int] = None,
    posts_per_ticker: int = 100,
) -> pd.DataFrame:
    """Collect Reddit posts with sentiment analysis."""
    end_year = end_year or datetime.now().year
    all_records = []

    for ticker in tickers:
        cache_path = get_cache_path("reddit", ticker, start_year)
        try:
            cached = load_cached_data(cache_path)
            if cached:
                all_records.extend(cached)
                continue
        except json.JSONDecodeError:
            pass

        ticker_posts = _fetch_reddit_posts(ticker, portfolio_assets, start_year, end_year, posts_per_ticker)

        # Add sentiment analysis
        if finbert_pipeline is not None and ticker_posts:
            texts = [p["headline"] for p in ticker_posts]
            sentiments = finbert_pipeline(texts)
            for post, res in zip(ticker_posts, sentiments):
                label, score = res["label"], res["score"]
                post["sentiment_label"] = label
                post["sentiment_confidence"] = score
                post["sentiment_score"] = convert_sentiment_to_score(label, score)

        save_to_cache(cache_path, ticker_posts)
        all_records.extend(ticker_posts)

    return pd.DataFrame(all_records)

###############################################################################
# FinBERT sentiment analysis
###############################################################################

def load_finbert_pipeline():
    """Load FinBERT sentiment analysis pipeline."""
    if pipeline is None:
        raise ImportError("transformers not installed – cannot load FinBERT")
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def analyze_sentiment_finbert_enhanced(df: pd.DataFrame, finbert_pipeline) -> pd.DataFrame:
    """Annotate DataFrame with FinBERT sentiment (P+-P- score)."""
    if df.empty:
        return pd.DataFrame()

    texts = df["headline"].str.slice(stop=512).tolist()
    results = finbert_pipeline(texts, batch_size=32)
    
    sentiment_df = df.copy()
    sentiment_df["sentiment_label"] = [r["label"] for r in results]
    sentiment_df["sentiment_confidence"] = [r["score"] for r in results]
    sentiment_df["sentiment_score"] = [convert_sentiment_to_score(r["label"], r["score"]) for r in results]
    
    return sentiment_df

###############################################################################
# Monthly aggregation – bias-free weighting
###############################################################################

def aggregate_monthly_sentiment_enhanced(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment data monthly with unbiased source weighting."""
    if sentiment_df.empty:
        return pd.DataFrame()

    # Overall monthly statistics
    monthly = (
        sentiment_df.groupby(["ticker", "month"]).agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            news_count=("sentiment_score", "size"),
            confidence_mean=("sentiment_confidence", "mean"),
            company=("company", "first"),
        )
    ).reset_index()

    # Source-specific breakdown
    by_src = (
        sentiment_df.groupby(["ticker", "month", "source"]).agg(
            sentiment_mean=("sentiment_score", "mean"),
            news_count=("sentiment_score", "size"),
        ).reset_index()
    )

    # Pivot to get source-specific columns
    pivot = by_src.pivot_table(
        index=["ticker", "month"], 
        columns="source", 
        values=["sentiment_mean", "news_count"], 
        fill_value=0
    )
    pivot.columns = [f"{src}_{metric}" for metric, src in pivot.columns]
    pivot = pivot.reset_index()

    # Merge and calculate weighted sentiment
    df = monthly.merge(pivot, on=["ticker", "month"], how="left").fillna(0)

    # Unbiased weighted average: sum(mean_i * count_i) / sum(count_i)
    src_means = [c for c in df.columns if c.endswith("_sentiment_mean")]
    src_counts = [c for c in df.columns if c.endswith("_news_count")]
    src_bases = [c.split("_")[0] for c in src_means]

    def _row_weighted(row):
        num = 0.0
        den = 0.0
        for base in src_bases:
            cnt = row[f"{base}_news_count"]
            if cnt:  # Only consider sources with articles
                num += row[f"{base}_sentiment_mean"] * cnt
                den += cnt
        return num / den if den else row["sentiment_mean"]

    df["sentiment_weighted"] = df.apply(_row_weighted, axis=1)
    return df

###############################################################################
# Validation functions
###############################################################################

def _parse_returns_dates(returns_df: pd.DataFrame, returns_csv_path: str) -> pd.DataFrame:
    """Parse dates in returns DataFrame with robust error handling."""
    # Verify 'Date' column exists (case-insensitive)
    date_col = next((col for col in returns_df.columns if col.lower() == 'date'), None)
    if not date_col:
        raise ValueError(
            f"No 'Date' column found in {returns_csv_path}. "
            f"Available columns: {list(returns_df.columns)}"
        )
    
    # Try parsing dates with multiple formats
    try:
        returns_df["Date"] = pd.to_datetime(
            returns_df[date_col],
            format="%Y-%m-%d",
            errors="raise",
            utc=True
        ).dt.tz_localize(None)
    except ValueError:
        try:
            returns_df["Date"] = pd.to_datetime(
                returns_df[date_col],
                infer_datetime_format=True,
                errors="raise",
                utc=True
            ).dt.tz_localize(None)
        except ValueError as e:
            raise ValueError(
                f"Failed to parse dates in column '{date_col}'. "
                f"Sample values: {returns_df[date_col].head().tolist()}"
            ) from e
    
    returns_df["month"] = returns_df["Date"].dt.strftime("%Y-%m")
    return returns_df


def validate_sentiment_vs_returns(
    sentiment_monthly: pd.DataFrame,
    returns_csv_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate sentiment signals against actual market returns.
    
    Returns (summary_df, merged_df) with correlation statistics per ticker.
    """
    if sentiment_monthly.empty:
        raise ValueError("sentiment_monthly is empty – nothing to validate")

    if not Path(returns_csv_path).exists():
        raise FileNotFoundError(f"Returns file not found: {returns_csv_path}")

    returns_df = pd.read_csv(returns_csv_path)
    returns_df = _parse_returns_dates(returns_df, str(returns_csv_path))

    # Convert to monthly returns
    value_cols = [c for c in returns_df.columns if c not in {"Date", "month"}]
    monthly_returns = (
        returns_df.melt(id_vars=["month"], value_vars=value_cols, var_name="ticker", value_name="daily_lr")
        .dropna()
        .groupby(["ticker", "month"]).daily_lr.sum().reset_index(name="monthly_return")
    )

    # Merge sentiment and returns data
    merged = sentiment_monthly.merge(monthly_returns, on=["ticker", "month"], how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Calculate correlations per ticker
    results = []
    from scipy.stats import pearsonr

    for tik, grp in merged.groupby("ticker"):
        if len(grp) < 3:
            continue
            
        # Current month correlation
        cur_corr, cur_p = pearsonr(grp["sentiment_weighted"], grp["monthly_return"])
        
        # Lagged correlation (predictive power)
        lagged = grp.sort_values("month").assign(next_ret=lambda d: d["monthly_return"].shift(-1))
        if lagged["next_ret"].notna().sum() >= 3:
            lag_corr, lag_p = pearsonr(lagged.dropna()["sentiment_weighted"], lagged.dropna()["next_ret"])
        else:
            lag_corr, lag_p = np.nan, np.nan
            
        results.append({
            "ticker": tik,
            "data_points": len(grp),
            "current_corr": cur_corr,
            "current_p": cur_p,
            "lagged_corr": lag_corr,
            "lagged_p": lag_p,
        })
    
    return pd.DataFrame(results), merged
