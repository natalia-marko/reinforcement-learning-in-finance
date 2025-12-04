# --- Authenticated Reddit via PRAW -------------------------------------------
import os
import time
import pandas as pd
from urllib.parse import urlparse


import praw


def _praw_client(client_id=None, client_secret=None, user_agent=None):
    if praw is None:
        raise RuntimeError("praw is not installed. `pip install praw`")

    cid = client_id or os.getenv("REDDIT_CLIENT_ID")
    csec = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
    ua = user_agent or os.getenv("REDDIT_USER_AGENT") or "sentiment-scraper/1.0"
    if not (cid and csec and ua):
        raise RuntimeError("Missing Reddit credentials. "
                           "Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT.")

    return praw.Reddit(
        client_id=cid,
        client_secret=csec,
        user_agent=ua,
        ratelimit_seconds=5,  # basic backoff
    )

def _month_bounds(year: int, month: int):
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = (start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return start, end

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def collect_reddit_monthly_praw(
    tickers,
    asset_map=None,
    start_year=2018,
    end_year=None,
    subreddits=("stocks","investing","StockMarket","FinancialNews","wallstreetbets","Quant"),
    per_query_limit=120,
    client_id=None,
    client_secret=None,
    user_agent=None,
    verbose=True,
):
    """
    Authenticated Reddit scraping using PRAW search().
    We still enforce monthly windows and per-(ticker,month) caps.

    Output columns: ['ticker','date','headline','link','domain','is_quality_source','source']
    """
    reddit = _praw_client(client_id, client_secret, user_agent)
    if end_year is None:
        end_year = pd.Timestamp.utcnow().year
    now = pd.Timestamp.utcnow().normalize()

    rows = []
    for ticker in tickers:
        term = f'"{asset_map[ticker]}"' if asset_map and ticker in asset_map else ticker
        if verbose:
            print(f"[Reddit/PRAW] {ticker} → term={term}")

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                start, end = _month_bounds(year, month)
                if start > now:
                    break

                kept = 0
                for sub in subreddits:
                    if kept >= per_query_limit:
                        break

                    try:
                        # Use 'new' to ease time filtering; API doesn't accept explicit date ranges here.
                        # We'll filter by created_utc client-side.
                        it = reddit.subreddit(sub).search(
                            query=term,
                            sort="new",
                            syntax="lucene",  # allows quoted phrases
                            limit=200,        # oversample; we'll filter and cap
                        )
                        for post in it:
                            created = pd.to_datetime(post.created_utc, unit="s", utc=True)
                            if created < start:
                                # Since we sort by 'new', once we go earlier than month start,
                                # further items will likely be older—break for this subreddit.
                                break
                            if start <= created <= end:
                                title = (post.title or "").strip()
                                selftext = (post.selftext or "").strip() if hasattr(post, "selftext") else ""
                                headline = (title + " " + selftext).strip() if selftext else title
                                permalink = f"https://www.reddit.com{post.permalink}"
                                ext_url = getattr(post, "url", "") or permalink
                                dom = _domain(ext_url) or "reddit.com"

                                rows.append({
                                    "ticker": ticker,
                                    "date": created,
                                    "headline": headline,
                                    "link": permalink,
                                    "domain": dom,
                                    # Mark reddit as traceable but not a premium news outlet:
                                    "is_quality_source": dom.endswith("reddit.com"),
                                    "source": "reddit_auth",
                                })
                                kept += 1
                                if kept >= per_query_limit:
                                    break
                        time.sleep(0.4)  # gentle pacing per subreddit
                    except Exception as e:
                        if verbose:
                            print(f"   [Reddit/PRAW] {sub} {year}-{month:02d} error: {e}")
                        time.sleep(1.0)
                        continue
                # small pause between months
                time.sleep(0.2)

    df = pd.DataFrame(rows)
    if not df.empty:
        # enforce dtypes and simple dedup
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["ticker","date","headline"])
        if "link" in df.columns:
            df = df.drop_duplicates(subset=["link"])
        df = df.sort_values("date").reset_index(drop=True)
    return df
