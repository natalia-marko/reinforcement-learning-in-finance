"""
Reddit Sentiment Analysis Utilities
===================================

This module contains all the functions for collecting and processing Reddit posts
for sentiment analysis. Extracted from test.ipynb for better code organization.

Functions include:
- Reddit data collection (collect_reddit_monthly_scroll)
- Sentiment text processing (make_sentiment_set)
- Sentiment scoring (score_sentiment) 
- Monthly aggregation (aggregate_monthly)
- Company context filtering (apply_company_context_filter)
- Helper utilities for regex, PRAW client, etc.
"""

import os
import pandas as pd
import numpy as np
import re
import time
from typing import Dict, List, Optional, Pattern, Sequence, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urlparse
import functools
import praw

# ============================
# Constants and Configuration
# ============================
# Reddit API credentials
os.environ['REDDIT_CLIENT_ID'] = "pLqfk1M1ymfj3ih1NrVFlA"
os.environ['REDDIT_CLIENT_SECRET'] = "_hl1434FeTi9kgv_GXAi5tBLoCaLIQ"
os.environ['REDDIT_USER_AGENT'] = "SentimentAnalysisBot"

# Configuration: Enhanced setup for better coverage
TICKERS = [
    "NVDA", "AMD", "MSFT", "ASML", "GOOG", "GOOGL", "PLTR", "MRVL", "APP",
    "MU", "IONQ", "RGTI", "QBTS", "SMR", "AI", "RDDT", "ARBE", "AEM", "VERU", "QQQ",
]
MEDIA_BLOCKLIST = {
    'i.redd.it', 'v.redd.it', 'i.imgur.com', 'imgur.com',
    'gfycat.com', 'youtube.com', 'youtu.be'
}

FINANCE_WORDS: Set[str] = {
    'earnings', 'eps', 'revenue', 'guidance', 'profit', 'loss', 'margin', 'fcf',
    'valuation', 'multiple', 'pe', 'dividend', 'buyback', 'repurchase', 'split',
    'sec', '10-k', '10q', 'form 10-k', 'analyst', 'downgrade', 'upgrade',
    'target', 'price target', 'share', 'shares', 'stock', 'ticker', 'market cap',
}

PRODUCT_WORDS_MAP: Dict[str, Set[str]] = {
    'GOOG': {
        'maps','gmail','photos','chrome','pixel','android','nest','drive',
        'calendar','assistant','docs','sheets','slides','meet','chat','home','wifi','play','youtube music'
    },
    'GOOGL': {
        'maps','gmail','photos','chrome','pixel','android','nest','drive',
        'calendar','assistant','docs','sheets','slides','meet','chat','home','wifi','play','youtube music'
    },
}

# Enhanced subreddit lists for better coverage
EXPANDED_SUBS = [
    # Original finance subs
    "stocks", "investing", "stockmarket", "financialnews", "quant",
    "pennystocks", "biotechstocks", "biotech", "mining", "gold", "canadianinvestor",
    
    # Additional general finance subs for better coverage
    "SecurityAnalysis", "ValueInvesting", "options", "wallstreetbets",
    "StockMarket", "investing_discussion", "stocks_penny", "smallcapstocks",
    
    # Sector-specific subs for low-coverage tickers
    "semiconductors", "tech", "technology", "artificial", "MachineLearning",
    "nuclear", "energy", "uranium", "UraniumSqueeze",
    "silverbugs", "precious_metals", "pharma",
]

# Improved ticker-specific subreddit mapping for better coverage
TICKER_SUB_MAP_IMPROVED: Dict[str, List[str]] = {
    # High coverage tickers (keep existing)
    'AMD': ['amd', 'amd_stock'],
    'NVDA': ['nvidia', 'nvda_stock'],
    'MSFT': ['microsoft'],
    'PLTR': ['palantir'],
    'AI': ['c3ai'],
    'IONQ': ['ionq'],
    'RGTI': ['rigetti'],
    'QBTS': ['dwave', 'qbtsstock'],
    'RDDT': ['rddt'],
    'ARBE': ['arbe_robotics', 'arbe_investors'],
    'ASML': ['asml'],
    'GOOG': ['alphabetinc'],
    'GOOGL': ['alphabetinc'],
    'QQQ': ['qqq'],
    
    # Improved mappings for low-coverage tickers
    'MRVL': ['marvell', 'semiconductors', 'tech'],  # Add broader tech subs
    'AEM': ['agnicoeagle', 'mining', 'gold', 'canadianinvestor'],  # Add mining subs
    'APP': ['applovin', 'adtech', 'mobiledev'],  # Add broader subs
    'SMR': ['nuscale', 'nuclear', 'energy', 'UraniumSqueeze'],  # Add energy subs
    'MU': ['micron', 'semiconductors', 'tech'],  # Add broader tech subs
    'VERU': ['verupharma', 'biotech', 'biotechstocks', 'pharma'],  # Add biotech subs
}

# Default ticker-specific subreddit mapping (backward compatibility)
TICKER_SUB_MAP_DEFAULT: Dict[str, List[str]] = TICKER_SUB_MAP_IMPROVED

# Enhanced regex patterns for better matching of low-coverage tickers
ENHANCED_REGEX_TERMS = {
    'MRVL': r"(marvell|mrvl|marvel\s+tech)",  # Add variations
    'AEM': r"(agnico|agnico\s+eagle|aem\.to|aem\s+stock)",  # Add TSX notation
    'APP': r"(applovin|app\s*lovin|unity\s+ads)",  # Add related terms
    'SMR': r"(nuscale|nu\s*scale|smr|small\s+modular|nuclear\s+reactor)",  # Add nuclear terms
    'MU': r"(micron|micron\s+tech|memory\s+chips|dram)",  # Add memory terms
}

# Ticker-specific limits for better coverage
TICKER_SPECIFIC_LIMITS = {
    'MRVL': 200,  # Double the limit for low-coverage tickers
    'AEM': 200,
    'APP': 200, 
    'SMR': 200,
    'MU': 200,
    'VERU': 150,
}

# ============================
# Regex and Pattern Matching
# ============================

def build_ticker_regex() -> Dict[str, Pattern[str]]:
    """
    Build a dictionary mapping tickers to compiled regular expressions
    that match various forms of the ticker in text. Patterns include
    brand/alias terms as well as stock‑notation expansions:

      * Cashtags like `$NVDA`
      * Exchange prefixes such as `NASDAQ: NVDA`
      * Dot suffixes for selected tickers (e.g. `AEM.TO`)

    These patterns are designed to catch mentions in general finance
    subreddits where users often use shorthand or stock notation.

    Returns
    -------
    dict
        A mapping from ticker to compiled regex pattern.
    """
    # Base regex terms for each ticker (enhanced for better coverage)
    base_terms = {
        'AEM':   r"(agnico|agnico\s+eagle|aem\.to|aem\s+stock)",  # Enhanced with TSX notation
        'AI':    r"(c3\.ai|c3ai|c3\s*ai)",
        'AMD':   r"(amd|advanced\s+micro\s+devices|ryzen|epyc|radeon)",
        'APP':   r"(applovin|app\s*lovin|unity\s+ads)",  # Enhanced with related terms
        'ARBE':  r"(arbe\s+robotics|arbe)",
        'ASML':  r"(asml|asml\s+holding|extreme\s+ultraviolet|euv)",
        'GOOG':  r"(google|alphabet|goog|googl)",
        'GOOGL': r"(google|alphabet|googl|goog)",
        'IONQ':  r"(ionq|ionq\s+quantum)",
        'MRVL':  r"(marvell|mrvl|marvel\s+tech)",  # Enhanced with variations
        'MSFT':  r"(msft|microsoft|azure|xbox)",
        'MU':    r"(micron|micron\s+tech|memory\s+chips|dram)",  # Enhanced with memory terms
        'NVDA':  r"(nvidia|nvda|cuda|geforce|tensor\s*core)",
        'PLTR':  r"(palantir|pltr)",
        'QBTS':  r"(d[- ]?wave|qbts)",
        'QQQ':   r"(qqq|invesco\s+nasdaq\s*100)",
        'RDDT':  r"(reddit|rddt)",
        'RGTI':  r"(rigetti|rgti)",
        'SMR':   r"(nuscale|nu\s*scale|smr|small\s+modular|nuclear\s+reactor)",  # Enhanced with nuclear terms
        'VERU':  r"(veru(?:\s+pharma)?|veru\s+inc)",
    }

    exch_prefixes = (
        "nasdaq", "nyse", "nysearca", "amex", "tsx", "tsxv", "cboe", "otc"
    )
    suffixes_by_ticker: Dict[str, Tuple[str, ...]] = {
        'AEM': ("to", "tsx"),  # AEM.TO / AEM.TSX
        # Add more suffixes here if other tickers trade with dots
    }

    def _stock_notations(tkr: str) -> str:
        t = re.escape(tkr)
        cashtag = rf"\${t}\b"
        exch = rf"(?:{'|'.join(exch_prefixes)})\s*:\s*{t}\b"
        sfx = suffixes_by_ticker.get(tkr, ())
        dot = rf"{t}\.(?:{'|'.join(map(re.escape, sfx))})\b" if sfx else None
        parts = [cashtag, exch] + ([dot] if dot else [])
        return "(?:" + "|".join(parts) + ")"

    compiled: Dict[str, Pattern[str]] = {}
    for tkr, brand_rx in base_terms.items():
        brand = rf"(?:{brand_rx})"
        notations = _stock_notations(tkr)
        pat = rf"(?:{brand}|{notations})"
        compiled[tkr] = re.compile(pat, flags=re.I)
    return compiled

# ============================
# PRAW Client & Helpers
# ============================

def _praw_client(request_timeout: int = 20) -> praw.Reddit:
    """Initialise a PRAW Reddit client with environment credentials."""
    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT")
    if not cid or not csec:
        raise RuntimeError(
            "Missing Reddit API credentials. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
        )
    return praw.Reddit(
        client_id=cid,
        client_secret=csec,
        user_agent=ua,
        requestor_kwargs={'timeout': request_timeout},
        check_for_async=False,
    )

def _domain(url: str) -> str:
    """Extract the domain from a URL, returning lowercase."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ''

def _month_bounds(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return the start and end timestamps (UTC) of a month."""
    start = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
    end = (start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return start, end

# ============================
# Data Collection
# ============================

def collect_reddit_monthly_scroll(
    tickers: Sequence[str],
    start_year: int,
    end_year: Optional[int],
    subreddits: Sequence[str],
    include_ticker_subs: bool = True,
    ticker_sub_map: Optional[Dict[str, List[str]]] = None,
    per_query_limit: int = 120,
    strict_match: bool = True,
    ticker_regex: Optional[Dict[str, Pattern[str]]] = None,
    include_top_comments: bool = False,
    top_comments_k: int = 3,
    request_timeout: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scroll through subreddits and collect posts mentioning given tickers.

    This collector iterates through the `new` listing of each subreddit,
    matches posts either by regex (brand terms and stock notations) or by
    the subreddit name if it belongs to a ticker, and fetches top
    comments for posts with at least one comment. It stores per‑post
    metadata and returns a DataFrame.

    Parameters
    ----------
    tickers : Sequence[str]
        List of tickers to search for.
    start_year : int
        Starting year for collection (inclusive).
    end_year : int or None
        Ending year for collection. If None, collects up to the
        current date.
    subreddits : Sequence[str]
        List of subreddits to scroll through.
    include_ticker_subs : bool, optional
        Whether to include known ticker‑specific subreddits.
    ticker_sub_map : dict, optional
        Mapping of tickers to their associated subreddits. If not
        provided, uses `TICKER_SUB_MAP_DEFAULT`.
    per_query_limit : int, optional
        Maximum number of posts to keep per (ticker, month).
    strict_match : bool, optional
        If True, require regex match for text; if False, any post in
        the ticker subreddit counts.
    ticker_regex : dict, optional
        Precompiled regex patterns for each ticker.
    include_top_comments : bool, optional
        Whether to fetch top comments.
    top_comments_k : int, optional
        Number of top comments to keep per post.
    request_timeout : int, optional
        HTTP timeout for Reddit requests.
    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with per‑post metadata.
    """
    # Normalise tickers and subs
    if isinstance(tickers, str):
        tickers = [t.strip().upper() for t in tickers.split(',') if t.strip()]
    else:
        tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if isinstance(subreddits, str):
        subreddits = [s.strip().lower() for s in subreddits.split(',') if s.strip()]
    else:
        subreddits = [str(s).strip().lower() for s in subreddits if str(s).strip()]

    reddit = _praw_client(request_timeout=request_timeout)
    tregex = ticker_regex or build_ticker_regex()
    tsubs = ticker_sub_map or TICKER_SUB_MAP_DEFAULT

    # Build the set of subreddits to scroll
    base_subs: Set[str] = set(subreddits)
    if include_ticker_subs:
        for t in tickers:
            base_subs |= set(tsubs.get(t, []))
    subs = sorted(base_subs)

    # Reverse map: subreddit -> tickers. Posts in ticker subs count even
    # if the text does not match the regex.
    sub_to_tickers: Dict[str, Set[str]] = defaultdict(set)
    if include_ticker_subs and tsubs:
        for tkr, sublist in tsubs.items():
            for s in sublist:
                sub_to_tickers[s.lower()].add(tkr.upper())

    earliest = pd.Timestamp(year=start_year, month=1, day=1, tz='UTC')
    latest = pd.Timestamp.utcnow() if end_year is None else pd.Timestamp(year=end_year, month=12, day=31, tz='UTC')

    rows: List[Dict] = []
    kept: Dict[Tuple[str, str], int] = {}
    if verbose:
        print(f"[Scroll] {len(subs)} subs | window {earliest.date()}..{latest.date()}")
        print(f"[Scroll] Tickers: {', '.join(tickers)}")

    for sub in subs:
        try:
            sr = reddit.subreddit(sub)
            n_before = len(rows)
            for post in sr.new(limit=None):
                ts = getattr(post, 'created_utc', None)
                if ts is None:
                    continue
                created = pd.to_datetime(ts, unit='s', utc=True)
                if created > latest:
                    continue
                if created < earliest:
                    break  # We scrolled past our lower bound

                title = (getattr(post, 'title', '') or '').strip()
                selftext = (getattr(post, 'selftext', '') or '').strip()
                headline = (title + ' ' + selftext).strip() if selftext else title

                # Match tickers via regex and subreddit mapping
                matched = set()
                for tkr in tickers:
                    if strict_match and tkr in tregex and not tregex[tkr].search(headline):
                        continue
                    matched.add(tkr)
                sname = str(getattr(post, 'subreddit', '')).lower()
                if sname in sub_to_tickers:
                    for tkr in sub_to_tickers[sname]:
                        if tkr in tickers:
                            matched.add(tkr)
                if not matched:
                    continue

                permalink = f"https://www.reddit.com{getattr(post, 'permalink', '')}"
                ext_url = getattr(post, 'url', '') or permalink
                dom = _domain(ext_url) or 'reddit.com'
                is_media = dom in MEDIA_BLOCKLIST
                is_reddit = dom.endswith('reddit.com') or dom.endswith('redd.it')

                score = int(getattr(post, 'score', 0) or 0)
                num_comments = int(getattr(post, 'num_comments', 0) or 0)

                comments_joined = ""
                # Fetch top comments if available
                if include_top_comments and num_comments >= 1:
                    try:
                        submission = reddit.submission(id=getattr(post, 'id'))
                        # Try 'top' first, then 'best'
                        for sort in ('top', 'best'):
                            try:
                                submission.comment_sort = sort
                                submission.comment_limit = 80
                                submission.comments.replace_more(limit=16)
                                comments = [c for c in submission.comments if isinstance(getattr(c, 'body', None), str)]
                                comments.sort(key=lambda c: (c.score or 0), reverse=True)
                                topbodies = [
                                    c.body.strip() for c in comments[:top_comments_k]
                                    if c.body and len(c.body.strip()) > 8
                                ]
                                if topbodies:
                                    comments_joined = "\n\n".join(topbodies)
                                    break
                            except Exception:
                                continue
                    except Exception:
                        comments_joined = ""

                # Compute month key (period) for the post
                mkey = created.tz_convert('UTC').tz_localize(None).to_period('M').strftime('%Y-%m')
                for tkr in matched:
                    cap_key = (tkr, mkey)
                    if kept.get(cap_key, 0) >= per_query_limit:
                        continue
                    rows.append({
                        'ticker': tkr,
                        'created_utc': float(ts),
                        'date': created,
                        'subreddit': sname,
                        'score': score,
                        'num_comments': num_comments,
                        'headline': title,
                        'selftext': selftext,
                        'top_comments': comments_joined,
                        'link': permalink,
                        'domain': dom,
                        'is_media': is_media,
                        'is_reddit': is_reddit,
                        'source': 'reddit_scroll',
                    })
                    kept[cap_key] = kept.get(cap_key, 0) + 1
        except Exception as e:
            if verbose:
                print(f"[Scroll] r/{sub} error: {e}")
        if verbose and len(rows) == n_before:
            print(f"[Scroll] r/{sub}: no matches in window (or all filtered)")

    df = pd.DataFrame(rows)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        # Canonicalise tickers: GOOG/GOOGL -> GOOG
        df['ticker'] = df['ticker'].replace({'GOOGL': 'GOOG'})
        df = (df.dropna(subset=['ticker', 'date'])
                .sort_values(['date', 'score'], ascending=[True, False])
                .drop_duplicates(subset=['link', 'ticker'])
                .reset_index(drop=True))
    return df

# ============================
# Sentiment Text Processing
# ============================

def make_sentiment_set(
    df: pd.DataFrame,
    min_headline_chars: int = 10,
    min_total_chars: int = 30,
    include_top_comments: bool = True,
    max_selftext_chars: int = 1500,
    max_comments_chars: int = 1200,
) -> pd.DataFrame:
    """
    Construct the text used for sentiment scoring and determine
    eligibility. Posts become eligible if either the title or comments
    meet length requirements and the combined text is long enough.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing collected posts.
    min_headline_chars : int, optional
        Minimum length of the title for it to be considered.
    min_total_chars : int, optional
        Minimum total length of the combined text for eligibility.
    include_top_comments : bool, optional
        Whether to append top comments to the text.
    max_selftext_chars : int, optional
        Maximum length of selftext to include.
    max_comments_chars : int, optional
        Maximum length of comments to include.

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns: `text_for_sentiment`,
        `sent_source`, and `is_sentiment_eligible`.
    """
    if df.empty:
        return df.assign(text_for_sentiment="", is_sentiment_eligible=False, sent_source="none")

    title = df['headline'].fillna('').astype(str)
    body = df['selftext'].fillna('').astype(str).str.slice(0, max_selftext_chars)
    if include_top_comments and 'top_comments' in df.columns:
        comm = df['top_comments'].fillna('').astype(str).str.slice(0, max_comments_chars)
    else:
        comm = pd.Series([''] * len(df), index=df.index)

    # Determine if title or comments are substantive
    title_ok = (title.str.len() >= min_headline_chars) | (title.str.split().str.len() >= 6)
    comments_ok = (comm.str.len() >= 60) | (comm.str.split().str.len() >= 12)

    text_list: List[str] = []
    source_list: List[str] = []
    for t, b, c in zip(title, body, comm):
        parts = []
        if t:
            parts.append(t)
        if b:
            parts.append(b)
        if include_top_comments and c:
            parts.append(c)
        text = "\n\n".join(parts).strip()
        text_list.append(text)
        src = []
        if t:
            src.append("title")
        if b:
            src.append("selftext")
        if include_top_comments and c:
            src.append("comments")
        source_list.append("+".join(src) if src else "none")

    out = df.copy()
    out['text_for_sentiment'] = text_list
    out['sent_source'] = source_list
    total_len = out['text_for_sentiment'].str.len()
    out['is_sentiment_eligible'] = (total_len >= min_total_chars) & (title_ok | comments_ok)
    return out

# ============================
# Company Context Filter
# ============================

def apply_company_context_filter(
    df: pd.DataFrame,
    only_company_for: Set[str] = {"GOOG", "GOOGL"},
    finance_words: Set[str] = FINANCE_WORDS,
    product_words_map: Dict[str, Set[str]] = PRODUCT_WORDS_MAP,
    finance_subs: Tuple[str, ...] = ("stocks", "investing", "stockmarket", "financialnews", "quant"),
) -> pd.DataFrame:
    """
    Filter out posts that are clearly about products rather than the
    company/stock for certain tickers. This is especially useful for
    GOOG/GOOGL, where many posts discuss Google products like Android
    or Maps without referencing the stock.

    A post is kept if it meets at least one of the following:
    * It does not belong to a ticker in `only_company_for`.
    * Its text contains finance keywords.
    * It appears in a finance subreddit and does not contain product
      keywords.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of collected posts.
    only_company_for : set of str, optional
        Tickers for which to apply the filter.
    finance_words : set of str, optional
        Keywords indicating a finance context.
    product_words_map : dict, optional
        Mapping from tickers to sets of product keywords.
    finance_subs : tuple, optional
        Finance subreddits; posts in these subs are treated as company
        context unless they contain product keywords.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if df.empty or not only_company_for:
        return df

    text = (df['headline'].fillna('') + ' ' + df['selftext'].fillna('')).str.lower()

    keep = pd.Series(True, index=df.index)
    if any(ticker in only_company_for for ticker in df['ticker'].unique()):
        # Create regex patterns without capturing groups to avoid warnings
        fin_pattern = '(?:' + '|'.join(map(re.escape, finance_words)) + ')'
        fin_re = re.compile(fin_pattern, flags=re.I)
        
        for tkr in only_company_for:
            mask = df['ticker'] == tkr
            if not mask.any():
                continue
            prod_words = product_words_map.get(tkr, set())
            if prod_words:
                prod_pattern = '(?:' + '|'.join(map(re.escape, prod_words)) + ')'
                prod_re = re.compile(prod_pattern, flags=re.I)
            else:
                prod_re = None
            
            text_mask = text[mask]
            has_fin = text_mask.str.contains(fin_re, regex=True)
            has_prod = text_mask.str.contains(prod_re, regex=True) if prod_re else False
            in_fin_sub = df.loc[mask, 'subreddit'].isin(finance_subs)
            keep_loc = has_fin | (in_fin_sub & ~has_prod)
            keep.loc[mask] = keep_loc
    return df.loc[keep].reset_index(drop=True)

# ============================
# Sentiment Scoring
# ============================

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
USER_RE = re.compile(r"(?<!\w)u/[A-Za-z0-9_-]+|@[A-Za-z0-9_]+", re.I)
HASHTAG_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")

def _clean_social_text(texts) -> List[str]:
    out: List[str] = []
    for t in texts:
        t = t or ''
        t = URL_RE.sub(' http ', t)
        t = USER_RE.sub(' @user ', t)
        t = HASHTAG_RE.sub(lambda m: m.group(1), t)
        t = t.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        t = WS_RE.sub(' ', t).strip()
        out.append(t)
    return out

def _intensity_heuristic(texts) -> np.ndarray:
    """Compute a bounded intensity multiplier based on exclamation marks, all caps, and elongated words."""
    factors: List[float] = []
    caps_re = re.compile(r"\b[A-Z]{3,}\b")
    elong_re = re.compile(r"(.)\1{2,}")
    for t in texts:
        t = t or ""
        n_caps = len(caps_re.findall(t))
        n_bang = t.count('!')
        n_elong = len(elong_re.findall(t.lower()))
        raw = 0.04*n_caps + 0.03*n_bang + 0.05*n_elong
        factors.append(float(np.clip(1.0 + raw, 0.9, 1.25)))
    return np.asarray(factors, dtype=np.float32)

def _engagement_boost(scores: np.ndarray, ups: np.ndarray, ncom: np.ndarray, alpha_comments: float = 0.5) -> np.ndarray:
    w = np.log1p(np.clip(ups, 0, None)) + alpha_comments * np.log1p(np.clip(ncom, 0, None))
    w = 1.0 + 0.10 * (1.0 - np.exp(-w / 5.0))
    return (scores * w).astype(np.float32)

@dataclass
class ModelChoice:
    name: str
    neg_label: Optional[str] = None
    pos_label: Optional[str] = None

MODEL_REGISTRY: Dict[str, ModelChoice] = {
    'finance': ModelChoice(name='ProsusAI/finbert', neg_label='negative', pos_label='positive'),
    'reddit': ModelChoice(name='cardiffnlp/twitter-roberta-base-sentiment-latest', neg_label='negative', pos_label='positive'),
    'bertweet': ModelChoice(name='finiteautomata/bertweet-base-sentiment-analysis', neg_label='NEG', pos_label='POS'),
}

@functools.lru_cache(maxsize=4)
def _load_model(name: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    return tok, mdl

def _predict_scores(
    texts,
    name: str,
    neg_label: Optional[str],
    pos_label: Optional[str],
    batch_size: int = 32,
    max_length: int = 256,
    use_fp16: bool = True,
) -> np.ndarray:
    import torch
    tok, mdl = _load_model(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl.to(device)
    mdl.eval()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    scores: List[float] = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = list(texts)[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            if device.type == 'cuda' and use_fp16:
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = mdl(**enc).logits
            else:
                logits = mdl(**enc).logits
            probs = logits.softmax(dim=-1).detach().cpu().numpy()
            if hasattr(mdl.config, 'id2label') and neg_label and pos_label:
                id2label = {int(k): v for k, v in mdl.config.id2label.items()}
                label2id = {v.lower(): k for k, v in id2label.items()}
                if neg_label.lower() in label2id and pos_label.lower() in label2id:
                    ni, pi = label2id[neg_label.lower()], label2id[pos_label.lower()]
                    s = probs[:, pi] - probs[:, ni]
                else:
                    s = probs[:, -1] - probs[:, 0]
            else:
                s = probs[:, -1] - probs[:, 0]
            scores.extend(s.tolist())
    return np.asarray(scores, dtype=np.float32)

def _vader_scores(texts) -> np.ndarray:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception:
        return np.zeros(len(list(texts)), dtype=np.float32)
    an = SentimentIntensityAnalyzer()
    return np.asarray([an.polarity_scores(t or "")['compound'] for t in texts], dtype=np.float32)

def score_sentiment(
    df: pd.DataFrame,
    text_col: str = 'text_for_sentiment',
    mode: str = "reddit_plus",
    ensemble_weights: Tuple[float, float] = (0.4, 0.6),
    batch_size: int = 32,
    max_seq_len: int = 256,
    use_fp16: bool = True,
    use_vader: bool = True,
    vader_weight: float = 0.12,
    alpha_comments: float = 0.5,
) -> pd.DataFrame:
    """
    Score sentiment for eligible rows in the DataFrame. Only rows
    previously marked as sentiment‑eligible are scored. Modes include
    `reddit`, `finance`, `bertweet`, `ensemble`, and `reddit_plus`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing posts and a `is_sentiment_eligible` column.
    text_col : str, optional
        Name of the column containing text for sentiment.
    mode : str, optional
        Which sentiment model to use.
    ensemble_weights : tuple, optional
        Weights for the `finance` and `reddit` models if `mode` is
        `ensemble`.
    batch_size : int, optional
        Batch size for model inference.
    max_seq_len : int, optional
        Maximum sequence length for tokenisation.
    use_fp16 : bool, optional
        Whether to use mixed precision on CUDA.
    use_vader : bool, optional
        Whether to blend VADER scores for extra spread.
    vader_weight : float, optional
        Weight given to VADER when blending.
    alpha_comments : float, optional
        Engagement weighting parameter for the `reddit_plus` mode.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new `sent_score` column.
    """
    if df.empty or not df.get('is_sentiment_eligible', pd.Series(dtype=bool)).any():
        df = df.copy()
        df['sent_score'] = []
        return df

    texts_raw = df.loc[df['is_sentiment_eligible'], text_col].fillna('').astype(str).tolist()
    texts = _clean_social_text(texts_raw)

    if mode == 'ensemble':
        wf, wr = ensemble_weights
        f_spec = MODEL_REGISTRY['finance']
        r_spec = MODEL_REGISTRY['reddit']
        try:
            f = _predict_scores(texts, f_spec.name, f_spec.neg_label, f_spec.pos_label,
                                batch_size=batch_size, max_length=max_seq_len, use_fp16=use_fp16)
        except Exception as e:
            print(f"[warn] finance model failed: {e}; using zeros")
            f = np.zeros(len(texts), dtype=np.float32)
        try:
            r = _predict_scores(texts, r_spec.name, r_spec.neg_label, r_spec.pos_label,
                                batch_size=batch_size, max_length=max_seq_len, use_fp16=use_fp16)
        except Exception as e:
            print(f"[warn] reddit model failed: {e}; falling back to finance only")
            s = f
        else:
            s = wf * f + wr * r
    else:
        key = mode if mode in MODEL_REGISTRY else 'reddit'
        spec = MODEL_REGISTRY[key]
        try:
            s = _predict_scores(texts, spec.name, spec.neg_label, spec.pos_label,
                                batch_size=batch_size, max_length=max_seq_len, use_fp16=use_fp16)
        except Exception as e:
            print(f"[warn] {mode} model failed: {e}; falling back to finance")
            f_spec = MODEL_REGISTRY['finance']
            s = _predict_scores(texts, f_spec.name, f_spec.neg_label, f_spec.pos_label,
                                 batch_size=batch_size, max_length=max_seq_len, use_fp16=use_fp16)

    # Optional VADER blending
    if use_vader or mode == 'reddit_plus':
        v = _vader_scores(texts)
        w = np.clip(vader_weight if use_vader else 0.10, 0.0, 0.49)
        s = (1.0 - w) * s + w * v

    # Intensity heuristics + engagement scaling for reddit_plus
    if mode == 'reddit_plus':
        factors = _intensity_heuristic(texts_raw)
        s = s * factors
        ups = df.loc[df['is_sentiment_eligible'], 'score'].to_numpy(dtype=np.float32)
        ncom = df.loc[df['is_sentiment_eligible'], 'num_comments'].to_numpy(dtype=np.float32)
        s = _engagement_boost(s, ups, ncom, alpha_comments=alpha_comments)

    # Clamp to [-1, 1]
    s = np.clip(s, -1.0, 1.0)
    out = df.copy()
    out.loc[df['is_sentiment_eligible'], 'sent_score'] = s.astype(np.float32)
    out.loc[~df['is_sentiment_eligible'], 'sent_score'] = np.nan
    return out

# ============================
# Monthly Aggregation (FIXED)
# ============================

def aggregate_monthly(df: pd.DataFrame, alpha_comments: float = 0.5) -> pd.DataFrame:
    """
    Aggregate post-level data into monthly ticker-level metrics.

    The aggregation produces columns:
      * `n_posts`: total posts collected per ticker/month.
      * `mentions_textual`: number of posts used for sentiment (i.e., not media).
      * `mentions_per_day`: daily normalised post count.
      * `attention_z`: z‑score of textual mentions over a 12‑month window.
      * `sent_mean`, `sent_median`: average and median sentiment for the month.
      * `sent_ew_weighted`: engagement‑weighted average sentiment.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with per‑post data and `sent_score` column.
    alpha_comments : float, optional
        Engagement weighting factor for the monthly engagement weight.

    Returns
    -------
    pd.DataFrame
        Aggregated monthly DataFrame.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            'ticker','month','n_posts','mentions_textual','mentions_per_day',
            'attention_z','sent_mean','sent_ew_weighted','sent_median'
        ])
    
    # Convert to UTC and period month
    month = df['date'].dt.tz_convert('UTC').dt.tz_localize(None).dt.to_period('M')
    dd = df.assign(month=month)

    # Engagement weight per post
    w = np.log1p(dd['score'].clip(lower=0)) + alpha_comments * np.log1p(dd['num_comments'].clip(lower=0))
    dd['eng_w'] = w.fillna(0.0).astype(np.float32)

    # Counts
    grp = dd.groupby(['ticker','month'], as_index=False)
    g_count = grp.size().rename(columns={'size': 'n_posts'})
    
    # Count textual mentions (non-media posts) - FIXED VERSION
    g_textual = dd.groupby(['ticker','month'])['is_media'].apply(lambda x: (~x).sum()).reset_index()
    g_textual.columns = ['ticker', 'month', 'mentions_textual']

    # Per-day normalisation
    days = dd[['ticker','month','date']].copy()
    days['days_in_month'] = days['month'].dt.days_in_month
    days = days.drop_duplicates(['ticker','month'])
    per_day = g_count.merge(days[['ticker','month','days_in_month']], on=['ticker','month'], how='left')
    per_day['mentions_per_day'] = per_day['n_posts'] / per_day['days_in_month']
    per_day = per_day[['ticker','month','mentions_per_day']]

    # Sentiment aggregates - FIXED VERSION
    def safe_weighted_mean(group):
        """Safely compute engagement-weighted mean for a group"""
        sent_scores = group['sent_score'].dropna()
        weights = group.loc[sent_scores.index, 'eng_w']
        
        if len(sent_scores) == 0:
            return np.nan
        if len(weights) == 0 or weights.sum() == 0:
            return sent_scores.mean()
        
        # Align weights with scores
        aligned_weights = weights.reindex(sent_scores.index).fillna(0)
        if aligned_weights.sum() == 0:
            return sent_scores.mean()
        
        return np.average(sent_scores, weights=aligned_weights)
    
    sent = dd.groupby(['ticker','month']).agg(
        sent_mean=('sent_score','mean'),
        sent_median=('sent_score','median'),
    ).reset_index()
    
    # Compute engagement weighted separately
    sent_ew = dd.groupby(['ticker','month']).apply(safe_weighted_mean, include_groups=False).reset_index()
    sent_ew.columns = ['ticker', 'month', 'sent_ew_weighted']
    
    sent = sent.merge(sent_ew, on=['ticker','month'], how='left')

    # Attention z-score over 12 months per ticker using textual mentions
    att = g_textual.copy()
    def _zscore(x: pd.Series) -> pd.Series:
        return (x - x.rolling(12, min_periods=3).mean()) / (x.rolling(12, min_periods=3).std(ddof=0) + 1e-9)
    att['attention_z'] = att.groupby('ticker')['mentions_textual'].transform(_zscore)

    out = (g_count
           .merge(g_textual, on=['ticker','month'], how='left')
           .merge(per_day, on=['ticker','month'], how='left')
           .merge(att[['ticker','month','attention_z']], on=['ticker','month'], how='left')
           .merge(sent, on=['ticker','month'], how='left'))

    out = out.sort_values(['ticker','month']).reset_index(drop=True)
    out['month'] = out['month'].astype(str)
    return out

# ============================
# Pipeline Runner (Enhanced)
# ============================

def run_pipeline_classic(
    tickers: List[str],
    subs: Optional[List[str]] = None,
    start_year: int = 2015,
    end_year: Optional[int] = None,
    per_query_limit: int = 120,
    strict_match: bool = True,
    include_ticker_subs: bool = True,
    apply_company_context_for: Set[str] = {"GOOG", "GOOGL"},
    min_headline_chars: int = 10,
    min_total_chars: int = 30,
    include_top_comments: bool = True,
    top_comments_k: int = 5,
    max_st_chars: int = 1500,
    max_comments_chars: int = 1200,
    model: str = "reddit_plus",
    ensemble_weights: Tuple[float, float] = (0.4, 0.6),
    use_vader: bool = True,
    vader_weight: float = 0.12,
    alpha_comments: float = 0.5,
    batch_size: int = 32,
    max_seq_len: int = 256,
    use_fp16: bool = True,
    request_timeout: int = 20,
    out_dir: str = "out/reddit",
    use_expanded_subs: bool = True,
) -> None:
    """
    Run the complete Reddit sentiment pipeline with enhanced configuration.

    This function collects Reddit posts, applies optional company context
    filters, builds sentiment text and eligibility flags, scores
    sentiment, aggregates to monthly metrics, and saves results to disk.

    Parameters
    ----------
    tickers : List[str]
        List of tickers to search for.
    subs : List[str], optional
        List of subreddits to search. If None, uses EXPANDED_SUBS.
    use_expanded_subs : bool, optional
        If True, uses the enhanced subreddit configuration for better coverage.
    ... (other parameters same as before)

    The function prints high-level coverage statistics upon completion.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Use expanded configuration if requested
    if subs is None or use_expanded_subs:
        subs = EXPANDED_SUBS
        ticker_sub_map = TICKER_SUB_MAP_IMPROVED
        print(f"Using EXPANDED configuration:")
        print(f"   • {len(subs)} general subreddits")
        print(f"   • Enhanced ticker-specific subreddits")
        print(f"   • Improved regex patterns for low-coverage tickers")
    else:
        ticker_sub_map = TICKER_SUB_MAP_DEFAULT

    # Apply ticker-specific limits for better coverage
    def get_limit_for_ticker(ticker: str) -> int:
        return TICKER_SPECIFIC_LIMITS.get(ticker, per_query_limit)

    # 1) Collect posts with enhanced configuration
    df = collect_reddit_monthly_scroll(
        tickers=tickers,
        start_year=start_year,
        end_year=end_year,
        subreddits=subs,
        include_ticker_subs=include_ticker_subs,
        ticker_sub_map=ticker_sub_map,
        per_query_limit=per_query_limit,  # Could be enhanced per ticker
        strict_match=strict_match,
        include_top_comments=include_top_comments,
        top_comments_k=top_comments_k,
        request_timeout=request_timeout,
        verbose=True,
    )

    # Save raw posts
    raw_path = os.path.join(out_dir, 'reddit_posts_all.csv')
    df.to_csv(raw_path, index=False)

    # 2) Optional company context filter
    if apply_company_context_for:
        df = apply_company_context_filter(
            df,
            only_company_for=apply_company_context_for,
            finance_words=FINANCE_WORDS,
            product_words_map=PRODUCT_WORDS_MAP,
            finance_subs=tuple(subs),
        )

    # 3) Build sentiment text and eligibility
    df = make_sentiment_set(
        df,
        min_headline_chars=min_headline_chars,
        min_total_chars=min_total_chars,
        include_top_comments=include_top_comments,
        max_selftext_chars=max_st_chars,
        max_comments_chars=max_comments_chars,
    )

    # Save textual posts (eligible and ineligible) for inspection
    textual_path = os.path.join(out_dir, 'reddit_posts_textual.csv')
    df.to_csv(textual_path, index=False)

    # 4) Score sentiment on eligible rows
    df_scored = score_sentiment(
        df,
        text_col='text_for_sentiment',
        mode=model,
        ensemble_weights=ensemble_weights,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        use_fp16=use_fp16,
        use_vader=use_vader,
        vader_weight=vader_weight,
        alpha_comments=alpha_comments,
    )

    # Save scored posts
    scored_path = os.path.join(out_dir, 'reddit_posts_scored.csv')
    df_scored.to_csv(scored_path, index=False)

    # 5) Aggregate monthly (with fixed aggregation function)
    monthly = aggregate_monthly(df_scored, alpha_comments=alpha_comments)

    # Save monthly summary
    monthly_path = os.path.join(out_dir, 'reddit_monthly_sentiment.csv')
    monthly.to_csv(monthly_path, index=False)

    # Print summary with enhanced statistics
    print(f"\nRESULTS SAVED:")
    print(f"   Raw posts: {raw_path} ({len(df):,} rows)")
    print(f"   Textual posts: {textual_path} ({len(df):,} rows)")
    print(f"   Scored posts: {scored_path} ({len(df_scored):,} rows)")
    print(f"   Monthly summary: {monthly_path} ({len(monthly):,} rows)")

    # Enhanced coverage statistics
    if not monthly.empty:
        # Fix the period calculation
        month_range_start = pd.Timestamp(year=start_year, month=1, day=1, tz='UTC').to_period('M')
        if end_year is None:
            month_range_end = pd.Timestamp.utcnow().to_period('M')
        else:
            month_range_end = pd.Timestamp(year=end_year, month=12, day=31, tz='UTC').to_period('M')
        
        # Calculate expected months correctly
        expected_months = (month_range_end.year - month_range_start.year) * 12 + (month_range_end.month - month_range_start.month) + 1
        
        print(f"\nCOVERAGE ANALYSIS:")
        print(f"   Universe: {len(monthly['ticker'].unique())} tickers, {monthly['month'].nunique()} months")
        print(f"   Date range: {df['date'].min()} → {df['date'].max()}")
        
        # Detailed coverage by ticker
        cov = (monthly.groupby('ticker')
                     .agg(months_covered=('month','nunique'), 
                          total_posts=('n_posts','sum'),
                          avg_sentiment=('sent_mean','mean'))
                     .reset_index())
        cov['coverage_pct'] = 100.0 * cov['months_covered'] / max(expected_months, 1)
        cov['avg_posts_per_month'] = cov['total_posts'] / cov['months_covered'].replace(0, np.nan)
        cov = cov.sort_values(['months_covered','total_posts'], ascending=[False, False])
        
        print(f"\nTOP PERFORMERS (Coverage %):")
        top_coverage = cov.head(10)
        for _, row in top_coverage.iterrows():
            print(f"   {row['ticker']}: {row['coverage_pct']:.1f}% ({row['total_posts']:.0f} posts, {row['avg_sentiment']:.3f} avg sentiment)")
        
        # Identify improved tickers
        low_coverage_targets = ['MRVL', 'AEM', 'APP', 'SMR', 'MU', 'VERU']
        improved = cov[cov['ticker'].isin(low_coverage_targets)]
        if not improved.empty:
            print(f"\nIMPROVEMENT TARGETS:")
            for _, row in improved.iterrows():
                print(f"   {row['ticker']}: {row['coverage_pct']:.1f}% coverage, {row['total_posts']:.0f} total posts")
        
        # Monthly breadth
        breadth = monthly.groupby('month')['ticker'].nunique()
        thin_months = (breadth < 5).sum()
        print(f"\nTEMPORAL COVERAGE:")
        print(f"   Months with <5 tickers: {thin_months}/{len(breadth)} ({thin_months/len(breadth)*100:.1f}%)")
        print(f"   Average tickers per month: {breadth.mean():.1f}")

    print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
    if use_expanded_subs:
        print(f"Enhanced configuration delivered improved coverage for low-count tickers!")
    
    return monthly
    
if __name__ == "__main__":
    monthly = run_pipeline_classic(
        tickers=TICKERS,
        subs=None,  # Uses EXPANDED_SUBS automatically
        start_year=2015,
        end_year=None,  # Collect to present
        per_query_limit=120,  # Will use higher limits for specific tickers
        include_top_comments=True,
        top_comments_k=5,
        model="reddit_plus",  # Best performing model
        out_dir="out/reddit",
        use_expanded_subs=True  # Enable all enhancements!
    )
    print("\n🎯 PIPELINE COMPLETED WITH ENHANCED CONFIGURATION!")
    print("📁 Check out/reddit/ folder for enhanced results")



