"""
Utility functions for RL portfolio rebalancing data preparation.
Consolidates data fetching, feature engineering, and preprocessing functions.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import json
import warnings
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_log_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate log returns: ln(P_t / P_{t-n})
    
    Parameters:
    -----------
    series : pd.Series
        Price series
    periods : int
        Number of periods to look back (default: 1)
    
    Returns:
    --------
    pd.Series
        Log returns (NaN for first 'periods' values)
    
    Notes:
    ------
    - Log returns are time-additive: r_total = r1 + r2 + ... + rn
    - For small returns, log returns ≈ simple returns
    - Handles division by zero and negative prices gracefully (returns NaN)
    """
    if periods == 1:
        return np.log(series / series.shift(1))
    else:
        return np.log(series / series.shift(periods))


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1wk'
) -> pd.DataFrame:
    """Fetch weekly price data from Yahoo Finance."""
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False
    )
    
    if data.empty:
        raise ValueError(f"No data fetched for tickers: {tickers}")
    
    return data


def _load_api_key_from_file(key_name: str = 'fred', config_path: Optional[Path] = None) -> Optional[str]:
    """Load API key from api_keys.json file."""
    if config_path is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent
        config_path = project_root / 'api_keys.json'
    
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            keys = json.load(f)
        return keys.get(key_name)
    except (json.JSONDecodeError, IOError):
        return None


def fetch_macro_data(
    start_date: str,
    end_date: str,
    indicators: Optional[dict] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch macroeconomic indicators from FRED.

    Enhanced indicator set captures macro regime dynamics:
    - Inflation regime: CPI, PCE
    - Monetary policy: Fed Funds, 10Y Treasury
    - Credit conditions: IG/HY spreads, BAA-AAA spread
    - Risk sentiment: VIX, equity put/call ratio
    - Economic activity: Unemployment, ISM Manufacturing
    - Dollar strength: Trade-weighted USD index
    - Yield curve: 10Y-2Y spread (inversion signals recession)

    Parameters:
    -----------
    start_date : str
        Start date for data fetch
    end_date : str
        End date for data fetch
    indicators : Optional[dict]
        Custom indicator dict {name: FRED_series_id}. If None, uses default comprehensive set.
    api_key : Optional[str]
        FRED API key. If None, loads from api_keys.json

    Returns:
    --------
    pd.DataFrame
        Macro indicators with datetime index
    """

    from fredapi import Fred
    if api_key is None:
        api_key = _load_api_key_from_file('fred')


    if indicators is None:
        # ENHANCED MACRO INDICATORS (15 high-signal series)
        indicators = {
            # === Interest Rates & Monetary Policy ===
            'treasury_10y': 'DGS10',           # 10Y Treasury yield (risk-free rate)
            'treasury_2y': 'DGS2',             # 2Y Treasury yield (policy expectations)
            'fed_funds_rate': 'FEDFUNDS',      # Fed Funds Rate (current policy)
            'yield_curve_10y2y': 'T10Y2Y',     # 10Y-2Y spread (recession indicator)

            # === Inflation ===
            'cpi': 'CPIAUCSL',                 # CPI (headline inflation, will compute YoY)
            'pce_core_yoy': 'PCEPILFE',        # Core PCE (Fed's preferred measure)

            # === Credit & Risk Spreads ===
            'baa_aaa_spread': 'BAA10Y',        # BAA-AAA spread (credit stress)
            'high_yield_spread': 'BAMLH0A0HYM2', # HY OAS (credit risk premium)

            # === Market Risk & Volatility ===
            'vix': 'VIXCLS',                   # VIX (equity volatility/fear gauge)

            # === Economic Activity ===
            'unemployment_rate': 'UNRATE',      # Unemployment rate (labor market)
            # 'gdp_growth': 'A191RL1Q225SBEA',   # REMOVED: insufficient variation (only 3 values)
            'ism_manufacturing': 'ISRATIO',     # ISM Manufacturing PMI (activity)

            # === Commodities ===
            'oil_wti': 'DCOILWTICO',           # WTI Crude Oil (energy prices)

            # === Dollar Strength ===
            'dxy': 'DTWEXBGS',                 # Trade-weighted USD index (dollar strength)
        }
    
    try:
        fred = Fred(api_key=api_key)
    except Exception as e:
        warnings.warn(f"Failed to initialize FRED client: {e}", UserWarning)
        return pd.DataFrame()
    
    macro_data = pd.DataFrame()
    
    for name, series_id in indicators.items():
        try:
            series = fred.get_series(series_id, start=start_date, end=end_date)
            if not series.empty:
                macro_data[name] = series
        except Exception as e:
            warnings.warn(f"Could not fetch {name} ({series_id}): {e}", UserWarning)
    
    if macro_data.empty:
        return pd.DataFrame()

    macro_data.index = pd.to_datetime(macro_data.index)
    macro_data = macro_data.sort_index()

    # Compute derived indicators
    # CPI YoY log returns (12-month inflation rate)
    if 'cpi' in macro_data.columns:
        macro_data['cpi_yoy'] = calculate_log_returns(macro_data['cpi'], periods=12) * 100
        # Keep both CPI level and YoY log change

    # Don't resample - keep original dates and let merge_macro_data handle alignment
    # This allows proper date matching with price data dates

    return macro_data


def fetch_news_sentiment(
    tickers: List[str],
    days: int = 7,
    api_key: Optional[str] = None
) -> float:
    """
    Fetch news sentiment using NewsAPI + VADER sentiment analysis.

    Requires:
    - pip install newsapi-python nltk
    - nltk.download('vader_lexicon')

    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols to search for
    days : int
        Number of days to look back (default: 7 for weekly)
    api_key : Optional[str]
        NewsAPI key (or load from api_keys.json)

    Returns:
    --------
    float
        Average sentiment score: -1 (bearish) to +1 (bullish)
    """
    try:
        from newsapi import NewsApiClient
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        # Download VADER lexicon if not already present
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)

        sia = SentimentIntensityAnalyzer()

        if api_key is None:
            api_key = _load_api_key_from_file('newsapi')

        if api_key is None:
            warnings.warn("NewsAPI key not found. Skipping news sentiment.", UserWarning)
            return 0.0

        newsapi = NewsApiClient(api_key=api_key)

        articles = []
        for ticker in tickers:
            try:
                res = newsapi.get_everything(
                    q=ticker,
                    language='en',
                    page_size=100,
                    sort_by='publishedAt'
                )
                articles.extend(res.get('articles', []))
            except Exception as e:
                warnings.warn(f"Failed to fetch news for {ticker}: {e}", UserWarning)

        if not articles:
            return 0.0

        # Calculate VADER compound scores for article titles
        scores = [sia.polarity_scores(a['title'])['compound'] for a in articles if a.get('title')]

        return np.mean(scores) if scores else 0.0

    except ImportError:
        warnings.warn(
            "NewsAPI or NLTK not installed. "
            "Install with: pip install newsapi-python nltk",
            UserWarning
        )
        return 0.0
    except Exception as e:
        warnings.warn(f"News sentiment fetch failed: {e}", UserWarning)
        return 0.0


def fetch_twitter_sentiment(
    query: str,
    days: int = 7,
    max_tweets: int = 500
) -> float:
    """
    Fetch Twitter sentiment using snscrape + VADER sentiment analysis.

    Requires:
    - pip install snscrape nltk

    NOTE: Twitter API access is now restricted. Consider alternatives:
    - Reddit API (via PRAW) for r/wallstreetbets, r/stocks sentiment
    - StockTwits API for stock-specific social sentiment
    - FinBERT for more accurate financial sentiment

    Parameters:
    -----------
    query : str
        Search query (e.g., "$AAPL OR $MSFT")
    days : int
        Number of days to look back
    max_tweets : int
        Maximum number of tweets to fetch

    Returns:
    --------
    float
        Average sentiment score: -1 (bearish) to +1 (bullish)
    """
    try:
        import snscrape.modules.twitter as sntwitter
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        # Download VADER lexicon if not already present
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)

        sia = SentimentIntensityAnalyzer()

        tweets = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= max_tweets:
                break
            tweets.append(tweet.content)

        if not tweets:
            return 0.0

        scores = [sia.polarity_scores(t)['compound'] for t in tweets]
        return np.mean(scores) if scores else 0.0

    except ImportError:
        warnings.warn(
            "snscrape or NLTK not installed. "
            "Install with: pip install snscrape nltk",
            UserWarning
        )
        return 0.0
    except Exception as e:
        warnings.warn(f"Twitter sentiment fetch failed: {e}", UserWarning)
        return 0.0


def fetch_sentiment_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    source: str = 'news',  # 'news', 'twitter', or 'combined'
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch sentiment data for tickers from various sources.

    Sentiment captures market mood beyond price:
    - News sentiment: Earnings reports, analyst upgrades, regulatory news
    - Social sentiment: Retail trader sentiment, hype cycles, FUD
    - Combined: Aggregated multi-source sentiment for robustness

    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    start_date : str
        Start date for sentiment data
    end_date : str
        End date for sentiment data
    source : str
        Sentiment source: 'news', 'twitter', or 'combined'
    api_key : Optional[str]
        API key for NewsAPI (if using news sentiment)

    Returns:
    --------
    pd.DataFrame
        Sentiment data with columns: date, ticker, news_sentiment, twitter_sentiment

    Notes:
    ------
    - Sentiment is fetched once per date (not per ticker, as news/social covers market broadly)
    - For ticker-specific sentiment, modify to loop over tickers
    - Consider lagged sentiment (t-1, t-7) for predictive power
    - Resample to weekly by taking mean sentiment over 7-day windows
    """
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')  # Weekly on Fridays

    sentiment_records = []

    for date in dates:
        record = {'date': date}

        if source in ['news', 'combined']:
            record['news_sentiment'] = fetch_news_sentiment(tickers, days=7, api_key=api_key)

        if source in ['twitter', 'combined']:
            # Create query string: "$AAPL OR $MSFT OR ..."
            twitter_query = " OR ".join([f"${t}" for t in tickers])
            record['twitter_sentiment'] = fetch_twitter_sentiment(twitter_query, days=7)

        # Add per-ticker records (if needed for ticker-specific sentiment)
        for ticker in tickers:
            ticker_record = record.copy()
            ticker_record['ticker'] = ticker
            sentiment_records.append(ticker_record)

    sentiment_df = pd.DataFrame(sentiment_records)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    # Add derived features
    if 'news_sentiment' in sentiment_df.columns:
        sentiment_df['news_sentiment_7d_ma'] = sentiment_df.groupby('ticker')['news_sentiment'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()  # 4-week MA for weekly data
        )

    if 'twitter_sentiment' in sentiment_df.columns:
        sentiment_df['twitter_sentiment_7d_ma'] = sentiment_df.groupby('ticker')['twitter_sentiment'].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )

    return sentiment_df


def merge_sentiment_data(
    df: pd.DataFrame,
    sentiment_data: pd.DataFrame,
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Merge sentiment data with price dataframe by date and ticker.

    Parameters:
    -----------
    df : pd.DataFrame
        Price dataframe with 'date' and 'ticker' columns
    sentiment_data : pd.DataFrame
        Sentiment dataframe with 'date', 'ticker', and sentiment columns
    date_col : str
        Date column name (default: 'date')

    Returns:
    --------
    pd.DataFrame
        Merged dataframe with sentiment features
    """
    if sentiment_data.empty:
        warnings.warn("Sentiment data is empty, skipping merge", UserWarning)
        return df.copy()

    result = df.copy()

    if date_col not in result.columns or 'ticker' not in result.columns:
        warnings.warn(
            f"Date column '{date_col}' or 'ticker' not found, skipping sentiment merge",
            UserWarning
        )
        return result

    result[date_col] = pd.to_datetime(result[date_col])
    sentiment_data[date_col] = pd.to_datetime(sentiment_data[date_col])

    # Merge on date and ticker
    result = result.merge(
        sentiment_data,
        on=[date_col, 'ticker'],
        how='left'
    )

    # Forward fill sentiment within each ticker (handles sparse data)
    sentiment_cols = [col for col in sentiment_data.columns
                      if col not in [date_col, 'ticker']]
    for col in sentiment_cols:
        if col in result.columns:
            result[col] = result.groupby('ticker')[col].ffill()
            # Backward fill for any remaining NaN at start
            result[col] = result.groupby('ticker')[col].bfill()
            # Fill any remaining NaN with neutral sentiment (0)
            result[col] = result[col].fillna(0)

    return result


def fetch_benchmark_returns(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = '1wk'
) -> pd.Series:
    """Fetch benchmark returns for a single ticker."""
    data = fetch_price_data(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
        interval=interval
    )

    if data.empty:
        raise ValueError(f"No data fetched for benchmark ticker: {ticker}")

    # Handle both single-level and multi-level column structures
    if data.columns.nlevels == 1:
        # Single ticker - simple columns
        close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        close_prices = data[close_col]
    else:
        # Multi-level columns: (Price_Type, Ticker)
        if ('Adj Close', ticker) in data.columns:
            close_prices = data[('Adj Close', ticker)]
        elif ('Close', ticker) in data.columns:
            close_prices = data[('Close', ticker)]
        else:
            # Fallback: try accessing by price type first
            if 'Adj Close' in data.columns.get_level_values(0):
                close_prices = data['Adj Close'].iloc[:, 0]
            else:
                close_prices = data['Close'].iloc[:, 0]

    # Use log returns for consistency
    returns = calculate_log_returns(close_prices, periods=1)
    returns.index = pd.to_datetime(returns.index)

    return returns


def prepare_price_dataframe(price_data: pd.DataFrame) -> pd.DataFrame:
    """Convert multi-index price data to long format with ticker column. Uses Adjusted Close."""
    if price_data.empty:
        return pd.DataFrame(columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])

    dfs = []

    # Single level columns (single ticker download)
    if price_data.columns.nlevels == 1:
        single_ticker_data = price_data.copy()
        single_ticker_data.columns = [col if isinstance(col, str) else str(col) for col in single_ticker_data.columns]

        ticker_name = 'UNKNOWN'
        if len(single_ticker_data.columns) > 0:
            first_col = single_ticker_data.columns[0]
            if isinstance(first_col, tuple):
                ticker_name = first_col[0] if len(first_col) > 0 else 'UNKNOWN'

        df = single_ticker_data.reset_index()
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})

        close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        if close_col not in df.columns and 'Close' in df.columns:
            close_col = 'Close'

        result_df = pd.DataFrame({
            'date': pd.to_datetime(df['date']),
            'ticker': ticker_name,
            'open': df.get('Open', df.get('open', np.nan)),
            'high': df.get('High', df.get('high', np.nan)),
            'low': df.get('Low', df.get('low', np.nan)),
            'close': df[close_col],
            'volume': df.get('Volume', df.get('volume', np.nan))
        })

        return result_df

    # Multi-level columns (multiple tickers)
    # yfinance structure: Level 0 = Price Type (Open, High, Low, Close, Adj Close, Volume)
    #                     Level 1 = Ticker (AAPL, MSFT, etc.)

    # Get tickers from level 1 (not level 0!)
    tickers = price_data.columns.get_level_values(1).unique()

    for ticker in tickers:
        try:
            # Extract data for this ticker across all price types
            ticker_df = pd.DataFrame({
                'date': price_data.index,
                'open': price_data[('Open', ticker)] if ('Open', ticker) in price_data.columns else np.nan,
                'high': price_data[('High', ticker)] if ('High', ticker) in price_data.columns else np.nan,
                'low': price_data[('Low', ticker)] if ('Low', ticker) in price_data.columns else np.nan,
                'close': price_data[('Adj Close', ticker)] if ('Adj Close', ticker) in price_data.columns else price_data[('Close', ticker)],
                'volume': price_data[('Volume', ticker)] if ('Volume', ticker) in price_data.columns else np.nan
            })

            ticker_df['ticker'] = ticker
            ticker_df['date'] = pd.to_datetime(ticker_df['date'])

            dfs.append(ticker_df)
        except Exception as e:
            warnings.warn(f"Error processing {ticker}: {e}", UserWarning)
            continue

    if not dfs:
        return pd.DataFrame(columns=['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])

    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values(['date', 'ticker']).reset_index(drop=True)

    return result


def merge_macro_data(
    df: pd.DataFrame,
    macro_data: pd.DataFrame,
    date_col: str = 'date'
) -> pd.DataFrame:
    """Merge macroeconomic data with price dataframe by date."""
    if macro_data.empty:
        warnings.warn("Macro data is empty, skipping merge", UserWarning)
        return df.copy()
    
    result = df.copy()
    
    if date_col not in result.columns:
        warnings.warn(f"Date column '{date_col}' not found, skipping macro data merge", UserWarning)
        return result
    
    result[date_col] = pd.to_datetime(result[date_col])
    
    # Get unique dates from price data
    price_dates = pd.Series(result[date_col].unique()).sort_values()
    
    # Prepare macro data: filter to date range and reindex to price dates
    macro_copy = macro_data.copy()
    macro_copy.index = pd.to_datetime(macro_copy.index)
    
    # Filter to price data date range (with 7-day buffer for weekly alignment)
    price_date_min = price_dates.min()
    price_date_max = price_dates.max()
    macro_filtered = macro_copy[
        (macro_copy.index >= price_date_min - pd.Timedelta(days=7)) & 
        (macro_copy.index <= price_date_max)
    ]
    
    if macro_filtered.empty:
        warnings.warn("No macro data in price data date range, skipping merge", UserWarning)
        return result
    
    # Reindex macro data to price dates using forward fill
    # Convert price_dates Series to DatetimeIndex for reindex
    price_dates_index = pd.DatetimeIndex(price_dates)
    
    # Reindex and forward fill each column separately to handle sparse data
    macro_reindexed = pd.DataFrame(index=price_dates_index)
    for col in macro_filtered.columns:
        # Reindex with forward fill
        macro_reindexed[col] = macro_filtered[col].reindex(price_dates_index).ffill()
        # If still NaN at start, backward fill
        if macro_reindexed[col].isna().any():
            macro_reindexed[col] = macro_reindexed[col].bfill()
    
    # Reset index to make date a column
    macro_reindexed = macro_reindexed.reset_index()
    macro_reindexed.columns = [date_col] + list(macro_filtered.columns)
    
    # Merge on date
    result = result.merge(
        macro_reindexed,
        on=date_col,
        how='left'
    )
    
    # Forward fill macro columns within each ticker group (for any remaining NaN)
    macro_cols = list(macro_filtered.columns)
    for col in macro_cols:
        if col in result.columns:
            if 'ticker' in result.columns:
                result[col] = result.groupby('ticker')[col].ffill()
            else:
                result[col] = result[col].ffill()
            # Backward fill for any remaining NaN at the start
            result[col] = result[col].bfill()
    
    return result


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum and trend features."""
    result = df.copy()
    
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_df = df[mask].copy().sort_values('date')
        
        close_series = pd.Series(ticker_df['close'].values, index=ticker_df.index)
        high_series = pd.Series(ticker_df['high'].values, index=ticker_df.index)
        low_series = pd.Series(ticker_df['low'].values, index=ticker_df.index)
        
        # Use log returns for consistency
        returns = calculate_log_returns(close_series, periods=1)
        
        result.loc[mask, 'return_1w'] = returns
        result.loc[mask, 'return_4w'] = calculate_log_returns(close_series, periods=4)
        result.loc[mask, 'return_13w'] = calculate_log_returns(close_series, periods=13)
        result.loc[mask, 'return_26w'] = calculate_log_returns(close_series, periods=26)
        
        # RSI at multiple periods
        for window in [7, 14, 21]:
            if len(ticker_df) >= window + 1:
                rsi = RSIIndicator(close=close_series, window=window)
                result.loc[mask, f'rsi_{window}d'] = rsi.rsi()
            else:
                result.loc[mask, f'rsi_{window}d'] = np.nan

        # ROC at multiple periods
        for period in [1, 4, 13, 26]:
            if len(ticker_df) >= period + 1:
                roc = ROCIndicator(close=close_series, window=period)
                result.loc[mask, f'roc_{period}w'] = roc.roc()
            else:
                result.loc[mask, f'roc_{period}w'] = np.nan

        # MACD components - needs at least 35 data points (slow=26 + signal=9)
        if len(ticker_df) >= 35:
            macd = MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)
            result.loc[mask, 'macd'] = macd.macd()
            result.loc[mask, 'macd_signal'] = macd.macd_signal()
            result.loc[mask, 'macd_histogram'] = macd.macd() - macd.macd_signal()
        else:
            result.loc[mask, 'macd'] = np.nan
            result.loc[mask, 'macd_signal'] = np.nan
            result.loc[mask, 'macd_histogram'] = np.nan

        # Stochastic Oscillator at multiple periods
        for window in [7, 14, 21]:
            if len(ticker_df) >= window + 1:
                stoch = StochasticOscillator(high=high_series, low=low_series, close=close_series, window=window)
                result.loc[mask, f'stochastic_{window}w'] = stoch.stoch()
            else:
                result.loc[mask, f'stochastic_{window}w'] = np.nan
        
        # Price to SMA/EMA ratios
        for period in [4, 8, 12, 20, 26]:
            sma = close_series.rolling(period).mean()
            result.loc[mask, f'price_to_sma_{period}w'] = close_series / (sma + 1e-10)
            
            ema = close_series.ewm(span=period, adjust=False).mean()
            result.loc[mask, f'price_to_ema_{period}w'] = close_series / (ema + 1e-10)
        
        # Momentum indicators
        for period in [1, 4, 13, 26]:
            result.loc[mask, f'momentum_{period}w'] = close_series.diff(period)
        
        # Price position within range
        for period in [4, 13, 26, 52]:
            period_high = high_series.rolling(period).max()
            period_low = low_series.rolling(period).min()
            result.loc[mask, f'price_position_{period}w'] = (close_series - period_low) / (period_high - period_low + 1e-10)
    
    return result


def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility and risk features."""
    result = df.copy()
    
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_df = df[mask].copy().sort_values('date')
        
        close_series = pd.Series(ticker_df['close'].values, index=ticker_df.index)
        high_series = pd.Series(ticker_df['high'].values, index=ticker_df.index)
        low_series = pd.Series(ticker_df['low'].values, index=ticker_df.index)
        
        # Use log returns for consistency
        returns = calculate_log_returns(close_series, periods=1)
        
        for period in [4, 13, 26, 52]:
            vol = returns.rolling(period).std() * np.sqrt(52)
            result.loc[mask, f'volatility_{period}w'] = vol
        
        # Realized volatility (using close-to-close)
        for period in [4, 13, 26]:
            realized_vol = np.sqrt(returns.rolling(period).apply(
                lambda x: np.sum(x**2) * 52 / len(x), raw=False
            ))
            result.loc[mask, f'realized_volatility_{period}w'] = realized_vol

        # ATR - only calculate if we have enough data
        if len(ticker_df) >= 15:  # Need at least window + 1 for ATR
            atr = AverageTrueRange(high=high_series, low=low_series, close=close_series, window=14)
            result.loc[mask, 'atr_14d'] = atr.average_true_range() / close_series
        else:
            result.loc[mask, 'atr_14d'] = np.nan

        # ATR at multiple periods
        for period in [7, 14, 26]:
            if len(ticker_df) >= period + 1:
                atr_multi = AverageTrueRange(high=high_series, low=low_series, close=close_series, window=period)
                result.loc[mask, f'atr_{period}d'] = atr_multi.average_true_range() / close_series
            else:
                result.loc[mask, f'atr_{period}d'] = np.nan

        # Bollinger Bands - only calculate if we have enough data
        if len(ticker_df) >= 20:
            bb = BollingerBands(close=close_series, window=20, window_dev=2)
            bb_high = bb.bollinger_hband()
            bb_low = bb.bollinger_lband()
            result.loc[mask, 'bb_width'] = (bb_high - bb_low) / close_series
            result.loc[mask, 'bb_position'] = (close_series - bb_low) / (bb_high - bb_low + 1e-10)
        else:
            result.loc[mask, 'bb_width'] = np.nan
            result.loc[mask, 'bb_position'] = np.nan

        # Bollinger Bands at multiple periods
        for period in [13, 20, 26]:
            if len(ticker_df) >= period:
                bb_multi = BollingerBands(close=close_series, window=period, window_dev=2)
                bb_high_multi = bb_multi.bollinger_hband()
                bb_low_multi = bb_multi.bollinger_lband()
                result.loc[mask, f'bb_width_{period}w'] = (bb_high_multi - bb_low_multi) / close_series
                result.loc[mask, f'bb_position_{period}w'] = (close_series - bb_low_multi) / (bb_high_multi - bb_low_multi + 1e-10)
            else:
                result.loc[mask, f'bb_width_{period}w'] = np.nan
                result.loc[mask, f'bb_position_{period}w'] = np.nan
        
        for period in [4, 13, 26]:
            # Calculate directional volatilities with fallback to total volatility
            total_vol = returns.rolling(period).std() * np.sqrt(52)

            downside_vol = returns.rolling(period).apply(
                lambda x: x[x < 0].std() * np.sqrt(52) if len(x[x < 0]) > 1 else (x.std() * np.sqrt(52) if len(x) > 1 else 0),
                raw=False
            )
            upside_vol = returns.rolling(period).apply(
                lambda x: x[x > 0].std() * np.sqrt(52) if len(x[x > 0]) > 1 else (x.std() * np.sqrt(52) if len(x) > 1 else 0),
                raw=False
            )
            result.loc[mask, f'downside_volatility_{period}w'] = downside_vol
            result.loc[mask, f'upside_volatility_{period}w'] = upside_vol
            # Volatility ratio: use 1.0 as neutral value when either is near zero
            result.loc[mask, f'volatility_ratio_{period}w'] = np.where(
                downside_vol < 1e-6, 1.0, upside_vol / downside_vol
            )
        
        # Parkinson volatility estimator (using high-low)
        for period in [4, 13, 26]:
            parkinson_vol = np.sqrt(
                (1 / (4 * np.log(2))) * 
                ((np.log(high_series / low_series))**2).rolling(period).mean() * 52
            )
            result.loc[mask, f'parkinson_volatility_{period}w'] = parkinson_vol
    
    return result


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume and liquidity features."""
    result = df.copy()
    
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_df = df[mask].copy().sort_values('date')
        
        close_series = pd.Series(ticker_df['close'].values, index=ticker_df.index)
        high_series = pd.Series(ticker_df['high'].values, index=ticker_df.index)
        low_series = pd.Series(ticker_df['low'].values, index=ticker_df.index)
        volume_series = pd.Series(ticker_df['volume'].values, index=ticker_df.index)
        
        obv = (np.sign(close_series.diff()) * volume_series).cumsum()
        result.loc[mask, 'obv'] = obv
        
        # OBV rate of change at multiple periods
        for period in [1, 4, 13, 26]:
            result.loc[mask, f'obv_roc_{period}w'] = obv.pct_change(period)
        
        # Volume rate of change
        for period in [1, 4, 13, 26]:
            result.loc[mask, f'volume_roc_{period}w'] = volume_series.pct_change(period)
        
        # Relative volume at multiple periods
        for period in [4, 8, 13, 20, 26]:
            vol_ma = volume_series.rolling(period).mean()
            result.loc[mask, f'relative_volume_{period}w'] = volume_series / (vol_ma + 1e-10)
        
        # Volume momentum (change in volume)
        result.loc[mask, 'volume_momentum'] = volume_series.diff()
        result.loc[mask, 'volume_momentum_4w'] = volume_series.diff(4)
        
        # Volume trend (SMA/EMA ratios)
        vol_sma_20 = volume_series.rolling(20).mean()
        vol_ema_20 = volume_series.ewm(span=20, adjust=False).mean()
        result.loc[mask, 'volume_sma_ratio'] = volume_series / (vol_sma_20 + 1e-10)
        result.loc[mask, 'volume_ema_ratio'] = volume_series / (vol_ema_20 + 1e-10)
        
        # Volume-weighted price features
        typical_price = (high_series + low_series + close_series) / 3
        vwap = (typical_price * volume_series).rolling(20).sum() / volume_series.rolling(20).sum()
        result.loc[mask, 'vwap_distance'] = (close_series - vwap) / (vwap + 1e-10)
        
        # Volume-weighted average price momentum (VWAP is price-like, use log returns)
        result.loc[mask, 'vwap_momentum'] = calculate_log_returns(vwap, periods=1)
        result.loc[mask, 'vwap_trend'] = calculate_log_returns(vwap, periods=4)
        
        # Volume profile features
        vol_std = volume_series.rolling(20).std()
        vol_mean = volume_series.rolling(20).mean()
        result.loc[mask, 'volume_std'] = vol_std
        result.loc[mask, 'volume_cv'] = vol_std / (vol_mean + 1e-10)  # Coefficient of variation
        
        # Volume-price divergence
        # Use log returns for price change, keep pct_change for volume (volume is not price-like)
        price_change = calculate_log_returns(close_series, periods=1)
        volume_change = volume_series.pct_change()
        result.loc[mask, 'volume_price_divergence'] = price_change - volume_change
        result.loc[mask, 'volume_price_correlation'] = (
            price_change.rolling(13).corr(volume_change)
        )
        
        # Volume acceleration (volume uses pct_change, not log returns)
        volume_change_1w = volume_series.pct_change(1)
        volume_change_2w = volume_series.pct_change(2)
        result.loc[mask, 'volume_acceleration'] = volume_change_1w - volume_change_2w
        
        # Volume trend strength
        volume_trend = volume_series.rolling(13).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 13 else np.nan,
            raw=False
        )
        result.loc[mask, 'volume_trend_strength'] = volume_trend
        
        # Volume breakout indicator
        vol_max_20 = volume_series.rolling(20).max()
        result.loc[mask, 'volume_breakout'] = (volume_series > vol_max_20.shift(1)).astype(float)
        
        # Money Flow Index
        mfi = MFIIndicator(high=high_series, low=low_series, close=close_series, volume=volume_series, window=14)
        result.loc[mask, 'mfi_14d'] = mfi.money_flow_index()

        # Chaikin Money Flow
        mf_multiplier = ((close_series - low_series) - (high_series - close_series)) / (high_series - low_series + 1e-10)
        mf_volume = mf_multiplier * volume_series
        result.loc[mask, 'chaikin_money_flow'] = (
            mf_volume.rolling(20).sum() / (volume_series.rolling(20).sum() + 1e-10)
        )
        
        # Volume oscillator
        vol_short = volume_series.rolling(5).mean()
        vol_long = volume_series.rolling(20).mean()
        result.loc[mask, 'volume_oscillator'] = (vol_short - vol_long) / (vol_long + 1e-10)
        
        # Additional volume indicators
        # Volume-weighted MACD
        vw_macd = (typical_price * volume_series).ewm(span=12, adjust=False).mean() - \
                  (typical_price * volume_series).ewm(span=26, adjust=False).mean()
        result.loc[mask, 'volume_weighted_macd'] = vw_macd
        
        # Volume rate of change momentum
        result.loc[mask, 'volume_roc_momentum'] = volume_series.pct_change().diff()
    
    return result


def calculate_risk_adjusted_features(df: pd.DataFrame, benchmark_returns: Optional[pd.Series] = None) -> pd.DataFrame:
    """Calculate risk-adjusted performance features."""
    result = df.copy()

    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_df = df[mask].copy().sort_values('date')

        close_series = pd.Series(ticker_df['close'].values, index=ticker_df.index)
        # Use log returns for consistency
        returns = calculate_log_returns(close_series, periods=1)

        for period in [4, 13, 26, 52]:
            vol = returns.rolling(period).std() * np.sqrt(52)
            mean_return = returns.rolling(period).mean() * 52

            sharpe = mean_return / (vol + 1e-10)
            result.loc[mask, f'sharpe_{period}w'] = sharpe

            downside_vol = returns.rolling(period).apply(
                lambda x: x[x < 0].std() * np.sqrt(52), raw=False
            )
            sortino = mean_return / (downside_vol + 1e-10)
            result.loc[mask, f'sortino_{period}w'] = sortino

            rolling_max = close_series.rolling(period).max()
            max_dd = (close_series - rolling_max) / (rolling_max + 1e-10)
            calmar = mean_return / (abs(max_dd.rolling(period).min()) + 1e-10)
            result.loc[mask, f'calmar_{period}w'] = calmar

        if benchmark_returns is not None:
            # Create a series with datetime index for proper alignment
            dates = pd.to_datetime(ticker_df['date'].values)
            returns_with_dates = pd.Series(returns.values, index=dates)

            # Align benchmark returns to the same dates
            aligned_bench = benchmark_returns.reindex(dates, method='ffill')

            rolling_corr = returns_with_dates.rolling(60).corr(aligned_bench)
            result.loc[mask, 'correlation_to_qqq'] = rolling_corr.values

            rolling_cov = returns_with_dates.rolling(60).cov(aligned_bench)
            bench_var = aligned_bench.rolling(60).var()
            result.loc[mask, 'beta_to_market'] = (rolling_cov / (bench_var + 1e-10)).values

            result.loc[mask, 'relative_return'] = (returns_with_dates - aligned_bench).values

    return result


def calculate_drawdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate drawdown and path risk features."""
    result = df.copy()
    
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_df = df[mask].copy().sort_values('date')
        
        close_series = pd.Series(ticker_df['close'].values, index=ticker_df.index)
        # Returns calculated but not used in drawdown features (kept for potential future use)
        returns = calculate_log_returns(close_series, periods=1)
        
        rolling_max = close_series.expanding().max()
        current_dd = (close_series - rolling_max) / (rolling_max + 1e-10)
        result.loc[mask, 'current_drawdown'] = current_dd
        
        # Drawdown duration (consecutive periods in drawdown)
        in_drawdown = (current_dd < 0).astype(int)
        drawdown_duration = in_drawdown.groupby((in_drawdown != in_drawdown.shift()).cumsum()).cumsum()
        result.loc[mask, 'drawdown_duration'] = drawdown_duration
        
        # Drawdown features using expanding max (true drawdown from all-time high)
        running_max = close_series.expanding().max()
        drawdown_series = (close_series - running_max) / (running_max + 1e-10)

        for period in [4, 13, 26, 52]:
            # Max drawdown in period = most negative value in rolling window
            result.loc[mask, f'max_drawdown_{period}w'] = drawdown_series.rolling(period).min()

            # Average drawdown in period = mean drawdown in rolling window
            result.loc[mask, f'avg_drawdown_{period}w'] = drawdown_series.rolling(period).mean()

            # Drawdown volatility in period
            result.loc[mask, f'drawdown_volatility_{period}w'] = drawdown_series.rolling(period).std()

            # Recovery factor: ratio of potential gain to potential loss
            period_max = close_series.rolling(period).max()
            period_min = close_series.rolling(period).min()
            recovery = (period_max - close_series) / (close_series - period_min + 1e-10)
            result.loc[mask, f'recovery_factor_{period}w'] = recovery
        
        # Ulcer Index at multiple periods (using drawdown_series from expanding max)
        for period in [13, 26, 52]:
            squared_dd = (drawdown_series ** 2).rolling(period).mean()
            result.loc[mask, f'ulcer_index_{period}w'] = np.sqrt(squared_dd)

        # Pain Index at multiple periods (mean absolute drawdown, avoiding double rolling)
        for period in [13, 26, 52]:
            pain = abs(drawdown_series).rolling(period).mean()
            result.loc[mask, f'pain_index_{period}w'] = pain
        
        # Maximum adverse excursion (MAE) and maximum favorable excursion (MFE)
        for period in [4, 13, 26]:
            period_max = close_series.rolling(period).max()
            period_min = close_series.rolling(period).min()
            mae = (close_series - period_max) / (period_max + 1e-10)  # Worst case
            mfe = (close_series - period_min) / (period_min + 1e-10)  # Best case
            result.loc[mask, f'mae_{period}w'] = mae
            result.loc[mask, f'mfe_{period}w'] = mfe
    
    return result


def calculate_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistical and distribution features."""
    result = df.copy()
    
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_df = df[mask].copy().sort_values('date')
        
        close_series = pd.Series(ticker_df['close'].values, index=ticker_df.index)
        # Use log returns for consistency
        returns = calculate_log_returns(close_series, periods=1)
        
        result.loc[mask, 'skew_13w'] = returns.rolling(13).skew()
        result.loc[mask, 'skew_26w'] = returns.rolling(26).skew()
        result.loc[mask, 'kurt_13w'] = returns.rolling(13).kurt()
        result.loc[mask, 'kurt_26w'] = returns.rolling(26).kurt()
        
        autocorr_1w = returns.rolling(13).apply(lambda x: x.autocorr(lag=1), raw=False)
        result.loc[mask, 'autocorr_1w'] = autocorr_1w
        
        autocorr_4w = returns.rolling(26).apply(lambda x: x.autocorr(lag=4), raw=False)
        result.loc[mask, 'autocorr_4w'] = autocorr_4w
    
    return result


def calculate_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate calendar and cyclical features.

    Note: sin_dow and cos_dow are excluded for weekly data as they are constant
    (all weekly data points fall on the same day of week, typically Friday).
    """
    result = df.copy()

    if 'date' in result.columns:
        dates = pd.to_datetime(result['date'])
    else:
        dates = pd.to_datetime(result.index)

    result['sin_month'] = np.sin(2 * np.pi * dates.dt.month / 12)
    result['cos_month'] = np.cos(2 * np.pi * dates.dt.month / 12)
    # Skip sin_dow/cos_dow for weekly data - they're constant (same day each week)
    # result['sin_dow'] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
    # result['cos_dow'] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
    result['sin_dom'] = np.sin(2 * np.pi * dates.dt.day / 31)
    result['cos_dom'] = np.cos(2 * np.pi * dates.dt.day / 31)

    return result


def calculate_regime_and_concentration_features(
    df: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None
) -> pd.DataFrame:
    """Calculate regime indicators, relative momentum, and market concentration features.
    
    Features:
    - bull_market: (vix < 20) & (returns_52w > 0)
    - bear_market: (vix > 30) | (returns_52w < -0.1)
    - relative_momentum: asset_return - benchmark_return
    - market_concentration: Herfindahl index of market cap weights (using price as proxy)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with price data and macro indicators
    benchmark_returns : Optional[pd.Series]
        Benchmark returns for relative momentum calculation
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with regime and concentration features added
    """
    result = df.copy()
    
    # Calculate 52-week returns for each ticker
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_df = df[mask].copy().sort_values('date')
        
        close_series = pd.Series(ticker_df['close'].values, index=ticker_df.index)
        returns_52w = calculate_log_returns(close_series, periods=52)
        result.loc[mask, 'returns_52w'] = returns_52w
        
        # Calculate asset return (1-week log return)
        asset_return = calculate_log_returns(close_series, periods=1)
        
        # Relative momentum: asset_return - benchmark_return
        if benchmark_returns is not None:
            dates = pd.to_datetime(ticker_df['date'].values)
            asset_return_with_dates = pd.Series(asset_return.values, index=dates)
            aligned_bench = benchmark_returns.reindex(dates, method='ffill')
            result.loc[mask, 'relative_momentum'] = (asset_return_with_dates - aligned_bench).values
        else:
            result.loc[mask, 'relative_momentum'] = np.nan
    
    # Regime indicators (using VIX if available)
    if 'vix' in result.columns:
        # Bull market: low VIX and positive 52-week returns
        result['bull_market'] = ((result['vix'] < 20) & (result['returns_52w'] > 0)).astype(float)
        
        # Bear market: high VIX or significant negative 52-week returns
        result['bear_market'] = ((result['vix'] > 30) | (result['returns_52w'] < -0.1)).astype(float)
    else:
        # If VIX not available, use returns_52w only
        result['bull_market'] = (result['returns_52w'] > 0).astype(float)
        result['bear_market'] = (result['returns_52w'] < -0.1).astype(float)
    
    # Market concentration: Herfindahl index using price-weighted market share
    # Since we don't have market cap data, we use price as a proxy
    # Calculate concentration per date across all tickers
    result = result.sort_values(['date', 'ticker'])
    
    # Group by date and calculate Herfindahl index
    for date in result['date'].unique():
        date_mask = result['date'] == date
        date_data = result[date_mask]
        
        if len(date_data) > 1:
            # Use price as proxy for market cap
            prices = date_data['close'].values
            total_price = prices.sum()
            
            if total_price > 0:
                # Market share weights
                weights = prices / total_price
                # Herfindahl index: sum of squared weights
                herfindahl = np.sum(weights ** 2)
                result.loc[date_mask, 'market_concentration'] = herfindahl
            else:
                result.loc[date_mask, 'market_concentration'] = np.nan
        else:
            result.loc[date_mask, 'market_concentration'] = np.nan
    
    return result


def engineer_all_features(
    df: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    macro_data: Optional[pd.DataFrame] = None,
    sentiment_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculate all feature categories.

    Parameters:
    -----------
    df : pd.DataFrame
        Price dataframe with OHLCV data
    benchmark_returns : Optional[pd.Series]
        Benchmark returns for correlation/beta calculations
    macro_data : Optional[pd.DataFrame]
        Macroeconomic indicators from FRED
    sentiment_data : Optional[pd.DataFrame]
        Sentiment data (news, social media)

    Returns:
    --------
    pd.DataFrame
        Dataframe with all engineered features
    """
    result = df.copy()

    # Merge external data sources
    if macro_data is not None and not macro_data.empty:
        result = merge_macro_data(result, macro_data)

    if sentiment_data is not None and not sentiment_data.empty:
        result = merge_sentiment_data(result, sentiment_data)

    # Calculate price-based features
    result = calculate_momentum_features(result)
    result = calculate_volatility_features(result)
    result = calculate_volume_features(result)
    result = calculate_risk_adjusted_features(result, benchmark_returns)
    result = calculate_drawdown_features(result)
    result = calculate_statistical_features(result)
    result = calculate_calendar_features(result)
    
    # Calculate regime and concentration features (requires macro data and benchmark returns)
    result = calculate_regime_and_concentration_features(result, benchmark_returns)

    return result


# ============================================================================
# PREPROCESSING
# ============================================================================

def remove_constant_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_variance: float = 1e-8
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove features with zero or near-zero variance."""
    result = df.copy()
    available_cols = [col for col in feature_cols if col in result.columns]
    
    constant_features = []
    for col in available_cols:
        if result[col].nunique() <= 1:
            constant_features.append(col)
        elif result[col].std() < min_variance:
            constant_features.append(col)
    
    if constant_features:
        result = result.drop(columns=constant_features)
        warnings.warn(f"Removed {len(constant_features)} constant features: {constant_features}", UserWarning)
    
    return result, constant_features


def remove_outliers(df: pd.DataFrame, feature_cols: List[str], clip_percentiles: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
    """Remove outliers by clipping values at specified percentiles."""
    result = df.copy()
    
    for col in feature_cols:
        if col in result.columns:
            lower = result[col].quantile(clip_percentiles[0])
            upper = result[col].quantile(clip_percentiles[1])
            result[col] = result[col].clip(lower=lower, upper=upper)
    
    return result


def remove_highly_correlated_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.95
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features by keeping the feature with higher variance.

    For each pair of features with |correlation| > threshold, removes the feature
    with lower variance (less information content).
    """
    result = df.copy()
    available_cols = [col for col in feature_cols if col in result.columns]

    if len(available_cols) < 2:
        return result, []

    corr_matrix = result[available_cols].corr().abs()

    # Find all correlated pairs
    correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                correlated_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    # For each pair, keep the feature with higher variance
    to_remove = set()
    for feat1, feat2, corr_val in correlated_pairs:
        # Skip if either feature already marked for removal
        if feat1 in to_remove or feat2 in to_remove:
            continue

        var1 = result[feat1].var()
        var2 = result[feat2].var()

        # Remove the feature with lower variance
        if var1 < var2:
            to_remove.add(feat1)
        else:
            to_remove.add(feat2)

    to_remove_list = list(to_remove)

    if to_remove_list:
        result = result.drop(columns=to_remove_list)
        warnings.warn(
            f"Removed {len(to_remove_list)} highly correlated features (threshold={threshold}): "
            f"{to_remove_list[:10]}{'...' if len(to_remove_list) > 10 else ''}",
            UserWarning
        )

    return result, to_remove_list

def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    date_col: str = 'date'
) -> Dict[str, pd.DataFrame]:
    """Split data into train and test sets based on date."""
    if date_col in df.columns:
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        dates = df_sorted[date_col].unique()
    else:
        df_sorted = df.sort_index()
        dates = df_sorted.index.unique()
    
    n_dates = len(dates)
    split_idx = int(n_dates * train_ratio)
    
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    
    if date_col in df_sorted.columns:
        train_df = df_sorted[df_sorted[date_col].isin(train_dates)].copy()
        test_df = df_sorted[df_sorted[date_col].isin(test_dates)].copy()
    else:
        train_df = df_sorted[df_sorted.index.isin(train_dates)].copy()
        test_df = df_sorted[df_sorted.index.isin(test_dates)].copy()
    
    return {'train': train_df, 'test': test_df}


def normalize_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Normalize features using z-score scaling fitted on training data only."""
    train_result = train_df.copy()
    test_result = test_df.copy()
    
    available_cols = [col for col in feature_cols if col in train_result.columns]
    
    if len(available_cols) == 0:
        return train_result, test_result, None
    
    if scaler is None:
        scaler = StandardScaler()
        train_result[available_cols] = scaler.fit_transform(
            train_result[available_cols].fillna(0)
        )
    else:
        train_result[available_cols] = scaler.transform(
            train_result[available_cols].fillna(0)
        )
    
    test_result[available_cols] = scaler.transform(
        test_result[available_cols].fillna(0)
    )
    
    return train_result, test_result, scaler


def preprocess_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_ratio: float = 0.8,
    clip_percentiles: Tuple[float, float] = (0.01, 0.99),
    corr_threshold: float = 0.95,
    date_col: str = 'date'
) -> Dict[str, any]:
    """
    Complete preprocessing pipeline: LEAK-PROOF version.

    CRITICAL ORDER (prevents data leakage):
    1. Fill NaN values
    2. SPLIT DATA FIRST (chronological split)
    3. Remove constant features (fit on train, apply to test)
    4. Remove outliers (fit percentiles on train, clip test using train percentiles)
    5. Remove correlations (fit on train, remove same features from test)
    6. Normalize (fit scaler on train, transform test)

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    feature_cols : List[str]
        List of feature column names
    train_ratio : float
        Ratio for train/test split (default: 0.8)
    clip_percentiles : Tuple[float, float]
        Percentiles for outlier clipping (default: (0.01, 0.99))
    corr_threshold : float
        Correlation threshold for feature removal (default: 0.95)
    date_col : str
        Date column name for chronological split (default: 'date')

    Returns:
    --------
    Dict with keys: 'train', 'test', 'scaler', 'removed_features'
    """
    if len(df) == 0:
        raise ValueError("Input dataframe is empty.")

    result = df.copy()

    # Step 1: Fill remaining NaN values in feature columns with 0
    available_feature_cols = [col for col in feature_cols if col in result.columns]
    if len(available_feature_cols) > 0:
        result[available_feature_cols] = result[available_feature_cols].fillna(0)

    # Step 2: SPLIT DATA FIRST (prevents leakage)
    splits = split_data(result, train_ratio, date_col)
    train_df = splits['train'].copy()
    test_df = splits['test'].copy()

    if len(train_df) == 0:
        raise ValueError(
            f"Training set is empty after split. "
            f"Input had {len(df)} rows. Check data."
        )

    if len(test_df) == 0:
        raise ValueError(
            f"Test set is empty after split. "
            f"Input had {len(df)} rows. Check data."
        )

    # Step 3: Remove constant features (fit on train only)
    train_df, constant_features_removed = remove_constant_features(train_df, feature_cols)
    # Apply same removal to test
    test_df = test_df.drop(columns=constant_features_removed, errors='ignore')

    # Step 4: Remove outliers (fit percentiles on train only)
    train_percentiles = {}
    for col in feature_cols:
        if col in train_df.columns:
            lower = train_df[col].quantile(clip_percentiles[0])
            upper = train_df[col].quantile(clip_percentiles[1])
            train_percentiles[col] = (lower, upper)
            # Clip train
            train_df[col] = train_df[col].clip(lower=lower, upper=upper)
            # Clip test using TRAIN percentiles (no leakage)
            if col in test_df.columns:
                test_df[col] = test_df[col].clip(lower=lower, upper=upper)

    # Step 5: Remove highly correlated features (fit on train only)
    train_df, removed_features = remove_highly_correlated_features(
        train_df, feature_cols, corr_threshold
    )
    # Apply same removal to test
    test_df = test_df.drop(columns=removed_features, errors='ignore')

    # Step 6: Get final feature columns
    non_feature_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    updated_feature_cols = [
        col for col in train_df.columns
        if col not in non_feature_cols
    ]

    if len(updated_feature_cols) == 0:
        raise ValueError(
            f"No feature columns remaining after preprocessing. "
            f"Original features: {len(feature_cols)}, "
            f"Removed (constant): {len(constant_features_removed)}, "
            f"Removed (correlation): {len(removed_features)}. "
            f"Available columns: {list(train_df.columns)}"
        )

    # Step 7: Normalize (fit on train, transform test)
    train_norm, test_norm, scaler = normalize_features(
        train_df, test_df, updated_feature_cols
    )

    return {
        'train': train_norm,
        'test': test_norm,
        'scaler': scaler,
        'removed_features': removed_features + constant_features_removed,
        'train_percentiles': train_percentiles,  # Store for future use
    }


# ============================================================================
# COMPLETE INTEGRATED PIPELINE (LEAK-PROOF)
# ============================================================================

def complete_data_pipeline(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1wk',
    train_ratio: float = 0.8,
    include_macro: bool = True,
    include_sentiment: bool = False,
    sentiment_source: str = 'news',
    clip_percentiles: Tuple[float, float] = (0.01, 0.99),
    corr_threshold: float = 0.95,
    fred_api_key: Optional[str] = None,
    newsapi_key: Optional[str] = None
) -> Dict[str, any]:
    """
    Complete leak-proof data preparation pipeline.

    CRITICAL ORDER (prevents data leakage):
    1. Fetch raw price data
    2. SPLIT DATA FIRST (chronological split)
    3. Engineer features on train and test SEPARATELY
    4. Merge macro data (same data for both, but no leakage)
    5. Merge sentiment data (optional)
    6. Preprocess: outliers, correlations, normalization (fit on train, apply to test)

    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols to fetch
    start_date : str
        Start date for data (YYYY-MM-DD)
    end_date : str
        End date for data (YYYY-MM-DD)
    interval : str
        Data frequency: '1d', '1wk', '1mo' (default: '1wk')
    train_ratio : float
        Train/test split ratio (default: 0.8)
    include_macro : bool
        Whether to include macro indicators (default: True)
    include_sentiment : bool
        Whether to include sentiment data (default: False)
    sentiment_source : str
        Sentiment source: 'news', 'twitter', or 'combined' (default: 'news')
    clip_percentiles : Tuple[float, float]
        Percentiles for outlier clipping (default: (0.01, 0.99))
    corr_threshold : float
        Correlation threshold for feature removal (default: 0.95)
    fred_api_key : Optional[str]
        FRED API key (if None, loads from api_keys.json)
    newsapi_key : Optional[str]
        NewsAPI key (if None, loads from api_keys.json)

    Returns:
    --------
    Dict with keys:
        - 'train': Training dataframe (normalized)
        - 'test': Test dataframe (normalized)
        - 'scaler': StandardScaler fitted on training data
        - 'removed_features': List of removed feature names
        - 'metadata': Dict with pipeline metadata

    Example Usage:
    --------------
    ```python
    result = complete_data_pipeline(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2020-01-01',
        end_date='2025-01-01',
        include_macro=True,
        include_sentiment=True,
        sentiment_source='combined'
    )

    train_df = result['train']
    test_df = result['test']
    scaler = result['scaler']
    ```
    """
    print("=" * 80)
    print("LEAK-PROOF DATA PREPARATION PIPELINE")
    print("=" * 80)

    # Step 1: Fetch raw price data
    print("\n[1/7] Fetching price data...")
    price_data = fetch_price_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval=interval
    )
    price_df = prepare_price_dataframe(price_data)
    print(f"  ✓ Fetched {len(price_df)} rows for {len(tickers)} tickers")

    # Step 2: SPLIT DATA FIRST (prevents leakage)
    print("\n[2/7] Splitting data (chronological)...")
    splits = split_data(price_df, train_ratio=train_ratio)
    train_prices = splits['train'].copy()
    test_prices = splits['test'].copy()
    print(f"  ✓ Train: {len(train_prices)} rows ({train_prices['date'].min()} to {train_prices['date'].max()})")
    print(f"  ✓ Test:  {len(test_prices)} rows ({test_prices['date'].min()} to {test_prices['date'].max()})")

    # Step 3: Fetch benchmark returns
    print("\n[3/7] Fetching benchmark returns (QQQ)...")
    try:
        benchmark_returns = fetch_benchmark_returns(
            ticker='QQQ',
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        print(f"  ✓ Fetched benchmark returns")
    except Exception as e:
        warnings.warn(f"Failed to fetch benchmark returns: {e}", UserWarning)
        benchmark_returns = None

    # Step 4: Fetch macro data (same for train and test, no leakage)
    macro_data = None
    if include_macro:
        print("\n[4/7] Fetching macro indicators from FRED...")
        try:
            macro_data = fetch_macro_data(
                start_date=start_date,
                end_date=end_date,
                api_key=fred_api_key
            )
            if not macro_data.empty:
                print(f"  ✓ Fetched {len(macro_data.columns)} macro indicators")
            else:
                print("  ⚠ No macro data fetched (check FRED API key)")
        except Exception as e:
            warnings.warn(f"Failed to fetch macro data: {e}", UserWarning)
            print("  ⚠ Macro data fetch failed")
    else:
        print("\n[4/7] Skipping macro data (include_macro=False)")

    # Step 5: Fetch sentiment data (optional)
    sentiment_data = None
    if include_sentiment:
        print(f"\n[5/7] Fetching sentiment data (source: {sentiment_source})...")
        try:
            sentiment_data = fetch_sentiment_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                source=sentiment_source,
                api_key=newsapi_key
            )
            if not sentiment_data.empty:
                print(f"  ✓ Fetched sentiment data")
            else:
                print("  ⚠ Sentiment data empty")
        except Exception as e:
            warnings.warn(f"Failed to fetch sentiment data: {e}", UserWarning)
            print("  ⚠ Sentiment fetch failed")
    else:
        print("\n[5/7] Skipping sentiment data (include_sentiment=False)")

    # Step 6: Engineer features (separately for train and test)
    print("\n[6/7] Engineering features...")
    print("  → Training set...")
    train_featured = engineer_all_features(
        train_prices,
        benchmark_returns=benchmark_returns,
        macro_data=macro_data,
        sentiment_data=sentiment_data
    )
    print("  → Test set...")
    test_featured = engineer_all_features(
        test_prices,
        benchmark_returns=benchmark_returns,
        macro_data=macro_data,
        sentiment_data=sentiment_data
    )
    print(f"  ✓ Engineered {len(train_featured.columns)} columns")

    # Step 7: Preprocess (fit on train, apply to test)
    print("\n[7/7] Preprocessing (outliers, correlations, normalization)...")
    non_feature_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in train_featured.columns if col not in non_feature_cols]

    # Combine train and test for preprocessing pipeline (it will split internally)
    combined = pd.concat([train_featured, test_featured], ignore_index=True)

    preprocessed = preprocess_pipeline(
        df=combined,
        feature_cols=feature_cols,
        train_ratio=len(train_featured) / len(combined),  # Maintain exact split
        clip_percentiles=clip_percentiles,
        corr_threshold=corr_threshold,
        date_col='date'
    )

    train_final = preprocessed['train']
    test_final = preprocessed['test']
    scaler = preprocessed['scaler']
    removed_features = preprocessed['removed_features']

    final_feature_cols = [col for col in train_final.columns if col not in non_feature_cols]

    print(f"  ✓ Original features: {len(feature_cols)}")
    print(f"  ✓ Removed features: {len(removed_features)}")
    print(f"  ✓ Final features: {len(final_feature_cols)}")

    # Metadata
    metadata = {
        'tickers': tickers,
        'start_date': start_date,
        'end_date': end_date,
        'interval': interval,
        'train_ratio': train_ratio,
        'train_start': train_final['date'].min().isoformat(),
        'train_end': train_final['date'].max().isoformat(),
        'test_start': test_final['date'].min().isoformat(),
        'test_end': test_final['date'].max().isoformat(),
        'n_train_samples': len(train_final),
        'n_test_samples': len(test_final),
        'n_features': len(final_feature_cols),
        'feature_cols': final_feature_cols,
        'removed_features': removed_features,
        'include_macro': include_macro,
        'include_sentiment': include_sentiment,
        'sentiment_source': sentiment_source if include_sentiment else None,
    }

    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Train samples: {len(train_final)}")
    print(f"Test samples:  {len(test_final)}")
    print(f"Features:      {len(final_feature_cols)}")
    print("=" * 80 + "\n")

    return {
        'train': train_final,
        'test': test_final,
        'scaler': scaler,
        'removed_features': removed_features,
        'metadata': metadata,
        'train_percentiles': preprocessed.get('train_percentiles'),
    }


# ============================================================================
# DATA VERIFICATION AND VISUALIZATION FUNCTIONS
# ============================================================================

def verify_data_quality(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> None:
    """
    Verify data quality and check for common issues.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    feature_cols : List[str]
        List of feature column names
    """
    print("=" * 80)
    print("VERIFICATION TESTS")
    print("=" * 80)
    
    # Test 1: No temporal overlap (data leakage check)
    print("\n[Test 1] Checking for data leakage (temporal overlap)...")
    train_end = pd.to_datetime(train_df['date'].max())
    test_start = pd.to_datetime(test_df['date'].min())
    
    if train_end < test_start:
        print(f"  ✓ PASS: No overlap")
        print(f"    Train ends:  {train_end.date()}")
        print(f"    Test starts: {test_start.date()}")
        print(f"    Gap:         {(test_start - train_end).days} days")
    else:
        print(f"  ✗ FAIL: Overlap detected!")
        print(f"    Train ends:  {train_end.date()}")
        print(f"    Test starts: {test_start.date()}")
    
    # Test 2: Feature consistency
    print("\n[Test 2] Checking feature consistency...")
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    if train_cols == test_cols:
        print(f"  ✓ PASS: Train and test have same {len(train_cols)} columns")
    else:
        print(f"  ✗ FAIL: Column mismatch")
        print(f"    Train only: {train_cols - test_cols}")
        print(f"    Test only:  {test_cols - train_cols}")
    
    # Test 3: Normalization
    print("\n[Test 3] Checking normalization...")
    train_means = train_df[feature_cols].mean().mean()
    train_stds = train_df[feature_cols].std().mean()
    
    print(f"  Train mean: {train_means:.4f} (target: ~0.00)")
    print(f"  Train std:  {train_stds:.4f} (target: ~1.00)")
    
    if abs(train_means) < 0.1 and abs(train_stds - 1.0) < 0.2:
        print("  ✓ PASS: Training set properly normalized")
    else:
        print("  ⚠ WARNING: Normalization may be off")
    
    # Test 4: Macro indicators
    print("\n[Test 4] Checking macro indicators...")
    macro_keywords = ['treasury', 'fed_funds', 'vix', 'cpi', 'unemployment',
                      'ism', 'yield_curve', 'dxy', 'oil', 'spread', 'pce', 'baa']
    macro_cols = [c for c in train_df.columns if any(kw in c.lower() for kw in macro_keywords)]
    
    print(f"  Found {len(macro_cols)} macro indicator columns:")
    for col in sorted(macro_cols)[:10]:
        print(f"    - {col}")
    if len(macro_cols) > 10:
        print(f"    ... and {len(macro_cols) - 10} more")
    
    if len(macro_cols) >= 10:
        print("  ✓ PASS: Comprehensive macro coverage")
    else:
        print(f"  ⚠ WARNING: Only {len(macro_cols)} macro features (expected 10+)")
    
    # Test 5: Data quality
    print("\n[Test 5] Checking data quality...")
    train_na_pct = train_df[feature_cols].isna().sum().sum() / (len(train_df) * len(feature_cols)) * 100
    test_na_pct = test_df[feature_cols].isna().sum().sum() / (len(test_df) * len(feature_cols)) * 100
    
    print(f"  Train NaN: {train_na_pct:.2f}%")
    print(f"  Test NaN:  {test_na_pct:.2f}%")
    
    if train_na_pct < 1.0 and test_na_pct < 1.0:
        print("  ✓ PASS: Low missing data")
    else:
        print("  ⚠ WARNING: High missing data percentage")


# Plotting functions moved to viz.py
# Import from viz module: from viz import plot_price_data


def categorize_features(
    feature_cols: List[str],
    macro_cols: List[str]
) -> Dict[str, List[str]]:
    """
    Categorize features into groups.
    
    Parameters:
    -----------
    feature_cols : List[str]
        List of all feature column names
    macro_cols : List[str]
        List of macro indicator column names
    
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary mapping category names to feature lists
    """
    feature_categories = {
        'Momentum': [f for f in feature_cols if any(x in f.lower() for x in ['return', 'roc', 'momentum', 'rsi', 'macd', 'stochastic'])],
        'Volatility': [f for f in feature_cols if 'volatility' in f.lower() or 'atr' in f.lower() or 'parkinson' in f.lower()],
        'Volume': [f for f in feature_cols if 'volume' in f.lower() or 'obv' in f.lower() or 'mfi' in f.lower() or 'chaikin' in f.lower()],
        'Risk-Adjusted': [f for f in feature_cols if any(x in f.lower() for x in ['sharpe', 'sortino', 'calmar'])],
        'Drawdown': [f for f in feature_cols if any(x in f.lower() for x in ['drawdown', 'recovery', 'mae', 'mfe'])],
        'Technical': [f for f in feature_cols if any(x in f.lower() for x in ['bb_', 'sma', 'ema', 'vwap', 'price_to', 'price_position'])],
        'Macro': macro_cols,
        'Statistical': [f for f in feature_cols if any(x in f.lower() for x in ['skew', 'kurt', 'autocorr'])],
        'Calendar': [f for f in feature_cols if any(x in f.lower() for x in ['sin_', 'cos_'])]
    }
    
    # Remove duplicates and categorize remaining as 'Other'
    all_categorized = set()
    for cat_features in feature_categories.values():
        all_categorized.update(cat_features)
    feature_categories['Other'] = [f for f in feature_cols if f not in all_categorized]
    
    return feature_categories


def print_feature_summary(
    feature_cols: List[str],
    feature_categories: Dict[str, List[str]]
) -> None:
    """
    Print feature summary statistics.
    
    Parameters:
    -----------
    feature_cols : List[str]
        List of all feature column names
    feature_categories : Dict[str, List[str]]
        Dictionary mapping category names to feature lists
    """
    print("=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)
    print(f"\nTotal features: {len(feature_cols)}\n")
    
    for category, features in sorted(feature_categories.items(), key=lambda x: -len(x[1])):
        if features:
            print(f"{category:20s}: {len(features):3d} features")
    
    # Show sample features from top categories
    print("\n" + "=" * 80)
    print("SAMPLE FEATURES BY CATEGORY")
    print("=" * 80)
    
    for category, features in sorted(feature_categories.items(), key=lambda x: -len(x[1]))[:5]:
        if features:
            print(f"\n{category} (showing up to 5):")
            for feat in features[:5]:
                print(f"  - {feat}")


def verify_regime_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    Verify regime and concentration features (without plotting).
    For plotting, use viz.plot_regime_features()
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    """
    regime_features = ['bull_market', 'bear_market', 'relative_momentum', 'market_concentration', 'returns_52w']
    available_regime_features = [f for f in regime_features if f in train_df.columns]
    
    print("=" * 80)
    print("REGIME AND CONCENTRATION FEATURES VERIFICATION")
    print("=" * 80)
    
    if available_regime_features:
        print(f"\n✓ Found {len(available_regime_features)} regime/concentration features:")
        for feat in available_regime_features:
            if feat in train_df.columns:
                train_vals = train_df[feat].dropna()
                print(f"\n  {feat}:")
                print(f"    Train: mean={train_vals.mean():.4f}, std={train_vals.std():.4f}, "
                      f"min={train_vals.min():.4f}, max={train_vals.max():.4f}")
                if feat in test_df.columns:
                    test_vals = test_df[feat].dropna()
                    print(f"    Test:  mean={test_vals.mean():.4f}, std={test_vals.std():.4f}, "
                          f"min={test_vals.min():.4f}, max={test_vals.max():.4f}")
        
        print("\n" + "=" * 80)
        print("✓ Regime and concentration features successfully added to pipeline")
        print("=" * 80)
    else:
        print("\n⚠ WARNING: Regime features not found. They should be automatically")
        print("  included when using complete_data_pipeline() with macro data.")


# Plotting functions moved to viz.py
# Import from viz module: from viz import plot_feature_distributions


# Feature importance analysis moved to viz.py
# Import from viz module: from viz import analyze_feature_importance


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: StandardScaler,
    metadata: Dict,
    processed_data_dir: Path,
    results_dir: Path
) -> None:
    """
    Save processed datasets, scaler, and metadata.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    scaler : StandardScaler
        Fitted scaler
    metadata : Dict
        Metadata dictionary
    processed_data_dir : Path
        Directory to save processed data
    results_dir : Path
        Directory where visualizations were saved
    """
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save datasets
    train_df.to_parquet(processed_data_dir / 'train.parquet', index=False)
    test_df.to_parquet(processed_data_dir / 'test.parquet', index=False)
    
    # Save scaler
    with open(processed_data_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    with open(processed_data_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\n✓ Saved to: {processed_data_dir}")
    print(f"  - train.parquet ({len(train_df)} samples)")
    print(f"  - test.parquet ({len(test_df)} samples)")
    print(f"  - scaler.pkl")
    print(f"  - metadata.json ({len(metadata['feature_cols'])} features)")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nYou can now use these files for RL training:")
    print(f"  1. Load data: pd.read_parquet('{processed_data_dir}/train.parquet')")
    print(f"  2. Load scaler: pickle.load(open('{processed_data_dir}/scaler.pkl', 'rb'))")
    print(f"  3. Load metadata: json.load(open('{processed_data_dir}/metadata.json'))")
    print(f"\nVisualization saved to: {results_dir}")
    print("  - price_data_overview.png")
    print("  - feature_distributions.png")
    print("  - feature_importance.png")
    print("  - regime_features.png")


# Plotting functions moved to viz.py
# Import from viz module: from viz import plot_validation_test_gap

