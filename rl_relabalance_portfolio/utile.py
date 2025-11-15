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
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    """Fetch macroeconomic indicators from FRED."""

    from fredapi import Fred
    if api_key is None:
        api_key = _load_api_key_from_file('fred')

    
    if indicators is None:
        indicators = {
            'treasury_10y': 'DGS10',
            'fed_funds_rate': 'FEDFUNDS',
            'vix': 'VIXCLS',
            'unemployment_rate': 'UNRATE',
            'gdp_growth': 'A191RL1Q225SBEA',
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
    # Don't resample - keep original dates and let merge_macro_data handle alignment
    # This allows proper date matching with price data dates
    
    return macro_data


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
            rsi = RSIIndicator(close=close_series, window=window)
            result.loc[mask, f'rsi_{window}d'] = rsi.rsi()
        
        # ROC at multiple periods
        for period in [1, 4, 13, 26]:
            roc = ROCIndicator(close=close_series, window=period)
            result.loc[mask, f'roc_{period}w'] = roc.roc()
        
        # MACD components
        macd = MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)
        result.loc[mask, 'macd'] = macd.macd()
        result.loc[mask, 'macd_signal'] = macd.macd_signal()
        result.loc[mask, 'macd_histogram'] = macd.macd() - macd.macd_signal()
        
        # Stochastic Oscillator at multiple periods
        for window in [7, 14, 21]:
            stoch = StochasticOscillator(high=high_series, low=low_series, close=close_series, window=window)
            result.loc[mask, f'stochastic_{window}w'] = stoch.stoch()
        
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
        
        atr = AverageTrueRange(high=high_series, low=low_series, close=close_series, window=14)
        result.loc[mask, 'atr_14d'] = atr.average_true_range() / close_series
        
        # ATR at multiple periods
        for period in [7, 14, 26]:
            atr_multi = AverageTrueRange(high=high_series, low=low_series, close=close_series, window=period)
            result.loc[mask, f'atr_{period}d'] = atr_multi.average_true_range() / close_series
        
        bb = BollingerBands(close=close_series, window=20, window_dev=2)
        bb_high = bb.bollinger_hband()
        bb_low = bb.bollinger_lband()
        result.loc[mask, 'bb_width'] = (bb_high - bb_low) / close_series
        result.loc[mask, 'bb_position'] = (close_series - bb_low) / (bb_high - bb_low + 1e-10)
        
        # Bollinger Bands at multiple periods
        for period in [13, 20, 26]:
            bb_multi = BollingerBands(close=close_series, window=period, window_dev=2)
            bb_high_multi = bb_multi.bollinger_hband()
            bb_low_multi = bb_multi.bollinger_lband()
            result.loc[mask, f'bb_width_{period}w'] = (bb_high_multi - bb_low_multi) / close_series
            result.loc[mask, f'bb_position_{period}w'] = (close_series - bb_low_multi) / (bb_high_multi - bb_low_multi + 1e-10)
        
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


def engineer_all_features(
    df: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    macro_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Calculate all feature categories."""
    result = df.copy()
    
    if macro_data is not None and not macro_data.empty:
        result = merge_macro_data(result, macro_data)
    
    result = calculate_momentum_features(result)
    result = calculate_volatility_features(result)
    result = calculate_volume_features(result)
    result = calculate_risk_adjusted_features(result, benchmark_returns)
    result = calculate_drawdown_features(result)
    result = calculate_statistical_features(result)
    result = calculate_calendar_features(result)
    
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
    """Remove highly correlated features."""
    result = df.copy()
    available_cols = [col for col in feature_cols if col in result.columns]
    
    if len(available_cols) < 2:
        return result, []
    
    corr_matrix = result[available_cols].corr().abs()
    
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_remove = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    result = result.drop(columns=to_remove)
    
    return result, to_remove

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
    """Complete preprocessing pipeline: outlier removal, correlation filtering, split, normalization."""
    if len(df) == 0:
        raise ValueError("Input dataframe is empty.")
    
    result = df.copy()
    
    # Fill remaining NaN values in feature columns with 0 before preprocessing
    available_feature_cols = [col for col in feature_cols if col in result.columns]
    if len(available_feature_cols) > 0:
        result[available_feature_cols] = result[available_feature_cols].fillna(0)
    
    # Remove constant features (zero variance) before other preprocessing
    result, constant_features_removed = remove_constant_features(result, feature_cols)
    
    result = remove_outliers(result, feature_cols, clip_percentiles)
    
    result, removed_features = remove_highly_correlated_features(
        result, feature_cols, corr_threshold
    )

    
    splits = split_data(result, train_ratio, date_col)
    
    if len(splits['train']) == 0:
        raise ValueError(
            f"Training set is empty after preprocessing. "
            f"Input had {len(df)} rows. Check data filtering steps."
        )
    
    if len(splits['test']) == 0:
        raise ValueError(
            f"Test set is empty after preprocessing. "
            f"Input had {len(df)} rows. Check data filtering steps."
        )
    
    non_feature_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    updated_feature_cols = [
        col for col in splits['train'].columns 
        if col not in non_feature_cols
    ]
    
    if len(updated_feature_cols) == 0:
        raise ValueError(
            f"No feature columns remaining after preprocessing. "
            f"Original features: {len(feature_cols)}, "
            f"Removed by correlation: {len(removed_features)}. "
            f"Available columns: {list(splits['train'].columns)}"
        )
    
    train_norm, test_norm, scaler = normalize_features(
        splits['train'], splits['test'], updated_feature_cols
    )
    
    return {
        'train': train_norm,
        'test': test_norm,
        'scaler': scaler,
        'removed_features': removed_features + constant_features_removed,
    }

