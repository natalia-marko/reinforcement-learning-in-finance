"""
HARLF baseline notebook (Python script)
======================================

This script provides a skeleton for a Hierarchical Reinforcement Learning
pipeline tailored to portfolio management.  It follows the plan set out by the
user, covering data collection, feature engineering, sentiment integration
and basic validation.  The code is intended for use in a Jupyter
notebook environment and can be adapted as needed.

Notes on usage:

* Internet access is required for `yfinance` to download historical price
  data.  If running in an offline environment, comment out the call to
  `download_price_data()` and supply your own price DataFrame instead.
* Sentiment scraping functions (`fetch_yahoo_sentiment` and
  `fetch_reddit_sentiment`) are stubs.  They illustrate the required
  interfaces but do not perform network operations.  Replace their
  implementations with working versions when network access is available.
* The `portfolio_holdings.csv` file should contain at least a `Ticker`
  column listing the tickers in your portfolio.  You can modify the code
  to accommodate additional columns (e.g., weight, name).

"""

import datetime as dt
import itertools
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None  # yfinance will be unavailable in offline environments


def load_portfolio(filename: str) -> List[str]:
    """Load tickers from a CSV file.

    The CSV is expected to have a column named 'Ticker'.  Additional
    columns (e.g., weights, descriptions) are ignored.

    Args:
        filename: Path to the portfolio CSV file.

    Returns:
        A list of ticker symbols.
    """
    df = pd.read_csv(filename)
    if 'Ticker' not in df.columns:
        raise ValueError("Portfolio file must contain a 'Ticker' column")
    tickers = df['Ticker'].dropna().astype(str).str.upper().unique().tolist()
    return tickers


def download_price_data(tickers, TRAIN_START, TEST_END):

    # Download data with group_by ticker to get all price types
    data = yf.download(tickers, start=TRAIN_START, end=TEST_END, 
                       group_by='ticker', auto_adjust=False)
    
    # Try to use Adj Close first, fallback to Close if not available
    price_data = pd.DataFrame()
    volume_data = pd.DataFrame()
    
    for ticker in tickers:
        try:
            # Try Adj Close first
            if 'Adj Close' in data[ticker].columns:
                price_data[ticker] = data[ticker]['Adj Close']
                print(f"Using Adj Close for {ticker}")
            else:
                # Fallback to Close
                price_data[ticker] = data[ticker]['Close']
                print(f"Using Close for {ticker} (Adj Close not available)")
            
            # Extract volume data
            if 'Volume' in data[ticker].columns:
                volume_data[ticker] = data[ticker]['Volume']
                print(f"Volume data available for {ticker}")
            else:
                print(f"Warning: No volume data for {ticker}")
                volume_data[ticker] = 0
                
        except KeyError:
            # Handle case where ticker data is not available
            print(f"Warning: No data available for {ticker}")
            continue
    
    # Handle missing data
    price_data = price_data.ffill().bfill().round(2)
    volume_data = volume_data.ffill().bfill()
    
    return price_data.round(2), volume_data


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from price data.

    Args:
        prices: DataFrame of prices indexed by date.

    Returns:
        DataFrame of log returns with the same columns and an index
        shifted by one day relative to prices (first row will be NaN and
        dropped).
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def max_drawdown(series: pd.Series) -> float:
    """Compute the maximum drawdown of a cumulative return series.

    Args:
        series: Series of cumulative returns (e.g., cumulative product of
            (1 + returns)).  Should be indexed by date and increasing
            sequentially.

    Returns:
        The maximum drawdown (as a negative number).
    """
    cumulative_max = series.cummax()
    drawdowns = series / cumulative_max - 1.0
    return drawdowns.min()


def calmar_ratio(return_series: pd.Series, frequency: float = 12.0) -> float:
    """Compute the Calmar ratio for a return series.

    Calmar ratio is defined as annualized return divided by the absolute
    value of maximum drawdown.  Annualization assumes monthly data by
    default (frequency=12).  For daily data, set frequency=252.

    Args:
        return_series: Series of periodic returns (e.g., monthly returns).
        frequency: Number of periods per year.

    Returns:
        Calmar ratio.
    """
    compounded = (1 + return_series).prod() ** (frequency / len(return_series)) - 1
    dd = max_drawdown((1 + return_series).cumprod())
    if dd == 0:
        return np.inf
    return compounded / abs(dd)


def compute_volume_indicators(volume: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based indicators for each asset.
    
    Args:
        volume: DataFrame of volume data indexed by date.
        prices: DataFrame of price data indexed by date.
        
    Returns:
        DataFrame of volume indicators including:
        - Volume trend (monthly change)
        - Volume volatility
        - Price-volume correlation
        - On-balance volume (OBV)
        - Volume-weighted average price (VWAP)
        - Volume momentum
    """
    # Clean input data to handle NaN values gracefully
    volume_clean = volume.ffill().bfill().fillna(0)
    prices_clean = prices.ffill().bfill().fillna(0.01)
    
    # Check for ticker existence dates to avoid computing indicators for non-existent periods
    ticker_existence = {}
    for col in prices_clean.columns:
        if col in volume_clean.columns:
            # Find the first non-zero price and volume for this ticker
            ticker_prices = prices_clean[col].dropna()
            ticker_volume = volume_clean[col].dropna()
            
            if len(ticker_prices) > 0 and len(ticker_volume) > 0:
                # Find the first date where both price and volume are meaningful
                price_start = ticker_prices[ticker_prices > 0].index[0] if (ticker_prices > 0).any() else ticker_prices.index[0]
                volume_start = ticker_volume[ticker_volume > 0].index[0] if (ticker_volume > 0).any() else ticker_volume.index[0]
                ticker_existence[col] = max(price_start, volume_start)
            else:
                ticker_existence[col] = prices_clean.index[0]
        else:
            ticker_existence[col] = prices_clean.index[0]
    
    # Monthly volume aggregation
    monthly_volume = volume_clean.resample('ME').sum()
    monthly_prices = prices_clean.resample('ME').last()
    
    # Volume trend (monthly change) - respect ticker existence dates
    volume_trend = monthly_volume.pct_change().fillna(0)
    
    # Set indicators to 0 for periods before tickers existed
    for col in volume_clean.columns:
        if col in ticker_existence:
            existence_date = ticker_existence[col]
            # Convert to month-end for comparison
            existence_month = pd.Timestamp(existence_date).to_period('M').to_timestamp(how='end')
            volume_trend.loc[volume_trend.index < existence_month, col] = 0
    
    # Volume volatility (monthly)
    volume_vol = volume_clean.resample('ME').std().fillna(0)
    
    # Set indicators to 0 for periods before tickers existed
    for col in volume_clean.columns:
        if col in ticker_existence:
            existence_date = ticker_existence[col]
            existence_month = pd.Timestamp(existence_date).to_period('M').to_timestamp(how='end')
            volume_vol.loc[volume_vol.index < existence_month, col] = 0
    
    # Price-volume correlation (monthly) - handle NaN values gracefully
    pv_corr = {}
    for col in volume_clean.columns:
        if col in prices_clean.columns:
            # Compute correlation between price changes and volume changes
            price_changes = prices_clean[col].pct_change().dropna()
            volume_changes = volume_clean[col].pct_change().dropna()
            
            # Align the series and drop any remaining NaN values
            aligned_idx = price_changes.index.intersection(volume_changes.index)
            if len(aligned_idx) > 1:
                # Drop NaN values before correlation to avoid numpy warnings
                price_clean = price_changes.loc[aligned_idx].dropna()
                volume_clean_col = volume_changes.loc[aligned_idx].dropna()
                
                # Further align after dropping NaN
                final_idx = price_clean.index.intersection(volume_clean_col.index)
                if len(final_idx) > 1:
                    # Use numpy correlation to avoid pandas warnings
                    price_array = price_clean.loc[final_idx].values
                    volume_array = volume_clean_col.loc[final_idx].values
                    
                    # Use numpy corrcoef for more robust correlation calculation
                    if len(price_array) > 1:
                        # Remove any remaining NaN values
                        valid_mask = ~(np.isnan(price_array) | np.isnan(volume_array))
                        if valid_mask.sum() > 1:
                            price_valid = price_array[valid_mask]
                            volume_valid = volume_array[valid_mask]
                            
                            # Check if we have enough valid data
                            if len(price_valid) > 1 and len(volume_valid) > 1:
                                try:
                                    # Filter out any remaining invalid values
                                    price_final = price_valid[np.isfinite(price_valid)]
                                    volume_final = volume_valid[np.isfinite(volume_valid)]
                                    
                                    if len(price_final) > 1 and len(volume_final) > 1:
                                        # Use numpy corrcoef for more robust calculation
                                        corr_matrix = np.corrcoef(price_final, volume_final)
                                        if corr_matrix.shape == (2, 2) and np.isfinite(corr_matrix[0, 1]):
                                            pv_corr[col] = corr_matrix[0, 1]
                                        else:
                                            pv_corr[col] = 0
                                    else:
                                        pv_corr[col] = 0
                                except (ValueError, RuntimeWarning):
                                    pv_corr[col] = 0
                            else:
                                pv_corr[col] = 0
                        else:
                            pv_corr[col] = 0
                    else:
                        pv_corr[col] = 0
                else:
                    pv_corr[col] = 0
            else:
                pv_corr[col] = 0
        else:
            pv_corr[col] = 0
    
    # Create correlation DataFrame with proper index alignment
    pv_corr_expanded = pd.DataFrame(index=monthly_volume.index)
    for col in volume_clean.columns:
        pv_corr_expanded[f'pv_corr_{col}'] = pv_corr.get(col, 0)
    
    # Set correlations to 0 for periods before tickers existed
    for col in volume_clean.columns:
        if col in ticker_existence:
            existence_date = ticker_existence[col]
            existence_month = pd.Timestamp(existence_date).to_period('M').to_timestamp(how='end')
            pv_corr_expanded.loc[pv_corr_expanded.index < existence_month, f'pv_corr_{col}'] = 0
    
    # On-Balance Volume (OBV) - cumulative volume based on price direction
    obv = {}
    for col in volume_clean.columns:
        if col in prices_clean.columns:
            obv_series = pd.Series(index=volume_clean.index, dtype=float)
            obv_series.iloc[0] = volume_clean[col].iloc[0]
            
            for i in range(1, len(volume_clean)):
                if prices_clean[col].iloc[i] > prices_clean[col].iloc[i-1]:
                    obv_series.iloc[i] = obv_series.iloc[i-1] + volume_clean[col].iloc[i]
                elif prices_clean[col].iloc[i] < prices_clean[col].iloc[i-1]:
                    obv_series.iloc[i] = obv_series.iloc[i-1] - volume_clean[col].iloc[i]
                else:
                    obv_series.iloc[i] = obv_series.iloc[i-1]
            
            # Monthly OBV change
            monthly_obv = obv_series.resample('ME').last()
            obv_change = monthly_obv.pct_change().fillna(0)
            obv[col] = obv_change
        else:
            obv[col] = pd.Series(0, index=monthly_volume.index)
    
    obv_df = pd.DataFrame(obv)
    
    # Set OBV indicators to 0 for periods before tickers existed
    for col in volume_clean.columns:
        if col in ticker_existence:
            existence_date = ticker_existence[col]
            existence_month = pd.Timestamp(existence_date).to_period('M').to_timestamp(how='end')
            obv_df.loc[obv_df.index < existence_month, col] = 0
    
    # Volume-weighted average price (VWAP) - monthly
    vwap = {}
    for col in volume_clean.columns:
        if col in prices_clean.columns:
            # Daily VWAP - handle division by zero
            daily_vwap = pd.Series(index=volume_clean.index, dtype=float)
            for i in range(len(volume_clean)):
                if volume_clean[col].iloc[i] > 0:
                    daily_vwap.iloc[i] = (prices_clean[col].iloc[i] * volume_clean[col].iloc[i]) / volume_clean[col].iloc[i]
                else:
                    daily_vwap.iloc[i] = prices_clean[col].iloc[i]  # Use price if no volume
            
            # Monthly VWAP
            monthly_vwap = daily_vwap.resample('ME').mean()
            vwap[col] = monthly_vwap
        else:
            vwap[col] = pd.Series(0, index=monthly_volume.index)
    
    vwap_df = pd.DataFrame(vwap)
    
    # Set VWAP indicators to 0 for periods before tickers existed
    for col in volume_clean.columns:
        if col in ticker_existence:
            existence_date = ticker_existence[col]
            existence_month = pd.Timestamp(existence_date).to_period('M').to_timestamp(how='end')
            vwap_df.loc[vwap_df.index < existence_month, col] = 0
    
    # Volume momentum (relative to moving average) - handle division by zero
    volume_ma = volume_clean.rolling(window=20, min_periods=1).mean()  # 20-day moving average
    volume_momentum = pd.DataFrame(index=volume_clean.index, columns=volume_clean.columns)
    
    for col in volume_clean.columns:
        for i in range(len(volume_clean)):
            if volume_ma[col].iloc[i] > 0:
                volume_momentum.loc[volume_clean.index[i], col] = (volume_clean[col].iloc[i] / volume_ma[col].iloc[i]) - 1
            else:
                volume_momentum.loc[volume_clean.index[i], col] = 0
    
    volume_momentum = volume_momentum.fillna(0)
    monthly_momentum = volume_momentum.resample('ME').mean()
    
    # Set volume momentum indicators to 0 for periods before tickers existed
    for col in volume_clean.columns:
        if col in ticker_existence:
            existence_date = ticker_existence[col]
            existence_month = pd.Timestamp(existence_date).to_period('M').to_timestamp(how='end')
            monthly_momentum.loc[monthly_momentum.index < existence_month, col] = 0
    
    # Combine all volume indicators
    volume_indicators = pd.concat([
        volume_trend.add_prefix('vol_trend_'),
        volume_vol.add_prefix('vol_vol_'),
        pv_corr_expanded,
        obv_df.add_prefix('obv_'),
        vwap_df.add_prefix('vwap_'),
        monthly_momentum.add_prefix('vol_momentum_')
    ], axis=1)
    
    # Final cleanup - replace any remaining NaN or infinite values
    volume_indicators = volume_indicators.fillna(0).round(4)
    volume_indicators = volume_indicators.replace([np.inf, -np.inf], 0)
    
    return volume_indicators

def compute_monthly_indicators(
    prices: pd.DataFrame, returns: pd.DataFrame, volume: pd.DataFrame = None
) -> pd.DataFrame:
    """Compute monthly indicators for each asset.

    This function derives volatility, Sharpe ratio, Sortino ratio, max
    drawdown and Calmar ratio at a monthly frequency for each asset.  The
    correlation matrix is computed using daily returns within each
    month and flattened to a vector.

    Args:
        prices: DataFrame of prices indexed by date.
        returns: DataFrame of daily log returns (same index/columns as
            `prices`).

    Returns:
        A DataFrame indexed by year-month (PeriodIndex) with columns
        indicating metrics for each asset and correlation coefficients.
    """
    # Clean input data to handle NaN values gracefully
    prices_clean = prices.ffill().bfill().fillna(0.01)  # Small value for prices
    returns_clean = returns.ffill().bfill().fillna(0)   # Zero for returns
    
    # Monthly boundaries
    monthly_last = prices_clean.resample('ME').last()
    monthly_returns = monthly_last.pct_change().dropna()
    # Monthly volatility (annualized)
    monthly_vol = returns_clean.resample('ME').std() * np.sqrt(252)
    # Sharpe ratio (annualized): mean monthly return divided by std of daily returns * sqrt(252)
    returns_std = returns_clean.resample('ME').std()
    sharpe = monthly_returns.div(returns_std.replace(0, np.nan)) * np.sqrt(252)
    sharpe = sharpe.fillna(0)  # Replace NaN with 0 for zero volatility periods
    
    # Sortino ratio: use only negative daily returns for downside risk
    downside = returns_clean.where(returns_clean < 0, 0)
    downside_vol = downside.resample('ME').std() * np.sqrt(252)
    sortino = monthly_returns.div(downside_vol.replace(0, np.nan))
    sortino = sortino.fillna(0)  # Replace NaN with 0 for zero downside volatility periods
    # Max drawdown and Calmar ratio
    mdd = {}
    calmar = {}
    for col in monthly_returns.columns:
        r = monthly_returns[col].dropna()
        if len(r) == 0:
            mdd[col] = np.nan
            calmar[col] = np.nan
            continue
        # Construct cumulative return series for each month as a separate series
        cumulative = (1 + r).cumprod()
        mdd[col] = max_drawdown(cumulative)
        calmar[col] = calmar_ratio(r)
    mdd = pd.DataFrame([mdd])
    calmar = pd.DataFrame([calmar])
    mdd.index = [monthly_returns.index[-1]] if not monthly_returns.empty else []
    calmar.index = mdd.index

    # Correlation matrix for daily returns per month
    corr_list = []
    idx = []
    for period, group in returns_clean.groupby(returns_clean.index.to_period('M')):
        # Drop any rows with NaN values before computing correlation
        group_clean = group.dropna()
        if len(group_clean) > 1:  # Need at least 2 observations for correlation
            corr = group_clean.corr().fillna(0)
        else:
            # Create zero correlation matrix if insufficient data
            corr = pd.DataFrame(0, index=group.columns, columns=group.columns)
        
        # Flatten upper triangle (including diagonal) into a vector for reproducibility
        flat = []
        for i, asset_i in enumerate(corr.columns):
            for j, asset_j in enumerate(corr.columns):
                if j >= i:
                    flat.append(corr.iloc[i, j])
        corr_list.append(flat)
        idx.append(period)
    corr_df = pd.DataFrame(corr_list, index=idx)
    # Build a MultiIndex for metrics
    metrics = {}
    for metric_name, df in [('vol', monthly_vol), ('sharpe', sharpe), ('sortino', sortino)]:
        # Align index to monthly_returns index
        df = df.reindex(monthly_returns.index, fill_value=0)
        metrics[metric_name] = df
    # Combine metrics into a single DataFrame
    combined = pd.concat(metrics, axis=1)
    
    # Add correlation vectors with generic column names
    corr_columns = [f'corr_{i}' for i in range(corr_df.shape[1])]
    corr_df.columns = corr_columns
    
    # Align correlation data with monthly returns index
    corr_aligned = corr_df.reindex(monthly_returns.index, fill_value=0)
    
    # Create DataFrames for MDD and Calmar with proper index alignment
    mdd_expanded = pd.DataFrame(index=monthly_returns.index)
    calmar_expanded = pd.DataFrame(index=monthly_returns.index)
    
    for col in monthly_returns.columns:
        mdd_expanded[f'mdd_{col}'] = mdd[col]
        calmar_expanded[f'calmar_{col}'] = calmar[col]
    
    # Combine all components
    final = pd.concat([combined, mdd_expanded, calmar_expanded, corr_aligned], axis=1)
    
    # Add volume indicators if available
    if volume is not None:
        try:
            # Check for ticker existence dates to avoid computing indicators for non-existent periods
            volume_indicators = compute_volume_indicators(volume, prices)
            
            # Align volume indicators with the main indicators
            volume_aligned = volume_indicators.reindex(final.index, fill_value=0)
            
            final = pd.concat([final, volume_aligned], axis=1)
        except Exception as e:
            print(f"Warning: Could not compute volume indicators: {e}")
    
    # Remove any completely empty rows but be more lenient with NaN values
    final = final.dropna(how='all')
    
    # Fill remaining NaN values with 0 for numerical stability
    final = final.fillna(0)
    
    # Replace infinity values with 0 for numerical stability
    final = final.replace([np.inf, -np.inf], 0)
    
    return final.round(4)


def normalize_features(
    df: pd.DataFrame, training_period: Tuple[str, str]
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Normalize all numeric columns of df using MinMax scaling.

    The scaler is fitted on data within `training_period` (inclusive),
    defined by year-month strings (e.g., ('2003-01', '2017-12')).  The
    entire DataFrame is then transformed with the same scaler.

    Args:
        df: DataFrame with a DatetimeIndex (monthly) and numeric columns.
        training_period: Tuple of start and end period (inclusive) on which
            to fit the scaler.

    Returns:
        A tuple of (normalized DataFrame, fitted MinMaxScaler).
    """
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex of monthly periods")
    
    # Convert training period strings to datetime for comparison
    train_start = pd.to_datetime(training_period[0] + '-01')
    train_end = pd.to_datetime(training_period[1] + '-01')
    
    train_mask = (df.index >= train_start) & (df.index <= train_end)
    scaler = MinMaxScaler()
    scaler.fit(df.loc[train_mask])
    normalized = pd.DataFrame(
        scaler.transform(df), index=df.index, columns=df.columns
    )
    return normalized.round(4), scaler


def shift_returns_for_validation(
    monthly_returns: pd.DataFrame, periods: int = 1
) -> pd.DataFrame:
    """Shift monthly returns by a specified number of periods.

    Used to align sentiment scores at time t with returns at time t+1.

    Args:
        monthly_returns: DataFrame of monthly percentage returns.
        periods: Number of months to shift.

    Returns:
        Shifted DataFrame aligned on the original index.
    """
    return monthly_returns.shift(-periods)


def compute_sentiment_correlation(
    sentiment: pd.DataFrame, future_returns: pd.DataFrame
) -> pd.Series:
    """Compute Pearson correlation between sentiment and future returns.

    Args:
        sentiment: DataFrame with sentiment scores by asset and month.
        future_returns: DataFrame of future monthly returns (aligned on
            index and asset columns).

    Returns:
        A Series of correlation coefficients per asset.
    """
    corrs = {}
    for col in sentiment.columns:
        # Align indexes and drop rows where either is NaN
        s = sentiment[col].dropna()
        f = future_returns[col].dropna()
        aligned = s.index.intersection(f.index)
        if len(aligned) > 1:
            corrs[col] = s.loc[aligned].corr(f.loc[aligned])
        else:
            corrs[col] = np.nan
    return pd.Series(corrs)


def fetch_yahoo_sentiment(
    tickers: Iterable[str], start: str, end: str
) -> pd.DataFrame:
    """Placeholder for Yahoo News sentiment scraping.

    This function should fetch news articles for each ticker between the
    given dates, compute FinBERT sentiment scores for each article, and
    aggregate them by month as described in the plan.  Since internet
    access may be unavailable, this implementation returns random
    sentiment scores as a demonstration.

    Args:
        tickers: Iterable of ticker symbols.
        start: Start date (inclusive) as 'YYYY-MM-DD'.
        end: End date (exclusive) as 'YYYY-MM-DD'.

    Returns:
        DataFrame indexed by period (monthly) with columns per ticker.
    """
    # Generate a DatetimeIndex from start to end with month-end frequency
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    dates = pd.date_range(start_dt, end_dt, freq='ME')
    rng = np.random.default_rng(seed=42)
    data = rng.normal(loc=0.0, scale=0.1, size=(len(dates), len(tickers)))
    sentiment_df = pd.DataFrame(data, index=dates, columns=[t.upper() for t in tickers])
    
    return sentiment_df


def fetch_reddit_sentiment(
    tickers: Iterable[str], start: str, end: str
) -> pd.DataFrame:
    """Placeholder for Reddit sentiment scraping.

    Similar to the Yahoo function, this returns random sentiment scores.  In
    practice, replace with code that queries the Reddit API or Pushshift
    for posts mentioning each ticker, runs FinBERT or another sentiment
    classifier, and aggregates scores by month.
    """
    # For demonstration, call the Yahoo sentiment stub to reuse its random data
    return fetch_yahoo_sentiment(tickers, start, end)


def aggregate_sentiment(
    yahoo_sent: pd.DataFrame, reddit_sent: pd.DataFrame, weights: Tuple[float, float] = (0.5, 0.5)
) -> pd.DataFrame:
    """Combine sentiment scores from Yahoo and Reddit using a weighted average.

    Args:
        yahoo_sent: DataFrame of Yahoo sentiment scores.
        reddit_sent: DataFrame of Reddit sentiment scores.
        weights: Tuple of weights (yahoo_weight, reddit_weight).

    Returns:
        DataFrame of aggregated sentiment scores per asset and month.
    """
    if yahoo_sent.shape != reddit_sent.shape:
        raise ValueError("Sentiment DataFrames must have the same shape and index")
    w_y, w_r = weights
    combined = w_y * yahoo_sent + w_r * reddit_sent
    return combined


def main():
    # Configuration
    portfolio_path = 'portfolio_holdings.csv'
    benchmark = 'QQQ'
    train_start = '2003-01-01'
    train_end = '2017-12-31'
    test_start = '2018-01-01'
    test_end = '2024-12-31'
    full_start = train_start
    full_end = test_end

    # Load portfolio tickers
    try:
        tickers = load_portfolio(portfolio_path)
    except Exception as e:
        logging.warning(
            f"Could not load portfolio file '{portfolio_path}'. Falling back to demo tickers: {e}"
        )
        tickers = ['AAPL', 'MSFT', 'GOOG']
    # Include benchmark for comparison
    if benchmark.upper() not in tickers:
        tickers.append(benchmark.upper())

    # Download price and volume data
    try:
        prices, volume = download_price_data(tickers, full_start, full_end)
    except Exception as e:
        logging.error(
            f"Failed to download price data: {e}. Provide your own price DataFrame."
        )
        return

    # Compute daily log returns
    log_rets = compute_log_returns(prices)
    # Compute monthly price returns for future validation
    monthly_pct_returns = prices.resample('ME').last().pct_change().dropna()

    # Compute monthly indicators (including volume indicators)
    indicators = compute_monthly_indicators(prices, log_rets, volume)

    # Fetch sentiment data
    yahoo_sent = fetch_yahoo_sentiment(tickers, start=full_start, end=full_end)
    reddit_sent = fetch_reddit_sentiment(tickers, start=full_start, end=full_end)
    sentiment = aggregate_sentiment(yahoo_sent, reddit_sent)

    # Align sentiment with indicators
    sentiment = sentiment.loc[indicators.index]

    # Compute correlation between sentiment and future returns (t+1)
    future_returns = shift_returns_for_validation(monthly_pct_returns, periods=1)
    # Align index
    future_returns = future_returns.loc[indicators.index]
    sentiment_corr = compute_sentiment_correlation(sentiment, future_returns)
    print("Sentiment vs. next-month return correlation:")
    print(sentiment_corr)

    # Flatten column names for sklearn compatibility
    indicators_flat = indicators.copy()
    indicators_flat.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else str(col) 
                              for col in indicators_flat.columns]
    
    # Normalize features using training period
    normalized_indicators, scaler = normalize_features(
        indicators_flat, training_period=(train_start[:7], train_end[:7])
    )
    # Combine normalized indicators and sentiment as final state representation
    # We'll keep sentiment unnormalized to preserve interpretability
    final_features = pd.concat([normalized_indicators, sentiment.add_prefix('sent_')], axis=1)
    # Persist final features to disk for later RL training
    final_features.to_csv('monthly_features_with_sentiment.csv')
    print("Generated monthly features with sentiment saved to 'monthly_features_with_sentiment.csv'.")


if __name__ == '__main__':
    main()