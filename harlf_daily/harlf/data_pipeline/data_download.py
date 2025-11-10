"""
Data download module for stock, benchmark, and macro data
Phase 2: Data Collection (Daily Frequency)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr

from harlf.config import defaults as config
from harlf.utils import io as utils


def download_stock_data():
    """
    Download OHLCV data for all tickers

    Returns:
    --------
    pd.DataFrame
        Multi-level DataFrame with OHLCV data for all tickers
    """
    config.log(f"Downloading stock data for {len(config.TICKERS)} tickers...")
    config.log(f"Tickers: {config.TICKERS}")
    config.log(f"Date range: {config.START_DATE} to {config.END_DATE}")

    # Download data using batch download
    df = yf.download(
        tickers=config.TICKERS,
        start=config.START_DATE,
        end=config.END_DATE,
        auto_adjust=config.YFINANCE_AUTO_ADJUST,
        actions=config.YFINANCE_ACTIONS,
        progress=config.YFINANCE_PROGRESS,
    )

    # Handle multi-level column index
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns MultiIndex columns: (Price, Ticker)
        # We need to swap levels to get (Ticker, Price)
        df.columns = df.columns.swaplevel()
        config.log("Handled multi-level column index")

    config.log(f"Downloaded {len(df)} trading days")

    return df


def process_stock_data(df):
    """
    Process stock data: handle missing values, calculate returns

    Parameters:
    -----------
    df : pd.DataFrame
        Raw stock data from yfinance

    Returns:
    --------
    dict
        Dictionary with processed DataFrames for each OHLCV field
    pd.DataFrame
        Asset first valid dates
    """
    config.log("Processing stock data...")

    # Separate into individual DataFrames for each price type
    data = {}

    if isinstance(df.columns, pd.MultiIndex):
        # Multi-ticker case
        for price_type in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if price_type in df.columns.get_level_values(1):
                data[price_type.lower()] = df.xs(price_type, level=1, axis=1)
    else:
        # Single ticker case (shouldn't happen with batch download)
        data = {
            'open': df['Open'],
            'high': df['High'],
            'low': df['Low'],
            'close': df['Close'],
            'volume': df['Volume'],
        }

    # Track first valid date for each asset (IPO handling)
    asset_first_valid = {}
    for ticker in config.TICKERS:
        if ticker in data['close'].columns:
            first_valid = data['close'][ticker].first_valid_index()
            asset_first_valid[ticker] = first_valid
            config.log(f"  {ticker}: First valid date = {first_valid}")

    asset_first_valid_df = pd.DataFrame.from_dict(
        asset_first_valid,
        orient='index',
        columns=['first_valid_date']
    )
    asset_first_valid_df.index.name = 'ticker'

    # Handle missing data: forward fill with limit
    for key in data:
        # Forward fill within limit
        data[key] = data[key].ffill(limit=config.MAX_FORWARD_FILL_DAYS)

        # Count remaining NaNs
        nan_count = data[key].isna().sum().sum()
        if nan_count > 0:
            config.log(f"  {key}: {nan_count} NaN values after forward fill")

    # Calculate daily returns
    data['returns'] = data['close'].pct_change()

    config.log("Stock data processing complete")

    return data, asset_first_valid_df


def download_benchmark_data():
    """
    Download QQQ benchmark data

    Returns:
    --------
    pd.DataFrame
        Benchmark OHLCV data
    """
    config.log(f"Downloading benchmark data: {config.BENCHMARK}...")

    df = yf.download(
        tickers=config.BENCHMARK,
        start=config.START_DATE,
        end=config.END_DATE,
        auto_adjust=config.YFINANCE_AUTO_ADJUST,
        progress=config.YFINANCE_PROGRESS,
    )

    # Handle multi-level column index (single ticker still returns MultiIndex sometimes)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    config.log(f"Downloaded {len(df)} trading days for {config.BENCHMARK}")

    return df


def process_benchmark_data(df):
    """
    Process benchmark data

    Parameters:
    -----------
    df : pd.DataFrame
        Raw benchmark data

    Returns:
    --------
    dict
        Dictionary with processed benchmark data
    """
    config.log("Processing benchmark data...")

    data = {
        'open': df['Open'],
        'high': df['High'],
        'low': df['Low'],
        'close': df['Close'],
        'volume': df['Volume'],
    }

    # Forward fill missing data
    for key in data:
        data[key] = data[key].ffill(limit=config.MAX_FORWARD_FILL_DAYS)

    # Calculate returns
    data['returns'] = data['close'].pct_change()

    config.log("Benchmark data processing complete")

    return data


def download_vix_data():
    """
    Download VIX data from Yahoo Finance

    Returns:
    --------
    pd.Series
        VIX closing prices
    """
    config.log("Downloading VIX data...")

    df = yf.download(
        tickers='^VIX',
        start=config.START_DATE,
        end=config.END_DATE,
        progress=config.YFINANCE_PROGRESS,
    )

    # Handle multi-level column index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    vix = df['Close']
    vix.name = 'vix'

    # Forward fill missing data
    vix = vix.ffill(limit=config.MAX_FORWARD_FILL_DAYS)

    config.log(f"Downloaded {len(vix)} days of VIX data")

    return vix


def download_fred_data():
    """
    Download macro economic data from FRED

    Returns:
    --------
    pd.DataFrame
        Macro indicators
    """
    config.log("Downloading FRED macro data...")

    # Check if FRED API key is available
    if config.FRED_API_KEY is None:
        config.log("  WARNING: No FRED API key found - creating dummy macro data")
        config.log("  To use real data, add FRED API key to config/api_keys.json")

        # Create dummy macro data with neutral values
        date_range = pd.date_range(start=config.START_DATE, end=config.END_DATE, freq='D')
        macro_data = {}
        for feature_name in config.FRED_INDICATORS.values():
            macro_data[feature_name] = pd.Series(0.0, index=date_range)

        macro_df = pd.DataFrame(macro_data)
        config.log(f"Created dummy macro data: {len(macro_df)} days, {len(macro_df.columns)} indicators")
        return macro_df

    config.log(f"FRED API Key: {config.FRED_API_KEY[:10]}...")

    macro_data = {}

    for fred_code, feature_name in config.FRED_INDICATORS.items():
        try:
            config.log(f"  Downloading {fred_code} ({feature_name})...")
            series = pdr.DataReader(
                fred_code,
                'fred',
                config.START_DATE,
                config.END_DATE,
                api_key=config.FRED_API_KEY
            )

            # Rename column
            series.columns = [feature_name]

            # Store
            macro_data[feature_name] = series[feature_name]

            config.log(f"    Downloaded {len(series)} observations")

        except Exception as e:
            config.log(f"    ERROR downloading {fred_code}: {e}")
            macro_data[feature_name] = None

    # Combine into single DataFrame
    macro_df = pd.DataFrame(macro_data)

    # Resample to daily frequency and forward fill (with limit)
    macro_df = macro_df.resample('D').ffill(limit=config.MAX_MACRO_FORWARD_FILL_DAYS)

    config.log(f"FRED data downloaded: {len(macro_df)} days, {len(macro_df.columns)} indicators")

    return macro_df


def align_data(stock_data, benchmark_data, macro_data, vix_data):
    """
    Align all data to common trading days

    Parameters:
    -----------
    stock_data : dict
        Stock OHLCV data
    benchmark_data : dict
        Benchmark data
    macro_data : pd.DataFrame
        Macro indicators
    vix_data : pd.Series
        VIX data

    Returns:
    --------
    dict
        Aligned data
    """
    config.log("Aligning data to common dates...")

    # Get common dates (intersection of all data sources)
    stock_dates = stock_data['close'].index
    bench_dates = benchmark_data['close'].index
    macro_dates = macro_data.index
    vix_dates = vix_data.index

    # Find intersection
    common_dates = stock_dates.intersection(bench_dates)
    common_dates = common_dates.intersection(macro_dates)
    common_dates = common_dates.intersection(vix_dates)

    config.log(f"  Stock dates: {len(stock_dates)}")
    config.log(f"  Benchmark dates: {len(bench_dates)}")
    config.log(f"  Macro dates: {len(macro_dates)}")
    config.log(f"  VIX dates: {len(vix_dates)}")
    config.log(f"  Common dates: {len(common_dates)}")

    # Align all data
    aligned = {
        'stock': {},
        'benchmark': {},
        'macro': macro_data.loc[common_dates],
        'vix': vix_data.loc[common_dates],
    }

    for key in stock_data:
        aligned['stock'][key] = stock_data[key].loc[common_dates]

    for key in benchmark_data:
        aligned['benchmark'][key] = benchmark_data[key].loc[common_dates]

    config.log("Data alignment complete")

    return aligned


def save_raw_data(aligned_data, asset_first_valid):
    """
    Save raw data to files

    Parameters:
    -----------
    aligned_data : dict
        Aligned data
    asset_first_valid : pd.DataFrame
        Asset first valid dates
    """
    config.log("Saving raw data...")

    # Stock data
    stock_df = pd.DataFrame()
    for ticker in config.TICKERS:
        for price_type in ['open', 'high', 'low', 'close', 'volume', 'returns']:
            if ticker in aligned_data['stock'][price_type].columns:
                col_name = f"{ticker}_{price_type}"
                stock_df[col_name] = aligned_data['stock'][price_type][ticker]

    utils.save_dataframe(stock_df, config.RAW_STOCK_FILE)

    # Benchmark data
    bench_df = pd.DataFrame()
    for price_type in ['open', 'high', 'low', 'close', 'volume', 'returns']:
        bench_df[price_type] = aligned_data['benchmark'][price_type]

    utils.save_dataframe(bench_df, config.RAW_BENCHMARK_FILE)

    # Macro data
    utils.save_dataframe(aligned_data['macro'], config.RAW_MACRO_FILE)

    # VIX data
    vix_df = pd.DataFrame({'vix': aligned_data['vix']})
    utils.save_dataframe(vix_df, config.RAW_VIX_FILE)

    # Asset first valid dates
    utils.save_dataframe(asset_first_valid, config.METADATA_FILES['asset_first_valid'])

    config.log("Raw data saved successfully")


def load_raw_data():
    """
    Load raw data from files

    Returns:
    --------
    dict
        Dictionary with all raw data
    """
    config.log("Loading raw data from files...")

    data = {
        'stock': utils.load_dataframe(config.RAW_STOCK_FILE),
        'benchmark': utils.load_dataframe(config.RAW_BENCHMARK_FILE),
        'macro': utils.load_dataframe(config.RAW_MACRO_FILE),
        'vix': utils.load_dataframe(config.RAW_VIX_FILE),
        'asset_first_valid': utils.load_dataframe(config.METADATA_FILES['asset_first_valid']),
    }

    return data


def main():
    """
    Main function to download and process all data
    """
    config.log("="*60)
    config.log("PHASE 2: DATA COLLECTION")
    config.log("="*60)

    # Download stock data
    stock_raw = download_stock_data()
    stock_data, asset_first_valid = process_stock_data(stock_raw)

    # Download benchmark data
    benchmark_raw = download_benchmark_data()
    benchmark_data = process_benchmark_data(benchmark_raw)

    # Download VIX data
    vix_data = download_vix_data()

    # Download FRED macro data
    macro_data = download_fred_data()

    # Align all data
    aligned_data = align_data(stock_data, benchmark_data, macro_data, vix_data)

    # Save raw data
    save_raw_data(aligned_data, asset_first_valid)

    config.log("="*60)
    config.log("DATA COLLECTION COMPLETE")
    config.log("="*60)

    return aligned_data


if __name__ == "__main__":
    main()
