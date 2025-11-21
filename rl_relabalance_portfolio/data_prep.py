"""
Data Preparation for RL Portfolio Training

Streamlined pipeline following feature_fixes_weekly.py pattern:
- Clear, focused functions
- Minimal feature set (evidence-based)
- Per-ticker normalization
- Weekly frequency

Usage:
    from data_prep import prepare_weekly_data
    train_df, test_df, metadata = prepare_weekly_data()
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict, List
import yfinance as yf
from datetime import datetime
import warnings


def fetch_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1wk'
) -> pd.DataFrame:
    """
    Fetch price data from Yahoo Finance

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval ('1wk', '1d', etc.)

    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume
    """
    print(f"Fetching {interval} data for {len(tickers)} tickers...")

    all_data = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True)

            # Handle MultiIndex columns (yfinance returns these for single tickers)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df['ticker'] = ticker

            # Convert column names to lowercase
            df.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]

            all_data.append(df)
            print(f"  ✓ {ticker}: {len(df)} periods")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    # Combine all data
    data = pd.concat(all_data, ignore_index=True)
    data = data.sort_values(['date', 'ticker']).reset_index(drop=True)

    print(f"✓ Fetched {len(data)} total rows")
    return data


def fetch_macro_features(
    start_date: str,
    end_date: str,
    api_key: str = None
) -> pd.DataFrame:
    """
    Fetch macro features: VIX, Treasury 10Y, Yield Curve, DXY, Oil WTI
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: Optional FRED API key for treasury/yield curve data
    
    Returns:
        DataFrame with columns: date, vix, treasury_10y, yield_curve_10y2y, dxy, oil_wti
    """
    print("\nFetching macro features...")
    
    macro_data = []
    
    # Fetch VIX from Yahoo Finance
    try:
        vix = yf.download('^VIX', start=start_date, end=end_date, interval='1wk', progress=False, auto_adjust=True)
        if not vix.empty:
            # Handle MultiIndex columns
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix = vix.reset_index()
            # Convert column names to lowercase FIRST (before creating date column)
            vix.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in vix.columns]
            # Now handle date column - it should be 'date' after lowercase conversion
            if 'date' in vix.columns:
                vix['date'] = pd.to_datetime(vix['date'])
            else:
                # Fallback if date column doesn't exist
                vix['date'] = pd.to_datetime(vix.index)
            vix = vix[['date', 'close']].rename(columns={'close': 'vix'})
            macro_data.append(vix)
            print("  ✓ VIX")
    except Exception as e:
        warnings.warn(f"Failed to fetch VIX: {e}", UserWarning)
    
    # Fetch DXY (Dollar Index) from Yahoo Finance
    try:
        dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date, interval='1wk', progress=False, auto_adjust=True)
        if dxy.empty:
            # Try alternative symbol
            dxy = yf.download('UUP', start=start_date, end=end_date, interval='1wk', progress=False, auto_adjust=True)
        if not dxy.empty:
            # Handle MultiIndex columns
            if isinstance(dxy.columns, pd.MultiIndex):
                dxy.columns = dxy.columns.get_level_values(0)
            dxy = dxy.reset_index()
            # Convert column names to lowercase FIRST (before creating date column)
            dxy.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in dxy.columns]
            # Now handle date column - it should be 'date' after lowercase conversion
            if 'date' in dxy.columns:
                dxy['date'] = pd.to_datetime(dxy['date'])
            else:
                # Fallback if date column doesn't exist
                dxy['date'] = pd.to_datetime(dxy.index)
            dxy = dxy[['date', 'close']].rename(columns={'close': 'dxy'})
            macro_data.append(dxy)
            print("  ✓ DXY")
    except Exception as e:
        warnings.warn(f"Failed to fetch DXY: {e}", UserWarning)
    
    # Fetch Oil WTI from Yahoo Finance
    try:
        oil = yf.download('CL=F', start=start_date, end=end_date, interval='1wk', progress=False, auto_adjust=True)
        if not oil.empty:
            # Handle MultiIndex columns
            if isinstance(oil.columns, pd.MultiIndex):
                oil.columns = oil.columns.get_level_values(0)
            oil = oil.reset_index()
            # Convert column names to lowercase FIRST (before creating date column)
            oil.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in oil.columns]
            # Now handle date column - it should be 'date' after lowercase conversion
            if 'date' in oil.columns:
                oil['date'] = pd.to_datetime(oil['date'])
            else:
                # Fallback if date column doesn't exist
                oil['date'] = pd.to_datetime(oil.index)
            oil = oil[['date', 'close']].rename(columns={'close': 'oil_wti'})
            macro_data.append(oil)
            print("  ✓ Oil WTI")
    except Exception as e:
        warnings.warn(f"Failed to fetch Oil WTI: {e}", UserWarning)
    
    # Fetch Treasury yields and yield curve from FRED (if API key available)
    try:
        from fredapi import Fred
        
        # Try to load API key from file if not provided
        if api_key is None:
            try:
                api_keys_path = Path('api_keys.json')
                if api_keys_path.exists():
                    with open(api_keys_path, 'r') as f:
                        keys = json.load(f)
                        api_key = keys.get('fred')
            except:
                pass
        
        if api_key:
            fred = Fred(api_key=api_key)
            
            # Fetch 10Y Treasury
            try:
                treasury_10y = fred.get_series('DGS10', start=start_date, end=end_date)
                if not treasury_10y.empty:
                    treasury_10y = treasury_10y.reset_index()
                    treasury_10y.columns = ['date', 'treasury_10y']
                    treasury_10y['date'] = pd.to_datetime(treasury_10y['date'])
                    # Resample to weekly (take last value of week)
                    treasury_10y = treasury_10y.set_index('date').resample('W').last().reset_index()
                    macro_data.append(treasury_10y)
                    print("  ✓ Treasury 10Y")
            except Exception as e:
                warnings.warn(f"Failed to fetch Treasury 10Y: {e}", UserWarning)
            
            # Fetch Yield Curve (10Y-2Y spread)
            try:
                yield_curve = fred.get_series('T10Y2Y', start=start_date, end=end_date)
                if not yield_curve.empty:
                    yield_curve = yield_curve.reset_index()
                    yield_curve.columns = ['date', 'yield_curve_10y2y']
                    yield_curve['date'] = pd.to_datetime(yield_curve['date'])
                    # Resample to weekly (take last value of week)
                    yield_curve = yield_curve.set_index('date').resample('W').last().reset_index()
                    macro_data.append(yield_curve)
                    print("  ✓ Yield Curve (10Y-2Y)")
            except Exception as e:
                warnings.warn(f"Failed to fetch Yield Curve: {e}", UserWarning)
        else:
            warnings.warn("FRED API key not found. Treasury and yield curve features will be missing.", UserWarning)
    except ImportError:
        warnings.warn("fredapi not installed. Install with: pip install fredapi. Treasury/yield curve features will be missing.", UserWarning)
    except Exception as e:
        warnings.warn(f"Failed to fetch FRED data: {e}", UserWarning)
    
    # Merge all macro data
    if not macro_data:
        warnings.warn("No macro data fetched. Continuing without macro features.", UserWarning)
        return pd.DataFrame()
    
    # Start with first dataframe
    result = macro_data[0].copy()
    
    # Merge remaining dataframes
    for df in macro_data[1:]:
        result = result.merge(df, on='date', how='outer')
    
    # Sort by date
    result = result.sort_values('date').reset_index(drop=True)
    
    # Forward fill missing values
    result = result.ffill().bfill()
    
    print(f"✓ Fetched {len([c for c in result.columns if c != 'date'])} macro features")
    return result


def calculate_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate calendar/seasonality features using sin/cos encoding for monthly patterns
    
    Since rebalancing happens monthly (every 4 weeks), monthly seasonality captures:
    - January effect
    - End-of-quarter effects  
    - End-of-year effects
    - Monthly earnings cycles
    
    Args:
        data: DataFrame with date column
    
    Returns:
        DataFrame with added calendar feature columns (month_sin, month_cos)
    """
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Month of year (1-12)
    df['month'] = df['date'].dt.month
    
    # Sin/cos encoding for monthly seasonality (captures cyclical patterns)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Drop intermediate column
    df = df.drop(columns=['month'])
    
    return df


def calculate_minimal_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate minimal feature set (evidence-based from literature)

    Features (19 ticker-specific + 5 macro + 2 calendar = 26 total):
    - Momentum: 1w, 4w, 13w, 52w returns
    - Volatility: 4w, 52w
    - Technical: SMA, EMA, MACD, Bollinger Bands
    - Volume: OBV, relative volume
    - Risk: Current drawdown, RSI, Stochastic
    - Macro: VIX, Treasury 10Y, Yield Curve, DXY, Oil
    - Calendar: Month sin/cos (seasonality for monthly rebalancing)

    Args:
        data: DataFrame with OHLCV data per ticker

    Returns:
        DataFrame with added feature columns
    """
    print("\nCalculating minimal features (19 ticker-specific)...")

    df = data.copy()

    # Group by ticker for feature calculation
    features = []

    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('date').reset_index(drop=True)

        # Price and returns
        ticker_df['returns'] = ticker_df['close'].pct_change()

        # Momentum features (4)
        ticker_df['momentum_1w'] = ticker_df['close'].pct_change(1)
        ticker_df['momentum_4w'] = ticker_df['close'].pct_change(4)
        ticker_df['momentum_13w'] = ticker_df['close'].pct_change(13)
        ticker_df['returns_52w'] = ticker_df['close'].pct_change(52)

        # Volatility features (2)
        ticker_df['volatility_4w'] = ticker_df['returns'].rolling(4).std()
        ticker_df['volatility_52w'] = ticker_df['returns'].rolling(52).std()

        # Technical indicators (6)
        ticker_df['price_to_sma_12w'] = ticker_df['close'] / ticker_df['close'].rolling(12).mean()
        ticker_df['price_to_ema_26w'] = ticker_df['close'] / ticker_df['close'].ewm(span=26).mean()

        # MACD
        ema_12 = ticker_df['close'].ewm(span=12).mean()
        ema_26 = ticker_df['close'].ewm(span=26).mean()
        ticker_df['macd_histogram'] = (ema_12 - ema_26) / ticker_df['close']

        # Bollinger Bands
        sma_20 = ticker_df['close'].rolling(20).mean()
        std_20 = ticker_df['close'].rolling(20).std()
        ticker_df['bb_position'] = (ticker_df['close'] - sma_20) / (2 * std_20)
        ticker_df['bb_width'] = (2 * std_20) / sma_20

        # RSI
        delta = ticker_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        ticker_df['rsi_14d'] = 100 - (100 / (1 + rs))

        # Volume features (4)
        ticker_df['obv'] = (ticker_df['volume'] * np.sign(ticker_df['returns'])).cumsum()
        ticker_df['obv_roc_4w'] = ticker_df['obv'].pct_change(4)
        ticker_df['relative_volume_4w'] = ticker_df['volume'] / ticker_df['volume'].rolling(4).mean()
        ticker_df['volume_roc_13w'] = ticker_df['volume'].pct_change(13)

        # Risk features (3)
        cummax = ticker_df['close'].expanding().max()
        ticker_df['current_drawdown'] = (ticker_df['close'] - cummax) / cummax

        # Stochastic Oscillator
        low_14 = ticker_df['low'].rolling(14).min()
        high_14 = ticker_df['high'].rolling(14).max()
        ticker_df['stochastic_14w'] = (ticker_df['close'] - low_14) / (high_14 - low_14 + 1e-10)

        # Downside volatility
        downside_returns = ticker_df['returns'].where(ticker_df['returns'] < 0, 0)
        ticker_df['downside_volatility_4w'] = downside_returns.rolling(4).std()

        features.append(ticker_df)

    result = pd.concat(features, ignore_index=True)
    result = result.sort_values(['date', 'ticker']).reset_index(drop=True)

    print(f"✓ Calculated 19 ticker-specific features")
    return result


def normalize_per_ticker(
    data: pd.DataFrame,
    feature_cols: List[str],
    window: int = 52,
    min_periods: int = 26
) -> pd.DataFrame:
    """
    Apply per-ticker rolling z-score normalization (Zhang et al. 2020)

    Args:
        data: DataFrame with features
        feature_cols: List of feature columns to normalize
        window: Rolling window size (weeks)
        min_periods: Minimum periods for calculation

    Returns:
        DataFrame with normalized features (suffix '_norm')
    """
    print(f"\nApplying per-ticker z-score normalization (window={window})...")

    df = data.copy()

    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker

        for col in feature_cols:
            if col not in df.columns:
                continue

            # Calculate rolling mean and std per ticker
            rolling_mean = df.loc[mask, col].rolling(window=window, min_periods=min_periods).mean()
            rolling_std = df.loc[mask, col].rolling(window=window, min_periods=min_periods).std()

            # Z-score normalization
            df.loc[mask, f'{col}_norm'] = (df.loc[mask, col] - rolling_mean) / (rolling_std + 1e-8)

    print(f"✓ Normalized {len(feature_cols)} features per ticker")
    return df


def normalize_macro_features(
    data: pd.DataFrame,
    macro_cols: List[str]
) -> pd.DataFrame:
    """
    Apply global z-score normalization to macro features (not per-ticker)
    
    Args:
        data: DataFrame with macro features
        macro_cols: List of macro feature columns to normalize
    
    Returns:
        DataFrame with normalized macro features (suffix '_norm')
    """
    if not macro_cols:
        return data
    
    print(f"\nNormalizing macro features (global z-score)...")
    
    df = data.copy()
    
    for col in macro_cols:
        if col not in df.columns:
            continue
        
        # Global mean and std (across all dates and tickers)
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val == 0:
            std_val = 1e-8
        
        df[f'{col}_norm'] = (df[col] - mean_val) / std_val
    
    print(f"✓ Normalized {len([c for c in macro_cols if c in df.columns])} macro features")
    return df


def split_train_test(
    data: pd.DataFrame,
    test_start_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets by date

    Args:
        data: DataFrame with date column
        test_start_date: Start date for test set (YYYY-MM-DD)

    Returns:
        train_df, test_df
    """
    print(f"\nSplitting data at {test_start_date}...")

    test_date = pd.to_datetime(test_start_date)
    train = data[data['date'] < test_date].copy()
    test = data[data['date'] >= test_date].copy()

    print(f"✓ Train: {len(train)} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"✓ Test:  {len(test)} rows ({test['date'].min().date()} to {test['date'].max().date()})")

    return train, test


def prepare_weekly_data(
    tickers: List[str] = None,
    start_date: str = '2015-01-01',
    end_date: str = '2025-11-01',
    test_start_date: str = '2023-09-01',
    output_dir: Path = None,
    fred_api_key: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Complete weekly data preparation pipeline

    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        test_start_date: Date to split train/test
        output_dir: Directory to save outputs
        fred_api_key: Optional FRED API key for treasury/yield curve data

    Returns:
        train_df, test_df, metadata
    """
    print("="*80)
    print("WEEKLY DATA PREPARATION PIPELINE")
    print("="*80)

    # Default tickers
    if tickers is None:
        tickers = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']

    # Default output dir
    if output_dir is None:
        output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch price data
    data = fetch_price_data(tickers, start_date, end_date, interval='1wk')

    # Step 2: Fetch macro features
    macro_data = fetch_macro_features(start_date, end_date, api_key=fred_api_key)
    
    # Step 3: Merge macro data with price data
    if not macro_data.empty:
        data = data.merge(macro_data, on='date', how='left')
        # Forward fill macro features within each ticker group
        macro_cols = [c for c in macro_data.columns if c != 'date']
        for col in macro_cols:
            if col in data.columns:
                data[col] = data.groupby('ticker')[col].ffill().bfill()
        print(f"✓ Merged {len(macro_cols)} macro features")
    else:
        macro_cols = []
        print("⚠ No macro features available")

    # Step 4: Calculate calendar features (month sin/cos for monthly rebalancing)
    data = calculate_calendar_features(data)
    calendar_cols = ['month_sin', 'month_cos']
    print(f"✓ Calculated {len(calendar_cols)} calendar features (monthly seasonality)")

    # Step 5: Calculate ticker-specific features
    data = calculate_minimal_features(data)

    # Step 6: Get feature columns
    base_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'returns']
    ticker_feature_cols = [c for c in data.columns if c not in base_cols and c not in macro_cols and c not in calendar_cols]
    
    # Step 7: Normalize ticker features per ticker
    data = normalize_per_ticker(data, ticker_feature_cols, window=52, min_periods=26)
    
    # Step 8: Normalize macro features globally
    if macro_cols:
        data = normalize_macro_features(data, macro_cols)
    
    # Step 9: Calendar features (month_sin, month_cos) are already in [-1, 1] range
    # No normalization needed - they encode cyclical patterns directly

    # Step 10: Split train/test
    train_df, test_df = split_train_test(data, test_start_date)

    # Step 11: Clean up - keep only normalized features
    # Note: Calendar features (month_sin, month_cos) are NOT normalized - they're already in [-1, 1]
    norm_ticker_features = [f'{c}_norm' for c in ticker_feature_cols]
    norm_macro_features = [f'{c}_norm' for c in macro_cols] if macro_cols else []
    # Calendar features don't get _norm suffix since they're not normalized
    norm_features = norm_ticker_features + norm_macro_features + calendar_cols
    
    keep_cols = base_cols + norm_features

    train_df = train_df[keep_cols].dropna()
    test_df = test_df[keep_cols].dropna()

    # Create metadata
    metadata = {
        'tickers': tickers,
        'start_date': start_date,
        'end_date': end_date,
        'test_start_date': test_start_date,
        'interval': '1wk',
        'feature_cols': norm_features,
        'n_features': len(norm_features),
        'ticker_features': norm_ticker_features,
        'macro_features': norm_macro_features,
        'calendar_features': calendar_cols,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'normalization': {
            'ticker_features': 'per_ticker_zscore',
            'macro_features': 'global_zscore',
            'calendar_features': 'none (already in [-1, 1] range)'
        },
        'window': 52,
        'created_at': datetime.now().isoformat()
    }

    # Save outputs
    train_path = output_dir / 'train_weekly.parquet'
    test_path = output_dir / 'test_weekly.parquet'
    metadata_path = output_dir / 'metadata_weekly.json'

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"✓ Train: {len(train_df)} samples, {len(norm_features)} features")
    print(f"  - Ticker features: {len(norm_ticker_features)}")
    print(f"  - Macro features: {len(norm_macro_features)}")
    print(f"  - Calendar features: {len(calendar_cols)} (monthly seasonality)")
    print(f"✓ Test:  {len(test_df)} samples")
    print(f"✓ Saved to: {output_dir}")
    print("="*80)

    return train_df, test_df, metadata


if __name__ == '__main__':
    # Run pipeline
    train_df, test_df, metadata = prepare_weekly_data()

    # Print summary
    print("\nMetadata:")
    print(json.dumps(metadata, indent=2))
