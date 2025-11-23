import yfinance as yf
import pandas as pd
import numpy as np
import ta
import argparse
from core.config import RAW_DATA_TRAIN_FILE, BENCHMARK_FILE, TEST_SET_FILE, TICKERS, MACRO_SYMBOLS, START_DATE, END_DATE, TEST_RATIO, MIN_HISTORY, FEATURE_WINDOWS, get_data_paths

# --- Expanding Z-Score (No Leakage) ---
def expanding_zscore(series, min_periods=20):
    """Calculate z-score using ONLY past data (expanding window)"""
    exp_mean = series.expanding(min_periods=min_periods).mean().shift(1)
    exp_std = series.expanding(min_periods=min_periods).std().shift(1)
    exp_std = exp_std.fillna(1e-8).replace(0, 1e-8)
    z = (series - exp_mean) / exp_std
    return z.clip(-3, 3)


def create_expanded_features(prices, volumes=None, tickers=TICKERS):
    """
    Create EXPANDED feature set with all technical indicators
    
    Features per ticker (~25 features × 7 tickers = 175):
    - Returns (1)
    - Momentum (4 windows: 1w, 4w, 13w, 52w)
    - Volatility (4 windows: 4w, 13w, 26w, 52w)
    - Downside Volatility (2 windows: 4w, 26w)
    - Technical Indicators (9): SMA, EMA, MACD, BB Position, BB Width, RSI, Stochastic, Drawdown
    - Volume (5): OBV, OBV ROC, Volume Ratio, Volume ROC, Relative Volume
    
    Macro (5): VIX, DXY, Oil, 10Y Treasury, Yield Curve (10Y-2Y)
    Calendar (2): Month sin/cos
    
    Total: ~180 features
    """
    print("Creating EXPANDED feature set with rich technical indicators...")
    
    features_dict = {}
    
    # --- Macro Features ---
    if 'tnx' in prices.columns and 'irx' in prices.columns:
        # Yield curve: 10Y - 2Y equivalent (using 13-week as proxy for 2Y)
        yield_curve = prices['tnx'] - prices['irx']
        features_dict['yield_curve_10y2y_norm'] = expanding_zscore(yield_curve)
    
    key_macros = ['vix', 'dxy', 'tnx']
    for col in key_macros:
        if col in prices.columns:
            features_dict[f'{col}_norm'] = expanding_zscore(prices[col])
    
    # Oil (if available)
    if 'oil' in prices.columns:
        features_dict['oil_wti_norm'] = expanding_zscore(prices['oil'])
    
    # --- Per-Ticker Features ---
    for ticker in tickers:
        if ticker not in prices.columns or ticker == 'QQQ':
            continue
        
        price = prices[ticker]
        simple_returns = price.pct_change()
        
        # 1. Returns (lagged to avoid leakage)
        features_dict[f'{ticker}_simple_ret'] = simple_returns.shift(1)
        
        # 2. Momentum (multiple windows)
        for window in FEATURE_WINDOWS:
            momentum = (price / price.shift(window) - 1).shift(1)
            features_dict[f'{ticker}_momentum_{window}w_norm'] = expanding_zscore(momentum)
        
        # 3. Volatility (standard)
        for window in FEATURE_WINDOWS:
            vol = simple_returns.rolling(window, min_periods=1).std() * np.sqrt(52)
            features_dict[f'{ticker}_volatility_{window}w_norm'] = expanding_zscore(vol.shift(1))
        
        # 4. Downside Volatility (Sortino-style)
        for window in [4, 26]:
            downside_returns = simple_returns.copy()
            downside_returns[downside_returns > 0] = 0
            downside_vol = downside_returns.rolling(window, min_periods=1).std() * np.sqrt(52)
            features_dict[f'{ticker}_downside_vol_{window}w_norm'] = expanding_zscore(downside_vol.shift(1))
        
        # 5. SMA-based features
        sma_12 = price.rolling(12, min_periods=1).mean()
        price_to_sma = (price / sma_12 - 1).shift(1)
        features_dict[f'{ticker}_price_to_sma_12w_norm'] = expanding_zscore(price_to_sma)
        
        # 6. EMA-based features
        ema_26 = price.ewm(span=26, adjust=False).mean()
        price_to_ema = (price / ema_26 - 1).shift(1)
        features_dict[f'{ticker}_price_to_ema_26w_norm'] = expanding_zscore(price_to_ema)
        
        # 7. MACD Histogram
        ema_12 = price.ewm(span=12, adjust=False).mean()
        ema_26_macd = price.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26_macd
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        features_dict[f'{ticker}_macd_histogram_norm'] = expanding_zscore(macd_histogram.shift(1))
        
        # 8. Bollinger Bands
        sma_20 = price.rolling(20, min_periods=1).mean()
        std_20 = price.rolling(20, min_periods=1).std()
        bb_upper = sma_20 + (2 * std_20)
        bb_lower = sma_20 - (2 * std_20)
        
        # BB Position: (price - lower) / (upper - lower)
        bb_position = (price - bb_lower) / (bb_upper - bb_lower + 1e-8)
        features_dict[f'{ticker}_bb_position_norm'] = expanding_zscore(bb_position.shift(1))
        
        # BB Width: (upper - lower) / sma
        bb_width = (bb_upper - bb_lower) / (sma_20 + 1e-8)
        features_dict[f'{ticker}_bb_width_norm'] = expanding_zscore(bb_width.shift(1))
        
        # 9. RSI
        rsi = ta.momentum.RSIIndicator(price, window=14).rsi() / 100.0
        features_dict[f'{ticker}_rsi_14d_norm'] = expanding_zscore(rsi.shift(1))
        
        # 10. Stochastic Oscillator
        stochastic = ta.momentum.StochasticOscillator(
            high=price, low=price, close=price, window=14
        ).stoch() / 100.0
        features_dict[f'{ticker}_stochastic_14w_norm'] = expanding_zscore(stochastic.shift(1))
        
        # 11. Drawdown
        rolling_max = price.expanding(min_periods=1).max()
        drawdown = (price / rolling_max - 1).shift(1)
        features_dict[f'{ticker}_current_drawdown_norm'] = expanding_zscore(drawdown)
        
        # 12. Volume Features (if available)
        if volumes is not None and ticker in volumes.columns:
            volume = volumes[ticker]
            
            # OBV (On-Balance Volume)
            obv = (np.sign(simple_returns) * volume).fillna(0).cumsum()
            features_dict[f'{ticker}_obv_norm'] = expanding_zscore(obv.shift(1))
            
            # OBV Rate of Change
            obv_roc = obv.pct_change(periods=4).shift(1)
            features_dict[f'{ticker}_obv_roc_4w_norm'] = expanding_zscore(obv_roc)
            
            # Volume Ratio (vs 20-week MA)
            volume_ma_20 = volume.rolling(20, min_periods=1).mean()
            volume_ratio = (volume / (volume_ma_20 + 1e-8)).shift(1)
            features_dict[f'{ticker}_volume_ratio_20w_norm'] = expanding_zscore(volume_ratio)
            
            # Volume Rate of Change
            volume_roc = volume.pct_change(periods=13).shift(1)
            features_dict[f'{ticker}_volume_roc_13w_norm'] = expanding_zscore(volume_roc)
            
            # Relative Volume (current vs 4-week avg)
            volume_ma_4 = volume.rolling(4, min_periods=1).mean()
            relative_volume = (volume / (volume_ma_4 + 1e-8)).shift(1)
            features_dict[f'{ticker}_relative_volume_4w_norm'] = expanding_zscore(relative_volume)
    
    # --- Calendar Features ---
    features_dict['month_sin'] = np.sin(2 * np.pi * prices.index.month / 12)
    features_dict['month_cos'] = np.cos(2 * np.pi * prices.index.month / 12)
    
    # Create DataFrame
    features_df = pd.DataFrame(features_dict, index=prices.index)
    
    # Clean up
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.iloc[MIN_HISTORY:].dropna()
    
    print(f"Expanded features shape: {features_df.shape}")
    
    return features_df


# --- Data Download (same as original) ---
def download_data_with_volume(tickers=TICKERS, start_date=START_DATE, end_date=END_DATE):
    """Download price AND volume data"""
    print(f"Downloading price and volume data from {start_date} to {end_date}...")

    # Download price and volume data
    data = yf.download(
        tickers, 
        start=start_date, 
        end=end_date, 
        interval='1wk', 
        auto_adjust=False
    )

    prices = data['Adj Close'].copy()
    volumes = data['Volume'].copy()

    # Macro data extraction
    macro_raw = yf.download(
        list(MACRO_SYMBOLS.values()), 
        start=start_date, 
        end=end_date, 
        interval='1wk', 
        auto_adjust=False
    )['Adj Close']
    macro_raw.columns = list(MACRO_SYMBOLS.keys())

    # Benchmark (QQQ) - only price, no volume needed
    qqq_data = yf.download(
        'QQQ', 
        start=start_date, 
        end=end_date, 
        interval='1wk', 
        auto_adjust=False
    )
    qqq_price = qqq_data['Adj Close'].copy()
    qqq_price.name = 'QQQ'
    
    # Combine prices with macro data
    prices = pd.concat([prices, macro_raw], axis=1).ffill().dropna()
    
    # Align QQQ with main data index (forward fill to match dates)
    qqq_price = qqq_price.reindex(prices.index, method='ffill')

    print(f"Price data shape: {prices.shape}")
    print(f"Volume data shape: {volumes.shape}")
    print(f"QQQ benchmark aligned: {len(qqq_price)} weeks")

    return prices, volumes, qqq_price


# --- Train/Test Split (same as original) ---
def prepare_train_test_split(save_files=True, use_expanded=True):
    """
    Prepare data with proper train/test split
    use_expanded: If True, save to expanded directory (should always be True for this file)
    """
    prices, volumes, qqq_price = download_data_with_volume()

    test_size = int(len(prices) * TEST_RATIO)
    train_val_prices = prices.iloc[:-test_size].copy()
    test_prices = prices.iloc[-test_size:].copy()

    train_val_volumes = volumes.iloc[:-test_size].copy()
    test_volumes = volumes.iloc[-test_size:].copy()

    train_val_qqq = qqq_price.iloc[:-test_size].copy()
    test_qqq = qqq_price.iloc[-test_size:].copy()

    print(f"\nData Split:")
    print(f"Train+Val period: {train_val_prices.index[0]} to {train_val_prices.index[-1]} ({len(train_val_prices)} weeks)")
    print(f"Test period: {test_prices.index[0]} to {test_prices.index[-1]} ({len(test_prices)} weeks)")

    if save_files:
        # Get mode-specific paths - always use expanded mode for this file
        data_paths = get_data_paths(expanded_mode=True)
        
        print(f"\nSaving data files (EXPANDED mode)...")
        print(f"   Directory: {data_paths['data_dir']}")
        
        train_val_data = pd.concat([train_val_prices, train_val_volumes.add_suffix('_volume')], axis=1)
        train_val_data.to_csv(data_paths['train'])

        if isinstance(train_val_qqq, pd.Series):
            train_val_qqq.to_frame('QQQ').to_csv(data_paths['benchmark'])
        else:
            train_val_qqq.to_csv(data_paths['benchmark'])

        test_qqq_df = test_qqq.to_frame('QQQ') if isinstance(test_qqq, pd.Series) else test_qqq
        test_data = pd.concat([
            test_prices,
            test_volumes.add_suffix('_volume'),
            test_qqq_df
        ], axis=1)
        test_data.to_csv(data_paths['test'])

        print(f"\n✅ Saved training data to {data_paths['train']}")
        print(f"✅ Saved QQQ benchmark to {data_paths['benchmark']}")
        print(f"✅ Saved test set to {data_paths['test']}")

    return train_val_prices, train_val_volumes, test_prices, test_volumes, train_val_qqq, test_qqq


# --- Compatibility ---
def create_features_no_leakage(prices, tickers, volumes=None):
    """Compatibility wrapper - uses expanded features"""
    return create_expanded_features(prices, volumes, tickers)


# --- Main Execution ---
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"DATA PREPARATION (EXPANDED MODE - 175+ features)")
    print(f"{'='*60}")
    
    train_prices, train_volumes, test_prices, test_volumes, train_qqq, test_qqq = prepare_train_test_split()

    print("\nCreating EXPANDED features for train+val set...")
    train_features = create_expanded_features(train_prices, train_volumes)

    print("\n" + "="*60)
    print("EXPANDED FEATURE SUMMARY")
    print("="*60)

    categories = {
        'Returns': [f for f in train_features.columns if 'simple_ret' in f],
        'Momentum': [f for f in train_features.columns if 'momentum' in f],
        'Volatility (Standard)': [f for f in train_features.columns if 'volatility' in f and 'downside' not in f],
        'Volatility (Downside)': [f for f in train_features.columns if 'downside_vol' in f],
        'SMA/EMA': [f for f in train_features.columns if any(x in f for x in ['sma', 'ema'])],
        'MACD': [f for f in train_features.columns if 'macd' in f],
        'Bollinger Bands': [f for f in train_features.columns if 'bb_' in f],
        'RSI/Stochastic': [f for f in train_features.columns if any(x in f for x in ['rsi', 'stochastic'])],
        'Drawdown': [f for f in train_features.columns if 'drawdown' in f],
        'Volume': [f for f in train_features.columns if any(v in f for v in ['obv', 'volume'])],
        'Macro': [f for f in train_features.columns if any(m in f for m in ['vix', 'dxy', 'tnx', 'yield', 'oil'])],
        'Calendar': [f for f in train_features.columns if 'month' in f]
    }

    for category, feats in categories.items():
        count = len(feats)
        print(f"{category:25} : {count:3} features")

    print(f"\n{'Total features:':25} {train_features.shape[1]:3}")
    print(f"{'Sample size:':25} {train_features.shape[0]:3}")
    print(f"Feature/sample ratio: 1:{train_features.shape[0]/train_features.shape[1]:.1f}")
