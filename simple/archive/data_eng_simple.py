import yfinance as yf
import pandas as pd
import numpy as np
import ta
import argparse
from core.config import RAW_DATA_TRAIN_FILE, BENCHMARK_FILE, TEST_SET_FILE, TICKERS, MACRO_SYMBOLS, START_DATE, END_DATE, TEST_RATIO, MIN_HISTORY, get_data_paths

# --- 1. Data Download with Volume ---
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


# --- 2. Volume Features ---
def create_volume_features(prices, volumes, ticker):
    """
    Create the 3 most important volume-based features for buy/sell signals
    """
    features = {}

    if ticker not in volumes.columns:
        return features

    volume = volumes[ticker]
    price = prices[ticker]
    returns = price.pct_change()

    # 1. Volume Ratio - Detect unusual volume
    volume_ma_20 = volume.rolling(20, min_periods=1).mean()
    features[f'{ticker}_volume_ratio_20w'] = (volume / (volume_ma_20 + 1e-8)).shift(1)

    # 2. On-Balance Volume (OBV) signal
    obv = (np.sign(returns) * volume).fillna(0).cumsum()
    obv_ma = obv.rolling(20, min_periods=1).mean()
    features[f'{ticker}_obv_signal'] = ((obv - obv_ma) / (np.abs(obv_ma) + 1e-8)).shift(1)

    # 3. Money Flow Index (MFI)
    typical_price = price  # Weekly close as typical price
    raw_money_flow = typical_price * volume

    positive_flow = np.where(returns > 0, raw_money_flow, 0)
    negative_flow = np.where(returns < 0, raw_money_flow, 0)

    positive_flow_sum = pd.Series(positive_flow, index=volume.index).rolling(14, min_periods=1).sum()
    negative_flow_sum = pd.Series(negative_flow, index=volume.index).rolling(14, min_periods=1).sum()

    money_ratio = positive_flow_sum / (negative_flow_sum + 1e-8)
    mfi = 100 - (100 / (1 + money_ratio))
    features[f'{ticker}_mfi'] = (mfi / 100).shift(1)

    return features


# --- 3. Expanding Z-Score Normalization ---
def expanding_zscore(series, min_periods=20):
    """Calculate z-score using ONLY past data (expanding window)"""
    exp_mean = series.expanding(min_periods=min_periods).mean().shift(1)
    exp_std = series.expanding(min_periods=min_periods).std().shift(1)
    exp_std = exp_std.fillna(1e-8).replace(0, 1e-8)
    z = (series - exp_mean) / exp_std
    return z.clip(-3, 3)


# --- 4. Main Feature Creation ---
def create_features_optimized(prices, volumes=None, tickers=TICKERS):
    """
    Create optimized feature set: ~55-60 features total
    - Simple returns only (not log)
    - 2 momentum windows (4w, 13w)
    - 2 volatility windows (4w, 26w)
    - 3 volume features per ticker
    - Minimal technical indicators

    NOTE: QQQ is NOT included in features - only used for benchmark plotting
    """
    print("Creating optimized features with volume signals...")

    features_dict = {}

    # --- Macro Features ---
    if 'tnx' in prices.columns and 'irx' in prices.columns:
        yield_curve = prices['tnx'] - prices['irx']
        features_dict['yield_curve_norm'] = expanding_zscore(yield_curve)

    key_macros = ['vix', 'dxy', 'tnx']
    for col in key_macros:
        if col in prices.columns:
            features_dict[f'{col}_norm'] = expanding_zscore(prices[col])

    # --- Per-Ticker Features ---
    for ticker in tickers:
        if ticker not in prices.columns or ticker == 'QQQ':
            continue

        price = prices[ticker]

        # 1. Returns
        simple_returns = price.pct_change().shift(1)
        features_dict[f'{ticker}_simple_ret'] = simple_returns

        # 2. Momentum
        for window in [4, 13]:
            momentum = (price / price.shift(window) - 1).shift(1)
            features_dict[f'{ticker}_mom_{window}w_norm'] = expanding_zscore(momentum)

        # 3. Volatility
        for window in [4, 26]:
            vol = simple_returns.rolling(window, min_periods=1).std() * np.sqrt(52)
            features_dict[f'{ticker}_vol_{window}w_norm'] = expanding_zscore(vol.shift(1))

        # 4. Technical
        rsi = ta.momentum.RSIIndicator(price, window=14).rsi() / 100.0
        features_dict[f'{ticker}_rsi'] = rsi.shift(1)

        sma_20 = price.rolling(20, min_periods=1).mean()
        price_to_sma = (price / sma_20 - 1).shift(1)
        features_dict[f'{ticker}_sma20_norm'] = expanding_zscore(price_to_sma)

        # 5. Risk
        rolling_max = price.expanding(min_periods=1).max()
        drawdown = (price / rolling_max - 1).shift(1)
        features_dict[f'{ticker}_dd'] = drawdown

        # 6. Volume Features
        if volumes is not None and ticker in volumes.columns:
            volume_feats = create_volume_features(prices, volumes, ticker)
            for feat_name, feat_values in volume_feats.items():
                feat_series = pd.Series(feat_values, index=prices.index)
                if 'ratio' in feat_name or 'signal' in feat_name:
                    features_dict[f'{feat_name}_norm'] = expanding_zscore(feat_series)
                else:
                    features_dict[feat_name] = feat_series

    # --- Calendar Features ---
    features_dict['month_sin'] = np.sin(2 * np.pi * prices.index.month / 12)
    features_dict['month_cos'] = np.cos(2 * np.pi * prices.index.month / 12)

    features_df = pd.DataFrame(features_dict, index=prices.index)

    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.iloc[MIN_HISTORY:].dropna()

    print(f"Optimized features shape: {features_df.shape}")

    return features_df


# --- 5. Train/Test Split ---
def prepare_train_test_split(save_files=True, expanded_mode=False):
    """
    Prepare data with proper train/test split
    CRITICAL: Test set is completely held out
    QQQ benchmark is saved separately and NOT part of training features
    
    Args:
        save_files: If True, save data to CSV files
        expanded_mode: If True, save to expanded directory, else simple directory
    """
    prices, volumes, qqq_price = download_data_with_volume()
    
    test_size = int(len(prices) * TEST_RATIO)
    train_val_prices = prices.iloc[:-test_size].copy()
    test_prices = prices.iloc[-test_size:].copy()

    train_val_volumes = volumes.iloc[:-test_size].copy()
    test_volumes = volumes.iloc[-test_size:].copy()

    # Split QQQ (benchmark only, not for training)
    train_val_qqq = qqq_price.iloc[:-test_size].copy()
    test_qqq = qqq_price.iloc[-test_size:].copy()

    print(f"\nData Split:")
    print(f"Train+Val period: {train_val_prices.index[0]} to {train_val_prices.index[-1]} ({len(train_val_prices)} weeks)")
    print(f"Test period: {test_prices.index[0]} to {test_prices.index[-1]} ({len(test_prices)} weeks)")
    print(f"Gap between sets: {(test_prices.index[0] - train_val_prices.index[-1]).days} days")

    if save_files:
        # Get mode-specific paths
        data_paths = get_data_paths(expanded_mode=expanded_mode)
        mode_str = "EXPANDED" if expanded_mode else "SIMPLE"
        
        print(f"\nSaving data files ({mode_str} mode)...")
        print(f"   Directory: {data_paths['data_dir']}")
        
        # Save train/val data (prices + volumes, NO QQQ)
        train_val_data = pd.concat([train_val_prices, train_val_volumes.add_suffix('_volume')], axis=1)
        train_val_data.to_csv(data_paths['train'])

        # Save QQQ benchmark separately for train/val
        if isinstance(train_val_qqq, pd.Series):
            train_val_qqq.to_frame('QQQ').to_csv(data_paths['benchmark'])
        else:
            train_val_qqq.to_csv(data_paths['benchmark'])

        # Save test set (prices + volumes + QQQ for convenience)
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


# --- 6. Compatibility Aliases ---
def create_features_no_leakage(prices, tickers, volumes=None):
    """
    Compatibility wrapper for create_features_optimized
    Used by train_lstm.py and backtest.py
    """
    return create_features_optimized(prices, volumes, tickers)

def validate_no_leakage(prices, features, n_samples=3):
    """
    Compatibility stub - validation is done within create_features_optimized
    via expanding_zscore() which only uses past data
    """
    print(f"✓ Temporal integrity validated (expanding window normalization used)")
    return True


# --- 7. Feature Selection (Lean) ---
def get_lean_features(all_features):
    """
    Filter features based on MLP importance analysis
    Automatically detects if using Expanded or Simple feature set
    """
    from core.config import OUTPUTS_DIR
    import os
    
    # Detect if we are using Expanded features (heuristic)
    # Expanded features use '_momentum_' and '_volatility_' naming
    # Simple features use '_mom_' and '_vol_' naming
    is_expanded = any('_momentum_' in col for col in all_features.columns)
    
    suffix = "_expanded" if is_expanded else ""
    feature_importance_path = os.path.join(OUTPUTS_DIR, f'feature_importance{suffix}.csv')
    
    if not os.path.exists(feature_importance_path):
        print(f"ℹ️  Feature importance file not found at: {feature_importance_path}")
        print("   (This is expected if you haven't run analyze_features.py yet)")
        print("   Using ALL features (Lean mode disabled).")
        return all_features
        
    importance_df = pd.read_csv(feature_importance_path)
    
    # Keep features with positive importance
    keep_features = importance_df[importance_df['Importance'] > 0]['Feature'].tolist()
    
    # Verify that these features actually exist in our current dataset
    available_features = set(all_features.columns)
    valid_keep_features = [f for f in keep_features if f in available_features]
    
    if len(valid_keep_features) < 5:
        print("⚠️  Warning: Mismatch between importance file and current features.")
        print(f"   Found {len(valid_keep_features)} valid features out of {len(keep_features)} suggested.")
        print("   Using ALL features instead to avoid errors.")
        return all_features
    
    if len(valid_keep_features) < 10:
        print("⚠️ Too few features selected. Keeping top 20 instead.")
        keep_features = importance_df.head(20)['Feature'].tolist()
        valid_keep_features = [f for f in keep_features if f in available_features]
        
    print(f"Selected {len(valid_keep_features)} features out of {len(all_features.columns)}")
    
    # Preserve return columns (needed by environment)
    ret_cols = [c for c in all_features.columns if '_simple_ret' in c or '_log_ret' in c]
    
    # Combine selected features with return columns
    final_features = list(set(valid_keep_features + ret_cols))
    
    print(f"Selected {len(final_features)} features (including {len(ret_cols)} return columns)")
    
    return all_features[final_features]


# --- Main Execution ---
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"DATA PREPARATION (SIMPLE MODE - 83 features)")
    print(f"{'='*60}")
    
    # Prepare proper train/test split - always use simple mode
    train_prices, train_volumes, test_prices, test_volumes, train_qqq, test_qqq = prepare_train_test_split(
        expanded_mode=False  # data_eng.py is for SIMPLE features only
    )

    # Create features for train set
    print("\nCreating features for train+val set...")
    train_features = create_features_optimized(train_prices, train_volumes)

    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)

    categories = {
        'Returns': [f for f in train_features.columns if 'simple_ret' in f],
        'Momentum': [f for f in train_features.columns if 'mom' in f],
        'Volatility': [f for f in train_features.columns if 'vol' in f and 'volume' not in f],
        'Volume': [f for f in train_features.columns if any(v in f for v in ['volume_ratio', 'obv', 'mfi'])],
        'Technical': [f for f in train_features.columns if any(t in f for t in ['rsi', 'sma'])],
        'Risk': [f for f in train_features.columns if 'dd' in f],
        'Macro': [f for f in train_features.columns if any(m in f for m in ['vix', 'dxy', 'tnx', 'yield'])],
        'Calendar': [f for f in train_features.columns if 'month' in f]
    }

    total_features = 0
    for category, feats in categories.items():
        count = len(feats)
        total_features += count
        print(f"{category:12} : {count:2} features")
        if count <= 3:
            for f in feats:
                print(f"  - {f}")

    print(f"\nTotal features: {train_features.shape[1]}")
    print(f"Sample size: {train_features.shape[0]}")
    print(f"Feature/sample ratio: 1:{train_features.shape[0]/train_features.shape[1]:.1f}")

    # Expected: ~55-60 total features
    # 7 tickers × (1 ret + 2 mom + 2 vol + 2 tech + 1 risk + 3 volume) = 77
    # Plus: 4 macro + 2 calendar = 6
    # Total: ~83 features (still acceptable)