"""
Data Loading and Feature Engineering for Portfolio RL

Simplified version with:
- Cleaner structure
- Removed unused imports
- Helper functions for repetitive patterns
- All critical shifting logic preserved exactly

CRITICAL: Do not modify expanding_zscore or shift logic without careful verification!
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os

from core.config import (
    RAW_DATA_TRAIN_FILE, TEST_SET_FILE, TICKERS,
    MACRO_SYMBOLS, START_DATE, END_DATE, TEST_RATIO,
    MIN_HISTORY, FEATURE_WINDOWS, get_data_paths, OUTPUTS_DIR
)


# ============================================================================
# UTILITIES
# ============================================================================

def expanding_zscore(series, min_periods=20):
    """
    Calculate z-score using ONLY past data (expanding window).
    
    CRITICAL - DO NOT MODIFY WITHOUT VERIFICATION:
    At time t, returns: (series[t-1] - mean(series[0:t-2])) / std(series[0:t-2])
    
    This ensures NO look-ahead bias.
    """
    # Shift series first to avoid using current period data
    series_shifted = series.shift(1)

    # Calculate expanding stats on shifted series, then shift again
    exp_mean = series_shifted.expanding(min_periods=min_periods).mean().shift(1)
    exp_std = series_shifted.expanding(min_periods=min_periods).std().shift(1)
    exp_std = exp_std.fillna(1e-8).replace(0, 1e-8)

    # Normalize and clip
    z = (series_shifted - exp_mean) / exp_std
    return z.clip(-3, 3)


def cross_sectional_zscore(df):
    """
    Calculate cross-sectional z-score (row-wise normalization).
    Each row is normalized to have mean=0, std=1 across columns.
    """
    return df.apply(
        lambda row: (row - row.mean()) / (row.std() + 1e-8),
        axis=1
    ).clip(-3, 3)


def safe_get_column(df, col, default=0):
    """Get column from dataframe with fallback to default value."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_data(tickers=TICKERS, start_date=START_DATE, end_date=END_DATE):
    """Download price, volume, and macro data."""
    print(f"Downloading data from {start_date} to {end_date}...")

    # Equity data
    data = yf.download(tickers, start=start_date, end=end_date, interval='1wk', auto_adjust=False)
    prices = data['Adj Close'].copy()
    volumes = data['Volume'].copy()

    # Macro data
    macro = yf.download(
        list(MACRO_SYMBOLS.values()),
        start=start_date, end=end_date,
        interval='1wk', auto_adjust=False
    )['Adj Close']
    macro.columns = list(MACRO_SYMBOLS.keys())

    return prices, volumes, macro


def prepare_train_test_split(save_files=True):
    """
    Prepare train/test split with NO data leakage.
    Split BEFORE any cleaning to prevent information leak.
    """
    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT (LEAK-FREE)")
    print("=" * 70)

    # Download
    prices_raw, volumes_raw, macro_raw = download_data()

    # Merge for consistent indexing
    prices_full = pd.concat([prices_raw, macro_raw], axis=1).ffill()
    idx = prices_full.index

    # Split point
    split_idx = int(len(idx) * (1 - TEST_RATIO))
    split_date = idx[split_idx]

    print(f"\nSplit date: {split_date}")
    print(f"   Train: {idx[0]} to {split_date}")
    print(f"   Test:  {split_date} to {idx[-1]}")

    # Split BEFORE cleaning
    train_prices = prices_full.iloc[:split_idx].dropna()
    test_prices = prices_full.iloc[split_idx:].dropna()

    # Align other data
    train_volumes = volumes_raw.reindex(train_prices.index)
    test_volumes = volumes_raw.reindex(test_prices.index)

    print(f"\nTrain: {len(train_prices)} weeks, Test: {len(test_prices)} weeks")

    # Save
    if save_files:
        paths = get_data_paths()
        
        train_data = pd.concat([train_prices, train_volumes], axis=1, keys=['prices', 'volumes'])
        test_data = pd.concat([test_prices, test_volumes], axis=1, keys=['prices', 'volumes'])

        train_data.round(4).to_csv(paths['train'])
        test_data.round(4).to_csv(paths['test'])

        print(f"Saved: {paths['train']}")
        print(f"Saved: {paths['test']}")

    return train_prices, train_volumes, test_prices, test_volumes


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(prices, volumes, verbose=False):
    """
    Create features with proper temporal alignment (no look-ahead bias).
    Features are automatically filtered to selected subset via get_lean_features().
    
    Features include:
    - Macro indicators (VIX, yield curve, etc.)
    - Per-ticker: returns, momentum, volatility, technicals
    - Cross-sectional: relative performance vs peers
    - Calendar: month sin/cos encoding
    
    CRITICAL: All features use expanding_zscore or explicit shift(1) to avoid leakage.
    """
    tickers = [t for t in prices.columns if t not in MACRO_SYMBOLS.keys()]
    
    features_list = []

    # --- Macro Features ---
    macro_df = pd.DataFrame(index=prices.index)
    
    tnx = safe_get_column(prices, 'tnx')
    irx = safe_get_column(prices, 'irx')
    
    macro_df['yield_curve_10y2y_norm'] = expanding_zscore(tnx - irx)
    macro_df['vix_norm'] = expanding_zscore(safe_get_column(prices, 'vix'))
    macro_df['dxy_norm'] = expanding_zscore(safe_get_column(prices, 'dxy'))
    macro_df['oil_norm'] = expanding_zscore(safe_get_column(prices, 'oil'))
    macro_df['tnx_norm'] = expanding_zscore(tnx)
    
    features_list.append(macro_df)

    # --- Storage for cross-sectional features ---
    returns_raw = pd.DataFrame(index=prices.index)
    momentum_raw = {w: pd.DataFrame(index=prices.index) for w in FEATURE_WINDOWS}
    volatility_raw = {w: pd.DataFrame(index=prices.index) for w in FEATURE_WINDOWS}

    # --- Per-Ticker Features ---
    for ticker in tickers:
        if ticker not in prices.columns:
            continue

        price = prices[ticker]
        volume = volumes[ticker] if volumes is not None and ticker in volumes.columns else None
        simple_ret = price.pct_change()

        tf = pd.DataFrame(index=prices.index)

        # CRITICAL: Shift returns - at time t, we observe return from t-1
        tf[f'{ticker}_simple_ret'] = simple_ret.shift(1)

        # Store for cross-sectional (unshifted - will shift later)
        returns_raw[ticker] = simple_ret

        # Momentum & Volatility (multiple windows)
        for w in FEATURE_WINDOWS:
            mom = price / price.shift(w) - 1
            vol = simple_ret.rolling(w).std() * np.sqrt(52)
            downside = simple_ret.clip(upper=0).rolling(w).std() * np.sqrt(52)

            momentum_raw[w][ticker] = mom
            volatility_raw[w][ticker] = vol

            tf[f'{ticker}_momentum_{w}w_ts_norm'] = expanding_zscore(mom)
            tf[f'{ticker}_volatility_{w}w_ts_norm'] = expanding_zscore(vol)
            tf[f'{ticker}_downside_vol_{w}w_norm'] = expanding_zscore(downside)

        # Moving Averages
        sma_12 = price.rolling(12, min_periods=1).mean()
        ema_26 = price.ewm(span=26, adjust=False).mean()
        
        tf[f'{ticker}_price_to_sma_12_norm'] = expanding_zscore(price / sma_12 - 1)
        tf[f'{ticker}_price_to_ema_26_norm'] = expanding_zscore(price / ema_26 - 1)

        # MACD
        macd = price.ewm(span=12).mean() - price.ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        tf[f'{ticker}_macd_hist_norm'] = expanding_zscore(macd - signal)

        # Bollinger Bands
        bb_sma = price.rolling(20, min_periods=1).mean()
        bb_std = price.rolling(20, min_periods=1).std()
        tf[f'{ticker}_bb_position_norm'] = expanding_zscore((price - bb_sma) / (2 * bb_std + 1e-8))
        tf[f'{ticker}_bb_width_norm'] = expanding_zscore(4 * bb_std / (bb_sma + 1e-8))

        # RSI & Stochastic (shift explicitly - not through expanding_zscore)
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 1 - 1 / (1 + gain / (loss + 1e-8))

        low_14, high_14 = price.rolling(14).min(), price.rolling(14).max()
        stoch = (price - low_14) / (high_14 - low_14 + 1e-8)

        # CRITICAL: Shift to use last period's indicator values
        tf[f'{ticker}_rsi_14'] = rsi.shift(1)
        tf[f'{ticker}_stochastic_14'] = stoch.shift(1)

        # Drawdown
        rolling_max = price.expanding().max()
        tf[f'{ticker}_drawdown_norm'] = expanding_zscore((price - rolling_max) / (rolling_max + 1e-8))

        # Volume Features
        if volume is not None:
            dollar_vol = price * volume
            obv = (volume * np.sign(simple_ret.shift(1))).cumsum()
            vol_ma_20 = volume.rolling(20).mean()

            tf[f'{ticker}_dollar_volume_norm'] = expanding_zscore(dollar_vol)
            tf[f'{ticker}_obv_norm'] = expanding_zscore(obv)
            tf[f'{ticker}_obv_roc_4w_norm'] = expanding_zscore(obv / obv.shift(4) - 1)
            tf[f'{ticker}_volume_ratio_norm'] = expanding_zscore(volume / (vol_ma_20 + 1e-8))
            tf[f'{ticker}_vol_roc_13w_norm'] = expanding_zscore(volume / volume.shift(13) - 1)
            tf[f'{ticker}_relative_volume_norm'] = expanding_zscore(volume / (volume.rolling(4).mean() + 1e-8))

        features_list.append(tf)

    # --- Cross-Sectional Features ---
    # CRITICAL: Shift BEFORE cross-sectional calculation
    cross_list = []

    # Returns cross-sectional
    ret_shifted = returns_raw.shift(1)
    ret_cross_z = cross_sectional_zscore(ret_shifted)
    ret_cross_z.columns = [f'{t}_return_cross_z' for t in ret_cross_z.columns]
    cross_list.append(ret_cross_z)

    # Momentum & Volatility cross-sectional (per window)
    for w in FEATURE_WINDOWS:
        # Momentum
        mom_shifted = momentum_raw[w].shift(1)
        mom_z = cross_sectional_zscore(mom_shifted)
        mom_z.columns = [f'{t}_momentum_{w}w_cross_z' for t in mom_z.columns]
        mom_rank = mom_shifted.rank(axis=1, pct=True)
        mom_rank.columns = [f'{t}_momentum_{w}w_rank' for t in mom_rank.columns]
        cross_list.extend([mom_z, mom_rank])

        # Volatility
        vol_shifted = volatility_raw[w].shift(1)
        vol_z = cross_sectional_zscore(vol_shifted)
        vol_z.columns = [f'{t}_volatility_{w}w_cross_z' for t in vol_z.columns]
        vol_rank = vol_shifted.rank(axis=1, pct=True)
        vol_rank.columns = [f'{t}_volatility_{w}w_rank' for t in vol_rank.columns]
        cross_list.extend([vol_z, vol_rank])

    features_list.extend(cross_list)

    # --- Combine All Features ---
    features = pd.concat(features_list, axis=1)

    # Calendar features
    features['month_sin'] = np.sin(2 * np.pi * features.index.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * features.index.month / 12)

    # --- Cleanup ---
    # Remove burn-in period (need history for rolling calculations)
    features = features.iloc[MIN_HISTORY:]

    # Clip and drop NaN
    features = features.clip(-3, 3).dropna()

    # Always apply feature filtering for consistency
    features = get_lean_features(features, verbose=verbose)

    # Validate
    features = validate_features(features, verbose=verbose)

    if verbose:
        print(f"\nFeatures: {features.shape[1]} columns, {features.shape[0]} samples")

    return features


# ============================================================================
# FEATURE SELECTION & VALIDATION
# ============================================================================

def get_lean_features(all_features, verbose=True):
    """
    Filter to pre-selected important features.
    Uses selected_features_expanded.json from analyze_features.py
    """
    import json

    json_path = os.path.join(OUTPUTS_DIR, 'selected_features.json')
    csv_path = os.path.join(OUTPUTS_DIR, 'feature_importance.csv')

    # Return columns must always be kept
    ret_cols = [c for c in all_features.columns if '_simple_ret' in c]

    # Try JSON first
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            keep_features = json.load(f)
    elif os.path.exists(csv_path):
        if verbose:
            print("Using feature importance CSV (threshold=0.05)")
        df = pd.read_csv(csv_path)
        keep_features = df[df['Importance'] > 0.05]['Feature'].tolist()
    else:
        if verbose:
            print("No feature selection file found, using all features")
        return all_features

    # Filter to available features
    available = set(all_features.columns)
    valid_features = [f for f in keep_features if f in available]

    if len(valid_features) < 10:
        if verbose:
            print(f"Warning: Only {len(valid_features)} valid features, using all")
        return all_features

    # Combine with return columns
    final = list(set(valid_features + ret_cols))

    if verbose:
        print(f"Lean mode: {len(final)}/{len(all_features.columns)} features")

    return all_features[final]


def validate_features(features_df, verbose=False):
    """Remove problematic features (inf, excessive NaN, zero variance)."""
    initial = features_df.shape[1]

    # Remove inf
    inf_cols = features_df.columns[np.isinf(features_df).any()].tolist()
    features_df = features_df.drop(columns=inf_cols) if inf_cols else features_df

    # Remove >50% NaN
    nan_cols = features_df.columns[features_df.isnull().mean() > 0.5].tolist()
    features_df = features_df.drop(columns=nan_cols) if nan_cols else features_df

    removed = initial - features_df.shape[1]
    if verbose and removed > 0:
        print(f"Removed {removed} problematic features")

    return features_df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("DATA ENGINEERING PIPELINE")
    print("=" * 70)

    # Prepare data
    train_prices, train_volumes, test_prices, test_volumes = \
        prepare_train_test_split(save_files=True)

    # Create features
    train_features = create_features(train_prices, train_volumes, verbose=True)

    print(f"\nTrain features: {train_features.shape}")
    print(f"Date range: {train_features.index[0]} to {train_features.index[-1]}")

    # Save
    out_path = os.path.join(OUTPUTS_DIR, 'train_features.csv')
    train_features.round(4).to_csv(out_path)
    print(f"Saved: {out_path}")