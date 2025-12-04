import yfinance as yf
import pandas as pd
import numpy as np
import ta
import argparse
import os
from core.config import (
    RAW_DATA_TRAIN_FILE, BENCHMARK_FILE, TEST_SET_FILE, TICKERS, 
    MACRO_SYMBOLS, START_DATE, END_DATE, TEST_RATIO, MIN_HISTORY, 
    FEATURE_WINDOWS, get_data_paths, OUTPUTS_DIR
)

# ============================================================================
# UTILITIES
# ============================================================================

def expanding_zscore(series, min_periods=20):
    """Calculate z-score using ONLY past data (expanding window)"""
    exp_mean = series.expanding(min_periods=min_periods).mean().shift(1)
    exp_std = series.expanding(min_periods=min_periods).std().shift(1)
    exp_std = exp_std.fillna(1e-8).replace(0, 1e-8)
    z = (series - exp_mean) / exp_std
    return z.clip(-3, 3)

def safe_get_macro(df, sym, default=0):
    """Get macro symbol with fallback"""
    if sym in df.columns:
        return df[sym]
    print(f"âš ï¸  Warning: {sym} not available, using {default}")
    return pd.Series(default, index=df.index)

# ============================================================================
# DATA DOWNLOAD (NO LEAKAGE)
# ============================================================================

def download_data_with_volume_raw(tickers=TICKERS, start_date=START_DATE, end_date=END_DATE):
    """Download raw data without cleaning - for leak-free splitting"""
    print(f"ðŸ“¥ Downloading data from {start_date} to {end_date}...")
    
    # Download equity data
    data = yf.download(tickers, start=start_date, end=end_date, interval='1wk', auto_adjust=False)
    prices = data['Adj Close'].copy()
    volumes = data['Volume'].copy()
    
    # Download macro data
    macro_raw = yf.download(
        list(MACRO_SYMBOLS.values()), 
        start=start_date, 
        end=end_date, 
        interval='1wk', 
        auto_adjust=False
    )['Adj Close']
    macro_raw.columns = list(MACRO_SYMBOLS.keys())
    
    # Download benchmark
    qqq_data = yf.download('QQQ', start=start_date, end=end_date, interval='1wk', auto_adjust=False)
    qqq_price = qqq_data['Adj Close'].copy()
    qqq_price.name = 'QQQ'
    
    return prices, volumes, macro_raw, qqq_price

def prepare_train_test_split_fixed(save_files=True):
    """
    Prepare train/test split with NO LEAKAGE
    - Split BEFORE cleaning
    - Clean train and test independently
    """
    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT (LEAK-FREE)")
    print("="*70)
    
    # Download raw data
    prices_raw, volumes_raw, macro_raw, qqq_raw = download_data_with_volume_raw()
    
    # Merge prices with macro for consistent indexing (but don't clean yet)
    prices_full = pd.concat([prices_raw, macro_raw], axis=1).ffill()
    idx = prices_full.index
    
    # Calculate split point
    split_date_idx = int(len(idx) * (1 - TEST_RATIO))
    split_date = idx[split_date_idx]
    
    print(f"\nðŸ“Š Split date: {split_date}")
    print(f"   Train: {idx[0]} to {split_date}")
    print(f"   Test:  {split_date} to {idx[-1]}")
    
    # Split BEFORE cleaning (critical for no leakage)
    train_idx = idx[:split_date_idx]
    test_idx = idx[split_date_idx:]
    
    # Clean independently
    train_prices = prices_full.loc[train_idx].dropna()
    test_prices = prices_full.loc[test_idx].dropna()
    
    # Align volumes
    train_volumes = volumes_raw.loc[train_prices.index]
    test_volumes = volumes_raw.loc[test_prices.index]
    
    # Align QQQ
    train_qqq = qqq_raw.reindex(train_prices.index, method='ffill')
    test_qqq = qqq_raw.reindex(test_prices.index, method='ffill')
    
    print(f"\nâœ… Train size: {len(train_prices)} weeks")
    print(f"âœ… Test size:  {len(test_prices)} weeks")
    
    # Optionally save
    if save_files:
        paths = get_data_paths(expanded_mode=True)
        # Flatten columns for compatibility
        train_volumes_flat = train_volumes.add_suffix('_volume')
        test_volumes_flat = test_volumes.add_suffix('_volume')
        
        train_data = pd.concat([train_prices, train_volumes_flat], axis=1)
        test_data = pd.concat([test_prices, test_volumes_flat], axis=1)
        
        train_data.to_csv(paths['train'])
        test_data.to_csv(paths['test'])
        
        print(f"DEBUG: train_qqq type: {type(train_qqq)}")
        print(f"DEBUG: train_qqq shape: {train_qqq.shape if hasattr(train_qqq, 'shape') else 'N/A'}")
        
        if isinstance(train_qqq, pd.DataFrame):
            train_qqq_df = train_qqq
            if 'QQQ' not in train_qqq_df.columns and train_qqq_df.shape[1] == 1:
                train_qqq_df.columns = ['QQQ']
        elif isinstance(train_qqq, pd.Series):
            train_qqq_df = train_qqq.to_frame('QQQ')
        else:
            # Fallback for numpy array or list
            train_qqq_df = pd.DataFrame({'QQQ': np.squeeze(train_qqq)}, index=train_prices.index)
            
        train_qqq_df.to_csv(paths['benchmark'])
        
        print(f"\nðŸ’¾ Saved to:")
        print(f"   - {paths['train']}")
        print(f"   - {paths['test']}")
        print(f"   - {paths['benchmark']}")
    
    return train_prices, train_volumes, test_prices, test_volumes, train_qqq, test_qqq

# ============================================================================
# FEATURE ENGINEERING (WITH CROSS-SECTIONAL NORMALIZATION)
# ============================================================================

def create_features(prices, volumes=None, qqq=None, lean=True):
    """
    Create features with DUAL normalization:
    1. Time-series z-score (per-ticker vs own history)
    2. Cross-sectional z-score (per-ticker vs portfolio at each time point)
    3. Cross-sectional percentile ranks
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING (DUAL NORMALIZATION)")
    print("="*70)
    
    features_list = []
    
    # Initialize base DataFrame with index
    base_df = pd.DataFrame(index=prices.index)
    features_list.append(base_df)
    tickers = [t for t in prices.columns if t not in MACRO_SYMBOLS.keys()]
    
    # Handle QQQ (Optional)
    if qqq is not None:
        # Ensure qqq is a Series
        if isinstance(qqq, pd.DataFrame):
            qqq = qqq.squeeze()
        qqq_returns = qqq.pct_change()
    else:
        qqq_returns = None
    
    # ========================================================================
    # MACRO FEATURES
    # ========================================================================
    print("\nðŸ“ˆ Creating macro features...")
    
    tnx = safe_get_macro(prices, 'tnx')
    irx = safe_get_macro(prices, 'irx')
    vix = safe_get_macro(prices, 'vix')
    dxy = safe_get_macro(prices, 'dxy')
    wti = safe_get_macro(prices, 'wti')
    
    # Yield curve
    yield_curve = tnx - irx
    
    macro_df = pd.DataFrame(index=prices.index)
    macro_df['yield_curve_10y2y_norm'] = expanding_zscore(yield_curve.shift(1))
    
    # Other macros
    macro_df['vix_norm'] = expanding_zscore(vix.shift(1))
    macro_df['dxy_norm'] = expanding_zscore(dxy.shift(1))
    macro_df['wti_norm'] = expanding_zscore(wti.shift(1))
    macro_df['tnx_norm'] = expanding_zscore(tnx.shift(1))
    
    features_list.append(macro_df)
    
    # ========================================================================
    # PREPARE CROSS-SECTIONAL DATA STRUCTURES
    # ========================================================================
    print(f"ðŸ“Š Creating features for {len(tickers)} tickers...")
    
    # DataFrames to hold raw values for cross-sectional normalization
    returns_df = pd.DataFrame(index=prices.index)
    momentum_dfs = {window: pd.DataFrame(index=prices.index) for window in FEATURE_WINDOWS}
    volatility_dfs = {window: pd.DataFrame(index=prices.index) for window in FEATURE_WINDOWS}
    
    # ========================================================================
    # PER-TICKER TIME-SERIES FEATURES
    # ========================================================================
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
            
        price = prices[ticker]
        volume = volumes[ticker] if (volumes is not None and ticker in volumes.columns) else None
        
        # Simple returns (always included)
        simple_returns = price.pct_change()
        returns_df[ticker] = simple_returns
        # Collect all features for this ticker
        ticker_feats = pd.DataFrame(index=prices.index)
        ticker_feats[f'{ticker}_simple_ret'] = simple_returns
        # Log returns
        log_returns = np.log(price / price.shift(1))
        ticker_feats[f'{ticker}_log_ret'] = log_returns
        
        # Momentum
        for window in FEATURE_WINDOWS:
            momentum = (price / price.shift(window) - 1)
            momentum_dfs[window][ticker] = momentum
            ticker_feats[f'{ticker}_momentum_{window}w_ts_norm'] = expanding_zscore(momentum)
            
        # Volatility
        for window in FEATURE_WINDOWS:
            volatility = simple_returns.rolling(window).std() * np.sqrt(52)
            volatility_dfs[window][ticker] = volatility
            ticker_feats[f'{ticker}_volatility_{window}w_ts_norm'] = expanding_zscore(volatility)
            
        # Downside Volatility
        for window in FEATURE_WINDOWS:
            negative_returns = simple_returns.clip(upper=0)
            downside_vol = negative_returns.rolling(window).std() * np.sqrt(52)
            ticker_feats[f'{ticker}_downside_vol_{window}w_norm'] = expanding_zscore(downside_vol)
            
        # Moving Averages
        sma_12 = price.rolling(12, min_periods=1).mean()
        sma_26 = price.rolling(26, min_periods=1).mean()
        ema_26 = price.ewm(span=26, adjust=False).mean()
        
        price_to_sma_12 = (price / sma_12 - 1)
        price_to_ema_26 = (price / ema_26 - 1)
        
        ticker_feats[f'{ticker}_price_to_sma_12_norm'] = expanding_zscore(price_to_sma_12)
        ticker_feats[f'{ticker}_price_to_ema_26_norm'] = expanding_zscore(price_to_ema_26)
        
        # MACD
        macd_line = price.ewm(span=12).mean() - price.ewm(span=26).mean()
        signal_line = macd_line.ewm(span=9).mean()
        macd_histogram = (macd_line - signal_line)
        ticker_feats[f'{ticker}_macd_hist_norm'] = expanding_zscore(macd_histogram)
        
        # Bollinger Bands
        bb_sma = price.rolling(20, min_periods=1).mean()
        bb_std = price.rolling(20, min_periods=1).std()
        bb_position = ((price - bb_sma) / (2 * bb_std + 1e-8))
        bb_width = ((4 * bb_std) / (bb_sma + 1e-8))
        
        ticker_feats[f'{ticker}_bb_position_norm'] = expanding_zscore(bb_position)
        ticker_feats[f'{ticker}_bb_width_norm'] = expanding_zscore(bb_width)
        
        # RSI & Stoch
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = (100 - (100 / (1 + rs))) / 100  # Normalize to [0, 1]
        
        low_14 = price.rolling(14).min()
        high_14 = price.rolling(14).max()
        stoch = ((price - low_14) / (high_14 - low_14 + 1e-8))
        
        ticker_feats[f'{ticker}_rsi_14'] = rsi
        ticker_feats[f'{ticker}_stochastic_14'] = stoch
        
        # Drawdown
        rolling_max = price.expanding().max()
        drawdown = ((price - rolling_max) / (rolling_max + 1e-8))
        ticker_feats[f'{ticker}_drawdown_norm'] = expanding_zscore(drawdown)
        
        # Market Beta & Corr
        if qqq_returns is not None:
            rolling_cov = simple_returns.rolling(26).cov(qqq_returns)
            rolling_var = qqq_returns.rolling(26).var()
            beta = (rolling_cov / (rolling_var + 1e-8))
            
            rolling_corr = simple_returns.rolling(26).corr(qqq_returns)
            
            residual = simple_returns - beta * qqq_returns
            idio_vol = residual.rolling(26).std() * np.sqrt(52)
            
            ticker_feats[f'{ticker}_market_beta'] = expanding_zscore(beta)
            ticker_feats[f'{ticker}_market_corr_26w'] = expanding_zscore(rolling_corr)
            ticker_feats[f'{ticker}_idiosync_vol_26w'] = expanding_zscore(idio_vol)
            
        # Volume
        if volume is not None:
            # Dollar volume
            dollar_vol = (price * volume)
            
            # OBV (On-Balance Volume)
            obv = (volume * np.sign(simple_returns)).cumsum()
            
            # OBV rate of change
            obv_roc = (obv / obv.shift(4) - 1)
            
            # Volume ratio
            vol_ma_20 = volume.rolling(20).mean()
            vol_ratio = (volume / (vol_ma_20 + 1e-8))
            
            # Volume rate of change
            vol_roc = (volume / volume.shift(13) - 1)
            
            # Relative volume (vs 4-week average)
            vol_ma_4 = volume.rolling(4).mean()
            rel_vol = (volume / (vol_ma_4 + 1e-8))
            
            ticker_feats[f'{ticker}_dollar_volume_norm'] = expanding_zscore(dollar_vol)
            ticker_feats[f'{ticker}_obv_norm'] = expanding_zscore(obv)
            ticker_feats[f'{ticker}_obv_roc_4w_norm'] = expanding_zscore(obv_roc)
            ticker_feats[f'{ticker}_volume_ratio_norm'] = expanding_zscore(vol_ratio)
            ticker_feats[f'{ticker}_vol_roc_13w_norm'] = expanding_zscore(vol_roc)
            ticker_feats[f'{ticker}_relative_volume_norm'] = expanding_zscore(rel_vol)
            
        features_list.append(ticker_feats)
    
    # ========================================================================
    # CROSS-SECTIONAL FEATURES (NEW!)
    # ========================================================================
    print("\nðŸ”„ Creating cross-sectional features...")
    
    # Returns cross-sectional z-score
    returns_cross_z = returns_df.apply(
        lambda row: (row - row.mean()) / (row.std() + 1e-8),
        axis=1
    ).clip(-3, 3)
    
    # Cross-sectional features collection
    cross_feats_list = []
    
    # Returns cross-sectional z-score
    returns_cross_z = returns_df.apply(
        lambda row: (row - row.mean()) / (row.std() + 1e-8),
        axis=1
    ).clip(-3, 3)
    
    # Filter for valid tickers and rename columns
    valid_ret_tickers = [t for t in tickers if t in returns_cross_z.columns]
    if valid_ret_tickers:
        ret_z_subset = returns_cross_z[valid_ret_tickers].copy()
        ret_z_subset.columns = [f'{t}_return_cross_z' for t in valid_ret_tickers]
        cross_feats_list.append(ret_z_subset)
    
    # Momentum cross-sectional z-scores and ranks
    for window in FEATURE_WINDOWS:
        momentum_df = momentum_dfs[window]
        
        # Cross-sectional z-score
        momentum_cross_z = momentum_df.apply(
            lambda row: (row - row.mean()) / (row.std() + 1e-8),
            axis=1
        ).clip(-3, 3)
        
        # Percentile ranks (0 = worst, 1 = best)
        momentum_ranks = momentum_df.rank(axis=1, pct=True)
        
        valid_mom_tickers = [t for t in tickers if t in momentum_cross_z.columns]
        if valid_mom_tickers:
            # Z-scores
            mom_z_subset = momentum_cross_z[valid_mom_tickers].copy()
            mom_z_subset.columns = [f'{t}_momentum_{window}w_cross_z' for t in valid_mom_tickers]
            cross_feats_list.append(mom_z_subset)
            
            # Ranks
            mom_rank_subset = momentum_ranks[valid_mom_tickers].copy()
            mom_rank_subset.columns = [f'{t}_momentum_{window}w_rank' for t in valid_mom_tickers]
            cross_feats_list.append(mom_rank_subset)
    
    # Volatility cross-sectional z-scores and ranks
    for window in FEATURE_WINDOWS:
        volatility_df = volatility_dfs[window]
        
        # Cross-sectional z-score
        volatility_cross_z = volatility_df.apply(
            lambda row: (row - row.mean()) / (row.std() + 1e-8),
            axis=1
        ).clip(-3, 3)
        
        # Percentile ranks (0 = lowest vol, 1 = highest vol)
        volatility_ranks = volatility_df.rank(axis=1, pct=True)
        
        valid_vol_tickers = [t for t in tickers if t in volatility_cross_z.columns]
        if valid_vol_tickers:
            # Z-scores
            vol_z_subset = volatility_cross_z[valid_vol_tickers].copy()
            vol_z_subset.columns = [f'{t}_volatility_{window}w_cross_z' for t in valid_vol_tickers]
            cross_feats_list.append(vol_z_subset)
            
            # Ranks
            vol_rank_subset = volatility_ranks[valid_vol_tickers].copy()
            vol_rank_subset.columns = [f'{t}_volatility_{window}w_rank' for t in valid_vol_tickers]
            cross_feats_list.append(vol_rank_subset)
            
    if cross_feats_list:
        cross_feats = pd.concat(cross_feats_list, axis=1)
        features_list.append(cross_feats)
    
    # ========================================================================
    # CONCATENATE ALL FEATURES
    # ========================================================================
    print(f"ðŸ§© Concatenating {len(features_list)} feature blocks...")
    features = pd.concat(features_list, axis=1)
    
    # ========================================================================
    # CALENDAR FEATURES
    # ========================================================================
    features['month_sin'] = np.sin(2 * np.pi * features.index.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * features.index.month / 12)
    
    # ========================================================================
    # CLEANUP & VALIDATION
    # ========================================================================
    print(f"\nðŸ§¹ Cleaning features...")
    print(f"   Raw features: {features.shape[1]} features, {features.shape[0]} samples")
    
    # Remove burn-in period
    features = features.iloc[MIN_HISTORY:]
    print(f"   After burn-in: {features.shape[1]} features, {features.shape[0]} samples")
    
    # Clip and drop NaN
    features = features.clip(lower=-3, upper=3).dropna()
    print(f"   After cleanup: {features.shape[1]} features, {features.shape[0]} samples")
    
    # Apply lean mode if requested
    if lean:
        print(f"\nðŸŽ¯ Applying lean feature selection...")
        features = get_lean_features(features)
        print(f"   Final features: {features.shape[1]} features, {features.shape[0]} samples")
    
    # Validate
    features = validate_features(features)
    
    # Summary statistics
    print(f"\nðŸ“Š Feature Summary:")
    print(f"   Total features: {features.shape[1]}")
    print(f"   Total samples: {features.shape[0]}")
    print(f"   Feature-to-sample ratio: {features.shape[1] / features.shape[0]:.3f}")
    
    return features

# ============================================================================
# FEATURE SELECTION & VALIDATION
# ============================================================================

def get_lean_features(all_features):
    """
    Filter features based on MLP importance analysis
    Automatically detects if using Expanded or Simple feature set
    """
    suffix = "_expanded"
    feature_importance_path = os.path.join(OUTPUTS_DIR, f'feature_importance{suffix}.csv')
    
    if not os.path.exists(feature_importance_path):
        print(f"   â„¹ï¸  Feature importance file not found at: {feature_importance_path}")
        print("      (This is expected if you haven't run analyze_features.py yet)")
        print("      Using ALL features (Lean mode disabled).")
        return all_features
    
    importance_df = pd.read_csv(feature_importance_path)
    
    # Keep features with positive importance
    keep_features = importance_df[importance_df['Importance'] > 0]['Feature'].tolist()
    
    # Verify that these features actually exist in our current dataset
    available_features = set(all_features.columns)
    valid_keep_features = [f for f in keep_features if f in available_features]
    
    if len(valid_keep_features) < 5:
        print("      âš ï¸  Warning: Mismatch between importance file and current features.")
        print(f"      Found {len(valid_keep_features)} valid features out of {len(keep_features)} suggested.")
        print("      Using ALL features instead to avoid errors.")
        return all_features
    
    if len(valid_keep_features) < 10:
        print("      âš ï¸  Too few features selected. Keeping top 20 instead.")
        keep_features = importance_df.head(20)['Feature'].tolist()
        valid_keep_features = [f for f in keep_features if f in available_features]
    
    # Preserve return columns (needed by environment)
    ret_cols = [c for c in all_features.columns if '_simple_ret' in c or '_log_ret' in c]
    
    # Combine selected features with return columns
    final_features = list(set(valid_keep_features + ret_cols))
    
    print(f"      Selected {len(final_features)} features (including {len(ret_cols)} return columns)")
    
    return all_features[final_features]

def validate_features(features_df):
    """Check feature quality before training"""
    # Check for inf values
    inf_cols = features_df.columns[np.isinf(features_df).any()]
    if len(inf_cols) > 0:
        raise ValueError(f"âŒ Inf values in: {inf_cols.tolist()}")
    
    # Check for excessive NaN
    nan_pct = features_df.isnull().mean()
    if (nan_pct > 0.1).any():
        print(f"   âš ï¸  Warning: >10% NaN in features:")
        print(nan_pct[nan_pct > 0.1])
    
    # Check for zero variance
    zero_var = features_df.std() == 0
    if zero_var.any():
        print(f"   âš ï¸  Warning: Zero variance features: {zero_var[zero_var].index.tolist()}")
    
    return features_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data engineering with cross-sectional features')
    parser.add_argument('--no-lean', action='store_true', help='Disable lean feature selection')
    parser.add_argument('--no-save', action='store_true', help='Do not save split files')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DATA ENGINEERING PIPELINE - CROSS-SECTIONAL VERSION")
    print("="*70)
    
    # Prepare train/test split
    train_prices, train_volumes, test_prices, test_volumes, train_qqq, test_qqq = \
        prepare_train_test_split_fixed(save_files=not args.no_save)
    
    # Create features
    train_features = create_features(
        train_prices, 
        train_volumes, 
        train_qqq, 
        lean=not args.no_lean
    )
    
    print("\n" + "="*70)
    print("âœ… PIPELINE COMPLETE")
    print("="*70)
    print(f"Train features shape: {train_features.shape}")
    print(f"Date range: {train_features.index[0]} to {train_features.index[-1]}")
    
    # Optionally save features
    if not args.no_save:
        feature_path = os.path.join(OUTPUTS_DIR, 'train_features_cross_sectional.csv')
        train_features.to_csv(feature_path)
        print(f"\nðŸ’¾ Features saved to: {feature_path}")