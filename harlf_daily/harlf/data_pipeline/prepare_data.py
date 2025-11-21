import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import pickle
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================================================
# CONFIGURATION
# ============================================================================

from harlf.config.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    WALK_FORWARD_DIR,
    METADATA_DIR,
    RAW_STOCK_FILE,
    RAW_BENCHMARK_FILE,
    RAW_MACRO_FILE,
    RAW_VIX_FILE,
    PROCESSED_FEATURES_FILE,
    METADATA_FILES,
)

# --- Use selected config from defaults.py (lines 42-50) ---
import harlf.config.defaults as defaults

class Config:
    """Configuration parameters"""

    # Load basic config values from defaults.py (lines 42-50)
    START_DATE = defaults.START_DATE
    END_DATE = defaults.END_DATE
    TICKERS = defaults.TICKERS
    BENCHMARK = defaults.BENCHMARK

    N_SPLITS = defaults.N_SPLITS
    INITIAL_TRAIN_SIZE = defaults.INITIAL_TRAIN_SIZE
    VAL_SIZE = defaults.VAL_SIZE
    TEST_SIZE = defaults.TEST_SIZE

    # Paths (delegated to harlf/config/paths.py)
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = DATA_DIR
    RAW_DATA_DIR = RAW_DATA_DIR
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
    WALK_FORWARD_DIR = WALK_FORWARD_DIR
    METADATA_DIR = METADATA_DIR

    RAW_STOCK_FILE = RAW_STOCK_FILE
    RAW_BENCHMARK_FILE = RAW_BENCHMARK_FILE
    RAW_MACRO_FILE = RAW_MACRO_FILE
    RAW_VIX_FILE = RAW_VIX_FILE
    PROCESSED_FEATURES_FILE = PROCESSED_FEATURES_FILE
    METADATA_FILES = METADATA_FILES

    # Feature parameters (kept local for reproducibility here)
    MOMENTUM_WINDOWS = {'short': 5, 'medium': 21}
    TREND_WINDOWS = {'sma': 20, 'macd_slow': 26, 'macd_fast': 12, 'macd_signal': 9}
    MEAN_REVERSION_WINDOWS = {'rsi': 14, 'bb': 20}
    VOLATILITY_WINDOWS = {'rolling': 21, 'atr': 14}
    VOLUME_WINDOWS = {'mfi': 14}

    # Preprocessing
    WINSORIZE_LIMITS = (0.05, 0.05)
    MAX_RETURN_CAP = 0.30
    TARGET_COLUMNS = ['forward_log_return', 'forward_return']

    YFINANCE_AUTO_ADJUST = True
    MAX_FORWARD_FILL_DAYS = 5

    @classmethod
    def log(cls, message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    @classmethod
    def get_fold_files(cls, fold_id):
        fold_dir = cls.WALK_FORWARD_DIR / f'fold_{fold_id}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        return {
            'train': fold_dir / 'train.csv',
            'val': fold_dir / 'val.csv',
            'test': fold_dir / 'test.csv',
            'train_features': fold_dir / 'train_features.csv',
            'train_targets': fold_dir / 'train_targets.csv',
            'val_features': fold_dir / 'val_features.csv',
            'val_targets': fold_dir / 'val_targets.csv',
            'test_features': fold_dir / 'test_features.csv',
            'test_targets': fold_dir / 'test_targets.csv',
        }

# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_and_save_data():
    """Download and save all raw data"""
    Config.log("="*80)
    Config.log("PHASE 1: DATA DOWNLOAD")
    Config.log("="*80)

    for directory in [Config.RAW_DATA_DIR, Config.PROCESSED_DATA_DIR,
                      Config.WALK_FORWARD_DIR, Config.METADATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Download stock data
    Config.log(f"Downloading stock data for {len(Config.TICKERS)} tickers...")
    df = yf.download(Config.TICKERS, start=Config.START_DATE, end=Config.END_DATE,
                     auto_adjust=Config.YFINANCE_AUTO_ADJUST, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.swaplevel()

    Config.log(f"Downloaded {len(df)} trading days")

    # Process and save
    data = {}
    for price_type in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if price_type in df.columns.get_level_values(1):
            data[price_type.lower()] = df.xs(price_type, level=1, axis=1)

    for key in data:
        data[key] = data[key].ffill(limit=Config.MAX_FORWARD_FILL_DAYS)

    data['returns'] = data['close'].pct_change()

    # Save stock data
    stock_df = pd.DataFrame()
    for ticker in Config.TICKERS:
        if ticker in data['close'].columns:
            for price_type in ['open', 'high', 'low', 'close', 'volume', 'returns']:
                stock_df[f"{ticker}_{price_type}"] = data[price_type][ticker]

    stock_df.to_csv(Config.RAW_STOCK_FILE)
    Config.log(f"Saved stock data: {Config.RAW_STOCK_FILE}")

    # Download benchmark
    Config.log(f"Downloading benchmark: {Config.BENCHMARK}...")
    bench = yf.download(Config.BENCHMARK, start=Config.START_DATE, end=Config.END_DATE,
                       auto_adjust=Config.YFINANCE_AUTO_ADJUST, progress=False)

    if isinstance(bench.columns, pd.MultiIndex):
        bench.columns = bench.columns.get_level_values(0)

    bench_df = pd.DataFrame({
        'close': bench['Close'],
        'returns': bench['Close'].pct_change()
    })
    bench_df.to_csv(Config.RAW_BENCHMARK_FILE)
    Config.log(f"Saved benchmark data: {Config.RAW_BENCHMARK_FILE}")

    return stock_df.index

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_all_features(close, high, low, volume, returns, benchmark_returns):
    """Calculate all features for one ticker"""
    features = pd.DataFrame(index=close.index)

    # Momentum
    features['return_5d'] = close.pct_change(periods=5)
    features['return_21d'] = close.pct_change(periods=21)

    # Trend
    sma_20 = close.rolling(window=20).mean()
    features['price_to_sma_20d'] = close / sma_20

    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    features['macd_histogram'] = macd.macd() - macd.macd_signal()

    # Mean reversion
    rsi = RSIIndicator(close=close, window=14)
    features['rsi_14d'] = rsi.rsi()

    bb = BollingerBands(close=close, window=20)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    features['bb_position'] = (close - bb_low) / (bb_high - bb_low)

    # Volatility
    features['volatility_21d'] = returns.rolling(window=21).std()

    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    features['atr_14d'] = atr.average_true_range()

    # Volume
    volume_sma = volume.rolling(window=20).mean()
    features['volume_ratio'] = volume / volume_sma

    mfi = MFIIndicator(high=high, low=low, close=close, volume=volume, window=14)
    features['mfi_14d'] = mfi.money_flow_index()

    # Cross-asset
    rolling_corr = returns.rolling(window=63).corr(benchmark_returns)
    features['correlation_to_qqq'] = rolling_corr

    rolling_cov = returns.rolling(window=63).cov(benchmark_returns)
    benchmark_var = benchmark_returns.rolling(window=63).var()
    features['beta_to_qqq'] = rolling_cov / benchmark_var

    # Interaction
    features['momentum_vol_interaction'] = features['return_5d'] * features['volatility_21d']

    # ✅ REMOVED: Regime clustering moved to per-fold calculation to prevent data leakage

    return features


def add_targets_with_cap(features_df, close, ticker):
    """Add forward returns with capping"""
    forward_log_return = np.log(close.shift(-1) / close)
    forward_return = close.pct_change().shift(-1)

    # Cap extreme returns
    n_capped = (forward_log_return.abs() > Config.MAX_RETURN_CAP).sum()
    if n_capped > 0:
        Config.log(f"    {ticker}: Capped {n_capped} extreme returns")

    features_df['forward_log_return'] = forward_log_return.clip(
        -Config.MAX_RETURN_CAP, Config.MAX_RETURN_CAP
    )
    features_df['forward_return'] = forward_return.clip(
        -Config.MAX_RETURN_CAP, Config.MAX_RETURN_CAP
    )

    return features_df


def engineer_all_features():
    """Engineer features for all tickers"""
    Config.log("="*80)
    Config.log("PHASE 2: FEATURE ENGINEERING")
    Config.log("="*80)

    # Load raw data
    stock_df = pd.read_csv(Config.RAW_STOCK_FILE, index_col=0, parse_dates=True)
    bench_df = pd.read_csv(Config.RAW_BENCHMARK_FILE, index_col=0, parse_dates=True)

    all_features = []

    for ticker in Config.TICKERS:
        Config.log(f"Engineering features for {ticker}...")

        # Extract ticker data
        close = stock_df[f'{ticker}_close']
        high = stock_df[f'{ticker}_high']
        low = stock_df[f'{ticker}_low']
        volume = stock_df[f'{ticker}_volume']
        returns = stock_df[f'{ticker}_returns']
        benchmark_returns = bench_df['returns']

        # Calculate features
        features = calculate_all_features(close, high, low, volume, returns, benchmark_returns)

        # ✅ FIXED: Removed unnecessary shift(1) - use most recent data available at time T
        # All features are calculated from data up to time T (close of day)
        # Target is return from T to T+1

        # Add targets WITHOUT lagging
        features = add_targets_with_cap(features, close, ticker)

        # Drop NaN
        features = features.dropna()

        # Add ticker
        features['ticker'] = ticker

        Config.log(f"  Final shape: {features.shape}")

        all_features.append(features)

    # Combine all tickers
    features_df = pd.concat(all_features, axis=0)
    features_df = features_df.sort_index()

    # Save
    features_df.to_csv(Config.PROCESSED_FEATURES_FILE)

    Config.log(f"\nFeature engineering complete:")
    Config.log(f"  Total rows: {len(features_df)}")
    Config.log(f"  Features: {len(features_df.columns) - 3}")  # -3 for ticker and targets
    Config.log(f"  Tickers: {features_df['ticker'].nunique()}")

    return features_df

# ============================================================================
# REGIME CLUSTERING (PER-FOLD, NO LEAKAGE)
# ============================================================================

def calculate_regime_feature(train_df, val_df, test_df):
    """
    Calculate regime clustering WITHOUT data leakage.
    Fit KMeans on training data only, then predict on val/test.
    Returns regime as one-hot encoded features (not normalized).
    """
    cluster_features = ['volatility_21d', 'return_21d', 'correlation_to_qqq']

    # Extract cluster features
    train_cluster = train_df[cluster_features].copy().ffill().bfill().fillna(0)
    val_cluster = val_df[cluster_features].copy().ffill().bfill().fillna(0)
    test_cluster = test_df[cluster_features].copy().ffill().bfill().fillna(0)

    # Fit scaler and KMeans on TRAINING data only
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    train_regime = kmeans.fit_predict(train_scaled)

    # Apply to val and test (no fitting!)
    val_scaled = scaler.transform(val_cluster)
    test_scaled = scaler.transform(test_cluster)

    val_regime = kmeans.predict(val_scaled)
    test_regime = kmeans.predict(test_scaled)

    # One-hot encode (prevents normalization issues)
    def one_hot_regime(regime_labels, index):
        regime_df = pd.DataFrame(index=index)
        for i in range(3):
            regime_df[f'regime_{i}'] = (regime_labels == i).astype(int)
        return regime_df

    train_regime_df = one_hot_regime(train_regime, train_df.index)
    val_regime_df = one_hot_regime(val_regime, val_df.index)
    test_regime_df = one_hot_regime(test_regime, test_df.index)

    return train_regime_df, val_regime_df, test_regime_df


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def walk_forward_split(data):
    """Create walk-forward folds using POSITIONAL indices to avoid duplication"""
    Config.log("="*80)
    Config.log("PHASE 3: WALK-FORWARD VALIDATION")
    Config.log("="*80)

    Config.log(f"Total data points: {len(data)}")

    # ✅ FIXED: Use integer positions to avoid duplicate index issues
    # Reset index to get clean positional slicing
    data_reset = data.reset_index(drop=False)  # Keep Date as a column

    folds = []
    start_idx = 0

    for i in range(Config.N_SPLITS):
        train_end = start_idx + Config.INITIAL_TRAIN_SIZE
        val_end = train_end + Config.VAL_SIZE
        test_end = val_end + Config.TEST_SIZE

        if test_end > len(data_reset):
            break

        fold = {
            'fold_id': i,
            'train': list(range(start_idx, train_end)),  # Integer positions
            'val': list(range(train_end, val_end)),
            'test': list(range(val_end, test_end)),
        }

        folds.append(fold)
        start_idx += Config.TEST_SIZE

    Config.log(f"Created {len(folds)} folds")
    return folds, data_reset

# ============================================================================
# PREPROCESSING WITH GLOBAL NORMALIZATION
# ============================================================================

def create_global_scaler(features_df, fold_0_train_indices):
    """Create global scaler from fold 0 training data"""
    Config.log("="*80)
    Config.log("CREATING GLOBAL SCALER (fold 0 training data)")
    Config.log("="*80)

    # ✅ FIXED: Use iloc for positional indexing
    train_0 = features_df.iloc[fold_0_train_indices].copy()

    # Identify columns (Note: 'regime' removed, will be added per-fold)
    # Exclude: targets, ticker, and Date column
    target_cols = ['forward_log_return', 'forward_return', 'ticker']
    exclude_cols = target_cols + ['Date']
    feature_cols = [c for c in train_0.columns if c not in exclude_cols]

    Config.log(f"  Features to normalize: {len(feature_cols)}")
    Config.log(f"  Targets (raw): {[c for c in target_cols if c in train_0.columns]}")

    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(train_0[feature_cols])

    Config.log(f"  Scaler fitted on {len(train_0)} samples")
    Config.log(f"  Mean range: [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]")
    Config.log(f"  Std range: [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]")

    return scaler, feature_cols


def preprocess_fold(train_df, val_df, test_df, global_scaler, feature_cols):
    """
    Preprocess fold with global scaler.
    Regime features are added AFTER this function (not normalized).
    """
    # Handle NaN
    train_df = train_df.ffill().bfill().fillna(0)
    val_df = val_df.ffill().bfill().fillna(0)
    test_df = test_df.ffill().bfill().fillna(0)

    # Separate features (no regime yet)
    train_feat = train_df[feature_cols]
    val_feat = val_df[feature_cols]
    test_feat = test_df[feature_cols]

    # Winsorize
    lower = train_feat.quantile(Config.WINSORIZE_LIMITS[0])
    upper = train_feat.quantile(1 - Config.WINSORIZE_LIMITS[1])

    train_feat = train_feat.clip(lower=lower, upper=upper, axis=1)
    val_feat = val_feat.clip(lower=lower, upper=upper, axis=1)
    test_feat = test_feat.clip(lower=lower, upper=upper, axis=1)

    # Apply global scaler
    train_scaled = pd.DataFrame(
        global_scaler.transform(train_feat),
        index=train_feat.index,
        columns=feature_cols
    )
    val_scaled = pd.DataFrame(
        global_scaler.transform(val_feat),
        index=val_feat.index,
        columns=feature_cols
    )
    test_scaled = pd.DataFrame(
        global_scaler.transform(test_feat),
        index=test_feat.index,
        columns=feature_cols
    )

    # ✅ FIXED: Calculate regime features per-fold (no leakage, not normalized)
    train_regime_df, val_regime_df, test_regime_df = calculate_regime_feature(
        train_df, val_df, test_df
    )

    # Add regime features (one-hot encoded, not normalized)
    train_scaled = pd.concat([train_scaled, train_regime_df], axis=1)
    val_scaled = pd.concat([val_scaled, val_regime_df], axis=1)
    test_scaled = pd.concat([test_scaled, test_regime_df], axis=1)

    # Add back targets
    for col in ['forward_log_return', 'forward_return', 'ticker']:
        if col in train_df.columns:
            train_scaled[col] = train_df[col].values  # Use .values to avoid index issues
            val_scaled[col] = val_df[col].values
            test_scaled[col] = test_df[col].values

    return train_scaled, val_scaled, test_scaled

# ============================================================================
# SAVE FOLD DATA
# ============================================================================

def save_fold_data(fold_id, train_df, val_df, test_df):
    """Save fold data split into features and targets"""
    fold_files = Config.get_fold_files(fold_id)

    target_cols = [c for c in Config.TARGET_COLUMNS if c in train_df.columns]
    meta_cols = ['ticker'] if 'ticker' in train_df.columns else []
    preserve_cols = target_cols + meta_cols

    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if len(target_cols) > 0:
            targets = df[preserve_cols].copy()
            features = df.drop(columns=target_cols).copy()

            targets.to_csv(fold_files[f'{split_name}_targets'])
            features.to_csv(fold_files[f'{split_name}_features'])

        df.to_csv(fold_files[split_name])

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_all_folds(features_df, max_folds=None):
    """Process all folds with global normalization"""
    Config.log("="*80)
    Config.log("PHASE 4: PROCESSING FOLDS WITH GLOBAL SCALER")
    Config.log("="*80)

    # ✅ FIXED: Create folds with positional indices
    folds, data_reset = walk_forward_split(features_df)

    if max_folds:
        folds = folds[:max_folds]
        Config.log(f"Limited to {max_folds} folds")

    # Create global scaler (using positional indices)
    global_scaler, feature_cols = create_global_scaler(data_reset, folds[0]['train'])

    # Save scaler
    scaler_path = Config.WALK_FORWARD_DIR / 'global_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(global_scaler, f)
    Config.log(f"Saved: {scaler_path}")

    # Save fold indices
    fold_df = pd.DataFrame([
        {
            'fold_id': f['fold_id'],
            'train_size': len(f['train']),
            'val_size': len(f['val']),
            'test_size': len(f['test']),
        }
        for f in folds
    ])
    fold_df.to_csv(Config.METADATA_DIR / 'fold_indices.csv', index=False)

    # Save preprocessing params (for verification script)
    params = {f'fold_{i}': {'global_scaler_path': str(scaler_path)} for i in range(len(folds))}
    import json
    with open(Config.METADATA_DIR / 'preprocessing_params.json', 'w') as f:
        json.dump(params, f)

    # Process folds
    Config.log(f"\nProcessing {len(folds)} folds...")

    for fold in folds:
        fold_id = fold['fold_id']

        if fold_id % 10 == 0 or fold_id < 5:
            Config.log(f"  Fold {fold_id}/{len(folds)}")

        # ✅ FIXED: Use iloc for positional indexing (no duplication!)
        train = data_reset.iloc[fold['train']].copy()
        val = data_reset.iloc[fold['val']].copy()
        test = data_reset.iloc[fold['test']].copy()

        # Reset index to original datetime index
        train = train.set_index('Date')
        val = val.set_index('Date')
        test = test.set_index('Date')

        train_p, val_p, test_p = preprocess_fold(train, val, test, global_scaler, feature_cols)

        save_fold_data(fold_id, train_p, val_p, test_p)

    Config.log(f"\n✅ Processed {len(folds)} folds with global scaler")

    return folds, global_scaler, feature_cols


def run_full_pipeline(max_folds=None, skip_download=False):
    """Run complete pipeline"""
    import time
    start = time.time()

    Config.log("="*80)
    Config.log(" HARLF FIXED DATA PIPELINE")
    Config.log("="*80)

    # Download
    if not skip_download or not Config.RAW_STOCK_FILE.exists():
        download_and_save_data()
    else:
        Config.log("\nUsing existing raw data")

    # Engineer features
    features_df = engineer_all_features()

    # Process folds
    folds, scaler, feature_cols = process_all_folds(features_df, max_folds)

    # Summary
    elapsed = time.time() - start

    Config.log("\n" + "="*80)
    Config.log("PIPELINE COMPLETE!")
    Config.log("="*80)
    Config.log(f"Time: {elapsed/60:.2f} minutes")
    Config.log(f"Folds: {len(folds)}")
    Config.log(f"Features: {len(feature_cols)} (base) + 3 (regime one-hot)")

    Config.log("\n✅ All critical fixes applied:")
    Config.log("  ✅ Global normalization (no data leakage)")
    Config.log("  ✅ Regime clustering per-fold (no data leakage)")
    Config.log("  ✅ Regime one-hot encoded (not normalized)")
    Config.log("  ✅ Removed unnecessary feature lag (use T to predict T→T+1)")
    Config.log("  ✅ Fixed 7x duplication bug (positional indexing)")
    Config.log("  ✅ Return capping (±30%)")
    Config.log("  ✅ Features/targets separated")

    Config.log("\nNext: python verify_pipeline_2.py")

    return {'features': features_df, 'folds': folds, 'scaler': scaler}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-folds', type=int, default=None)
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--quick-test', action='store_true')

    args = parser.parse_args()

    if args.quick_test:
        Config.log("QUICK TEST MODE (5 folds)")
        run_full_pipeline(max_folds=5, skip_download=True)
    else:
        run_full_pipeline(max_folds=args.max_folds, skip_download=args.skip_download)
