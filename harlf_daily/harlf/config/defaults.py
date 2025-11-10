"""
Configuration file for RL Portfolio Optimization Data Preparation Pipeline
Contains all constants, parameters, and configuration settings
"""

import os
import json
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent  # harlf_daily directory
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
WALK_FORWARD_DIR = DATA_DIR / "walk_forward"
METADATA_DIR = DATA_DIR / "metadata"
CONFIG_DIR = PROJECT_ROOT / "config"

# ============================================================================
# API KEYS
# ============================================================================
API_KEYS_FILE = CONFIG_DIR / "api_keys.json"

def load_api_keys():
    """Load API keys from config file (only needed for data collection)"""
    if API_KEYS_FILE.exists():
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    else:
        return {}  # Return empty dict if keys not found (not needed for evaluation)

# Load API keys lazily - only fail if actually trying to use them
API_KEYS = load_api_keys()
FRED_API_KEY = API_KEYS.get('fred', None)

# ============================================================================
# DATA COLLECTION PARAMETERS
# ============================================================================

# Stock tickers (7 tickers)
TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']

# Benchmark
BENCHMARK = 'QQQ'

# Date range
START_DATE = '2015-01-01'
END_DATE = '2025-11-09'

# Missing data handling
MAX_FORWARD_FILL_DAYS = 5  # Maximum days to forward fill missing data

# ============================================================================
# FRED MACRO INDICATORS
# ============================================================================

FRED_INDICATORS = {
    # Interest Rates
    'DGS10': 'treasury_10y',          # 10-Year Treasury Yield
    'DGS2': 'treasury_2y',            # 2-Year Treasury Yield
    'FEDFUNDS': 'fed_funds_rate',     # Federal Funds Rate

    # Economic Indicators
    'CPIAUCSL': 'cpi',                 # Consumer Price Index
}

# Macro data resampling
MAX_MACRO_FORWARD_FILL_DAYS = 30  # Avoid stale macro data

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# Technical indicators windows
MOMENTUM_WINDOWS = {
    'short': 5,   # 5-day momentum
    'medium': 21, # 21-day momentum (1 month)
}

TREND_WINDOWS = {
    'sma': 20,    # Simple Moving Average
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
}

MEAN_REVERSION_WINDOWS = {
    'rsi': 14,    # RSI period
    'bb': 20,     # Bollinger Bands period
    'bb_std': 2,  # Bollinger Bands standard deviations
}

VOLATILITY_WINDOWS = {
    'short': 21,  # Short-term volatility
    'long': 63,   # Long-term volatility (3 months)
    'atr': 14,    # Average True Range
}

VOLUME_WINDOWS = {
    'ratio': 20,  # Volume ratio period
    'mfi': 14,    # Money Flow Index
}

CROSS_ASSET_WINDOWS = {
    'correlation': 60,  # Rolling correlation with benchmark
    'beta': 60,         # Rolling beta
    'relative_strength': 21,  # Relative strength period
}

MACRO_WINDOWS = {
    'vix_change': 21,  # VIX change period
}

# Regime detection
REGIME_CLUSTERS = 3  # Number of market regimes
REGIME_TRANSITION_WINDOW = 5  # Days to check for regime change

# ============================================================================
# FEATURE NAMES (Total: 23 features)
# ============================================================================

# Core technical features (8)
TECHNICAL_FEATURES = [
    'return_5d',           # Momentum
    'return_21d',          # Momentum
    'price_to_sma_20d',    # Trend
    'macd_histogram',      # Trend
    'rsi_14d',             # Mean reversion
    'bb_position_20d',     # Mean reversion
    'volatility_21d',      # Volatility
    'atr_pct_14d',         # Volatility
]

# Volume features (2)
VOLUME_FEATURES = [
    'volume_ratio_20d',
    'mfi_14d',
]

# Cross-asset features (3)
CROSS_ASSET_FEATURES = [
    'bench_correlation_60d',
    'bench_beta_60d',
    'bench_relative_strength',
]

# Macro features (5)
MACRO_FEATURES = [
    'treasury_10y',
    'yield_curve_slope',
    'vix',
    'vix_change_21d',
]

# Feature interactions (3)
INTERACTION_FEATURES = [
    'momentum_volatility_ratio',
    'volume_weighted_rsi',
    'vol_regime_indicator',
]

# Regime features (2)
REGIME_FEATURES = [
    'regime_cluster',
    'regime_transition',
]

# All features (for agent observation - 23 features)
ALL_FEATURES = (
    TECHNICAL_FEATURES +
    VOLUME_FEATURES +
    CROSS_ASSET_FEATURES +
    MACRO_FEATURES +
    INTERACTION_FEATURES +
    REGIME_FEATURES
)

# Target columns (NOT for agent input - used by environment for returns)
TARGET_COLUMNS = [
    'forward_log_return',    # 1-day forward log return (for environment) - PRIMARY TARGET
    'forward_return',        # 1-day forward simple return (for reference/debugging) - will be removed by VIF
]

# ❌ REMOVED: close_price (non-stationary, not useful for RL rewards)
# Note: TARGET_COLUMNS are included in data files but excluded from agent observations and normalization

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# Outlier handling (Winsorization)
WINSORIZE_LIMITS = (0.01, 0.01)  # 1st and 99th percentiles

# Feature selection
MAX_CORRELATION = 0.90  # Maximum correlation between features
MAX_VIF = 20           # Maximum Variance Inflation Factor - Relaxed threshold for RL state space
MIN_FEATURE_IMPORTANCE = 0.01  # Minimum feature importance (1%)

# Target final features
TARGET_FINAL_FEATURES = 15

# ============================================================================
# WALK-FORWARD VALIDATION PARAMETERS
# ============================================================================

# Window sizes (in trading days)
INITIAL_TRAIN_SIZE = 756  # ~3 years (252 trading days/year)
VAL_SIZE = 126            # ~6 months
TEST_SIZE = 21            # ~1 month

# Number of folds
N_SPLITS = 10

# Random seed for reproducibility
RANDOM_STATE = 42

# ============================================================================
# DATA QUALITY PARAMETERS
# ============================================================================

# Stationarity test
ADF_SIGNIFICANCE_LEVEL = 0.05  # p-value threshold for ADF test

# NaN handling
DROP_NAN_THRESHOLD = 0.05  # Drop features with >5% NaN values

# ============================================================================
# OUTPUT PARAMETERS
# ============================================================================

# File formats
RAW_STOCK_FILE = RAW_DATA_DIR / "stocks_ohlcv.csv"
RAW_BENCHMARK_FILE = RAW_DATA_DIR / "benchmark_qqq.csv"
RAW_MACRO_FILE = RAW_DATA_DIR / "macro_fred.csv"
RAW_VIX_FILE = RAW_DATA_DIR / "vix.csv"

PROCESSED_FEATURES_FILE = PROCESSED_DATA_DIR / "features_full.csv"

METADATA_FILES = {
    'feature_definitions': METADATA_DIR / "feature_definitions.json",
    'fold_indices': METADATA_DIR / "fold_indices.csv",
    'preprocessing_params': METADATA_DIR / "preprocessing_params.json",
    'asset_first_valid': METADATA_DIR / "asset_first_valid.csv",
    'validation_report': METADATA_DIR / "data_preparation_report.md",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_fold_dir(fold_id):
    """Get directory path for a specific fold"""
    fold_dir = WALK_FORWARD_DIR / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir

def get_fold_files(fold_id):
    """
    Get file paths for train/val/test data for a specific fold
    Returns separate paths for features and targets to prevent normalization issues
    """
    fold_dir = get_fold_dir(fold_id)
    return {
        'train_features': fold_dir / "train_features.csv",
        'val_features': fold_dir / "val_features.csv",
        'test_features': fold_dir / "test_features.csv",
        'train_targets': fold_dir / "train_targets.csv",
        'val_targets': fold_dir / "val_targets.csv",
        'test_targets': fold_dir / "test_targets.csv",
        # Legacy paths (for backward compatibility)
        'train': fold_dir / "train.csv",
        'val': fold_dir / "val.csv",
        'test': fold_dir / "test.csv",
    }

# ============================================================================
# YFINANCE PARAMETERS
# ============================================================================

# Download parameters
YFINANCE_AUTO_ADJUST = False  # Keep raw OHLC data
YFINANCE_ACTIONS = True       # Include dividends and splits
YFINANCE_PROGRESS = True      # Show download progress

# ============================================================================
# LOGGING
# ============================================================================

VERBOSE = True  # Print progress messages

def log(message):
    """Simple logging function"""
    if VERBOSE:
        print(f"[INFO] {message}")
