"""
Generate test_weekly.parquet with 22 training features.

This script creates the test data file needed for backtesting the trained models.
Models expect 22 features: 17 ticker-specific + 5 macro features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path('data/processed')
OUTPUT_FILE = DATA_DIR / 'test_weekly.parquet'

# Check if source data exists
if not (DATA_DIR / 'test.parquet').exists():
    print("❌ Source file 'test.parquet' not found")
    print("   Run 01_data_preparation.ipynb first to generate raw data")
    exit(1)

print("=" * 80)
print("GENERATING test_weekly.parquet")
print("=" * 80)

# Load raw test data
print("\n📂 Loading test.parquet...")
test_df = pd.read_parquet(DATA_DIR / 'test.parquet')
print(f"   Loaded: {len(test_df)} rows, {len(test_df.columns)} columns")

# Define the 22 required features (before normalization)
REQUIRED_BASE_FEATURES = [
    # Ticker features (17)
    'momentum_1w', 'momentum_4w', 'momentum_13w', 'returns_52w',
    'volatility_4w', 'volatility_52w',
    'price_to_sma_12w', 'price_to_ema_26w',
    'macd_histogram', 'bb_position', 'stochastic_14w',
    'relative_volume_20w', 'obv_roc_13w',
    'sharpe_26w', 'sortino_26w', 'calmar_52w', 'max_drawdown_4w',
    # Macro features (5)
    'vix', 'treasury_10y', 'yield_curve_10y2y', 'dxy', 'oil_wti'
]

# Check which features are available
missing_features = [f for f in REQUIRED_BASE_FEATURES if f not in test_df.columns]

if missing_features:
    print(f"\n❌ Missing {len(missing_features)} required features:")
    for f in missing_features:
        print(f"   - {f}")
    print("\n🔧 Need to regenerate data with all required features")
    print("   Run 01_data_preparation.ipynb with macro features enabled")
    exit(1)

print(f"\n✓ All {len(REQUIRED_BASE_FEATURES)} required features found")

# Select required columns
KEEP_COLS = ['date', 'ticker'] + REQUIRED_BASE_FEATURES
test_subset = test_df[KEEP_COLS].copy()
test_subset['date'] = pd.to_datetime(test_subset['date'])

print(f"\n📊 Applying per-ticker z-score normalization (52-week window)...")

# Separate ticker and macro features
TICKER_FEATURES = [
    'momentum_1w', 'momentum_4w', 'momentum_13w', 'returns_52w',
    'volatility_4w', 'volatility_52w',
    'price_to_sma_12w', 'price_to_ema_26w',
    'macd_histogram', 'bb_position', 'stochastic_14w',
    'relative_volume_20w', 'obv_roc_13w',
    'sharpe_26w', 'sortino_26w', 'calmar_52w', 'max_drawdown_4w'
]

MACRO_FEATURES = ['vix', 'treasury_10y', 'yield_curve_10y2y', 'dxy', 'oil_wti']

# Normalize ticker features (per-ticker z-score with 52-week rolling window)
test_normalized = []

for ticker in test_subset['ticker'].unique():
    ticker_data = test_subset[test_subset['ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('date')

    for feature in TICKER_FEATURES:
        # Rolling z-score normalization
        rolling_mean = ticker_data[feature].rolling(window=52, min_periods=1).mean()
        rolling_std = ticker_data[feature].rolling(window=52, min_periods=1).std()
        rolling_std = rolling_std.replace(0, 1e-8)  # Avoid division by zero

        ticker_data[f'{feature}_norm'] = (ticker_data[feature] - rolling_mean) / rolling_std

    test_normalized.append(ticker_data)

test_normalized = pd.concat(test_normalized, ignore_index=True)

# Normalize macro features (global z-score)
print(f"   Normalizing macro features...")
for feature in MACRO_FEATURES:
    mean_val = test_normalized[feature].mean()
    std_val = test_normalized[feature].std()
    if std_val == 0:
        std_val = 1e-8
    test_normalized[f'{feature}_norm'] = (test_normalized[feature] - mean_val) / std_val

# Select final columns
NORMALIZED_FEATURES = [f'{f}_norm' for f in REQUIRED_BASE_FEATURES]
final_cols = ['date', 'ticker'] + NORMALIZED_FEATURES
test_final = test_normalized[final_cols].copy()

# Sort by date and ticker
test_final = test_final.sort_values(['date', 'ticker']).reset_index(drop=True)

print(f"\n💾 Saving to {OUTPUT_FILE}...")
test_final.to_parquet(OUTPUT_FILE, index=False)

# Verify
print(f"\n✓ test_weekly.parquet created successfully!")
print(f"   Rows: {len(test_final):,}")
print(f"   Columns: {len(test_final.columns)}")
print(f"   Date range: {test_final['date'].min().date()} to {test_final['date'].max().date()}")
print(f"   Tickers: {test_final['ticker'].nunique()} ({', '.join(sorted(test_final['ticker'].unique()))})")
print(f"   Features: {len(NORMALIZED_FEATURES)} (all with _norm suffix)")

# Verify features
print(f"\n✓ Features included:")
for i, feat in enumerate(NORMALIZED_FEATURES, 1):
    print(f"   {i:2d}. {feat}")

print(f"\n{'=' * 80}")
print("✅ DONE! You can now run 03_backtests_all_models.ipynb")
print("=" * 80)
