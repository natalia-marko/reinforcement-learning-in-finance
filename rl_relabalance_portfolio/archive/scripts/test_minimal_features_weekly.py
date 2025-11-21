#!/usr/bin/env python3
"""
Test the minimal feature pipeline on WEEKLY training data.
Compare old (100 features) vs new (~20 features) approach.
"""

import pandas as pd
import json
from pathlib import Path
from feature_fixes_weekly import (
    apply_minimal_feature_pipeline_weekly,
    get_all_minimal_features_weekly
)

# Paths
DATA_PATH = Path('data/processed/train.parquet')
METADATA_PATH = Path('data/processed/metadata.json')

print("=" * 80)
print("TESTING MINIMAL FEATURE PIPELINE (WEEKLY DATA)")
print("=" * 80)

# Load data
if not DATA_PATH.exists():
    print(f"Error: {DATA_PATH} not found")
    print("Please run 01_data_preparation.ipynb first")
    exit(1)

if not METADATA_PATH.exists():
    print(f"Error: {METADATA_PATH} not found")
    print("Please run 01_data_preparation.ipynb first")
    exit(1)

print(f"\nLoading data from {DATA_PATH}")
df_train = pd.read_parquet(DATA_PATH)

print(f"Loading metadata from {METADATA_PATH}")
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

current_features = metadata['feature_cols']

print(f"\nData shape: {df_train.shape}")
print(f"Current features: {len(current_features)}")
print(f"Date range: {df_train['date'].min()} to {df_train['date'].max()}")
print(f"Tickers: {df_train['ticker'].nunique()}")
print(f"Unique tickers: {sorted(df_train['ticker'].unique())}")

# Show sample/feature ratio BEFORE
n_samples = len(df_train)
print(f"\nCURRENT STATE:")
print(f"  Samples: {n_samples}")
print(f"  Features: {len(current_features)}")
print(f"  Ratio: {n_samples / len(current_features):.1f} samples/feature")
print(f"  Status: {'⚠️ HIGH OVERFITTING RISK' if n_samples / len(current_features) < 10 else '✓ OK'}")

# Show which minimal features are available
minimal_features_target = get_all_minimal_features_weekly()
available_minimal = [f for f in minimal_features_target if f in current_features]
missing_minimal = [f for f in minimal_features_target if f not in current_features]

print(f"\n" + "=" * 80)
print("FEATURE AVAILABILITY CHECK")
print("=" * 80)
print(f"Target minimal features: {len(minimal_features_target)}")
print(f"Available in data: {len(available_minimal)}")
print(f"Missing: {len(missing_minimal)}")

if available_minimal:
    print(f"\n✓ Available features ({len(available_minimal)}):")
    for f in available_minimal:
        print(f"  ✓ {f}")

if missing_minimal:
    print(f"\n✗ Missing features ({len(missing_minimal)}):")
    for f in missing_minimal:
        print(f"  ✗ {f}")

# Apply minimal feature pipeline
print("\n" + "=" * 80)
print("APPLYING MINIMAL FEATURE PIPELINE")
print("=" * 80)

df_processed, final_features = apply_minimal_feature_pipeline_weekly(
    df_train,
    current_features,
    ticker_col='ticker',
    normalize=True,
    norm_window=52,  # 52 weeks = 1 year for weekly data
    verbose=True
)

# Check for NaN values
print("\n" + "=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)

for feature in final_features:
    nan_count = df_processed[feature].isna().sum()
    nan_pct = (nan_count / len(df_processed)) * 100
    if nan_count > 0:
        print(f"  {feature}: {nan_count} NaN ({nan_pct:.1f}%)")

# Remove rows with NaN in normalized features
original_len = len(df_processed)
df_processed = df_processed.dropna(subset=final_features)
dropped_rows = original_len - len(df_processed)

if dropped_rows > 0:
    print(f"\nDropped {dropped_rows} rows with NaN ({dropped_rows/original_len*100:.1f}%)")
    print(f"Remaining samples: {len(df_processed)}")

# Save processed data
output_path = Path('data/processed/train_minimal_features_weekly.parquet')
df_processed.to_parquet(output_path)
print(f"\n✓ Saved processed data to: {output_path}")

# Update metadata
metadata_minimal = {
    'feature_cols': final_features,
    'tickers': metadata.get('tickers', sorted(df_train['ticker'].unique())),
    'date_col': metadata.get('date_col', 'date'),
    'data_frequency': 'weekly',
    'feature_engineering': 'evidence_based_minimal_weekly',
    'normalization': 'per_ticker_zscore_52w',
    'research_citations': [
        'Jiang et al. (2017)',
        'Zhang et al. (2020)',
        'Moody & Saffell (2001)'
    ]
}

metadata_output = Path('data/processed/metadata_minimal_weekly.json')
with open(metadata_output, 'w') as f:
    json.dump(metadata_minimal, f, indent=2)
print(f"✓ Saved metadata to: {metadata_output}")

# Final summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nFeature reduction: {len(current_features)} → {len(final_features)}")
print(f"Reduction: {(1 - len(final_features)/len(current_features))*100:.1f}%")
print(f"\nSample/feature ratio:")
print(f"  Before: {n_samples / len(current_features):.1f}")
print(f"  After:  {len(df_processed) / len(final_features):.1f}")
improvement = (len(df_processed) / len(final_features)) / (n_samples / len(current_features))
print(f"  Improvement: {improvement:.1f}x")

print("\n✓ Ready to retrain with minimal features (WEEKLY DATA)!")
print("\nNext steps:")
print("  1. Update config.py to use train_minimal_features_weekly.parquet")
print("  2. Load feature_cols from metadata_minimal_weekly.json")
print("  3. Retrain models")
print("  4. Compare performance to baseline (100 features)")
