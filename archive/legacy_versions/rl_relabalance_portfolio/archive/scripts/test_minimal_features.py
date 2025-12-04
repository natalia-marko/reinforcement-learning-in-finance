#!/usr/bin/env python3
"""
Test the minimal feature pipeline on actual training data.
Compare old (100 features) vs new (20-25 features) approach.
"""

import pandas as pd
import json
from pathlib import Path
from feature_fixes import (
    apply_minimal_feature_pipeline,
    compare_feature_sets,
    get_all_minimal_features
)

# Paths
DATA_PATH = Path('data/processed/train.parquet')
METADATA_PATH = Path('data/processed/metadata.json')

print("=" * 80)
print("TESTING MINIMAL FEATURE PIPELINE")
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

# Show sample/feature ratio BEFORE
n_samples = len(df_train)
print(f"\nCURRENT STATE:")
print(f"  Samples: {n_samples}")
print(f"  Features: {len(current_features)}")
print(f"  Ratio: {n_samples / len(current_features):.1f} samples/feature")
print(f"  Status: {'⚠️ HIGH OVERFITTING RISK' if n_samples / len(current_features) < 10 else '✓ OK'}")

# Apply minimal feature pipeline
print("\n" + "=" * 80)
print("APPLYING MINIMAL FEATURE PIPELINE")
print("=" * 80)

df_processed, final_features = apply_minimal_feature_pipeline(
    df_train,
    current_features,
    ticker_col='ticker',
    normalize=True,
    norm_window=52,  # 52 weeks for weekly data
    verbose=True
)

# Compare feature sets
print("\n")
comparison = compare_feature_sets(current_features, final_features, verbose=True)

# Show which features were kept
print("\n" + "=" * 80)
print("FEATURES KEPT FROM ORIGINAL SET")
print("=" * 80)

minimal_features_no_suffix = get_all_minimal_features()
kept_from_original = [f for f in minimal_features_no_suffix if f in current_features]

if kept_from_original:
    print(f"\nFound {len(kept_from_original)}/{len(minimal_features_no_suffix)} minimal features in current data:")
    for f in kept_from_original:
        print(f"  ✓ {f}")
else:
    print("\nNo minimal features found in current data!")
    print("This suggests feature names may have changed.")

missing_features = [f for f in minimal_features_no_suffix if f not in current_features]
if missing_features:
    print(f"\nMissing {len(missing_features)} minimal features:")
    for f in missing_features[:10]:  # Show first 10
        print(f"  ✗ {f}")
    if len(missing_features) > 10:
        print(f"  ... and {len(missing_features) - 10} more")

# Save processed data for testing
output_path = Path('data/processed/train_minimal_features.parquet')
df_processed.to_parquet(output_path)
print(f"\n✓ Saved processed data to: {output_path}")

# Update metadata
metadata_minimal = {
    'feature_cols': final_features,
    'tickers': metadata.get('tickers', []),
    'date_col': metadata.get('date_col', 'date'),
    'feature_engineering': 'evidence_based_minimal',
    'normalization': 'per_ticker_zscore',
    'research_citations': [
        'Jiang et al. (2017)',
        'Zhang et al. (2020)',
        'Moody & Saffell (2001)'
    ]
}

metadata_output = Path('data/processed/metadata_minimal.json')
with open(metadata_output, 'w') as f:
    json.dump(metadata_minimal, f, indent=2)
print(f"✓ Saved metadata to: {metadata_output}")

# Final summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nFeature reduction: {len(current_features)} → {len(final_features)}")
print(f"Reduction: {comparison['reduction_pct']:.1f}%")
print(f"\nSample/feature ratio:")
print(f"  Before: {n_samples / len(current_features):.1f}")
print(f"  After:  {n_samples / len(final_features):.1f}")
print(f"  Improvement: {(n_samples / len(final_features)) / (n_samples / len(current_features)):.1f}x")

print("\n✓ Ready to retrain with minimal features!")
print("\nNext steps:")
print("  1. Use train_minimal_features.parquet for training")
print("  2. Load feature_cols from metadata_minimal.json")
print("  3. Expect better generalization and less overfitting")
