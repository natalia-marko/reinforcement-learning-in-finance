"""Check the actual distributions of macro features."""
import pandas as pd
import numpy as np

# Load the processed data
train = pd.read_parquet('data/processed_new/train.parquet')
test = pd.read_parquet('data/processed_new/test.parquet')

print("="*80)
print("MACRO FEATURE DISTRIBUTIONS")
print("="*80)

# Get macro features
macro_features = ['fed_funds_rate', 'unemployment_rate', 'gdp_growth',
                  'ism_manufacturing', 'cpi', 'treasury_2y', 'dxy']

for feat in macro_features:
    if feat not in train.columns:
        print(f"\n⚠️  {feat} NOT FOUND in training data")
        continue

    print(f"\n{feat.upper()}")
    print("-" * 40)
    nunique = train[feat].nunique()
    print(f"Unique values: {nunique}")
    print(f"Min: {train[feat].min():.4f}, Max: {train[feat].max():.4f}")
    print(f"Mean: {train[feat].mean():.4f}, Std: {train[feat].std():.4f}")

    # Show value distribution if fewer than 15 unique values
    if nunique <= 15:
        print("\nValue distribution:")
        value_counts = train[feat].value_counts().sort_index()
        for val, count in value_counts.items():
            pct = (count / len(train)) * 100
            print(f"  {val:8.4f}: {count:5d} observations ({pct:5.1f}%)")
    else:
        # Show deciles for continuous features
        print("\nDeciles:")
        deciles = train[feat].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        for q, val in deciles.items():
            print(f"  {int(q*100):3d}%: {val:8.4f}")

print("\n" + "="*80)
print("\nDATA PERIOD")
print("-" * 40)
if 'date' in train.columns:
    print(f"Train: {train['date'].min()} to {train['date'].max()}")
    print(f"Test:  {test['date'].min()} to {test['date'].max()}")
    print(f"Train rows: {len(train):,}, Test rows: {len(test):,}")
print("="*80)
