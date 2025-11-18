"""
Example: Using the New Leak-Proof Data Pipeline

This script demonstrates how to use the enhanced pipeline with:
1. Fixed data leakage (split-first approach)
2. Enhanced macro indicators (15+ FRED series)
3. Optional sentiment data (news + social)

Run this to verify the pipeline works correctly.
"""

from pathlib import Path
import pandas as pd
import pickle
import json
from utile import complete_data_pipeline

# Configuration
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX']
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'
INTERVAL = '1wk'

# Directories
DATA_DIR = Path(__file__).parent / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed_new'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TESTING NEW LEAK-PROOF PIPELINE")
print("=" * 80)

# Test 1: Macro only (fastest, most reliable)
print("\n[TEST 1] Running pipeline with MACRO ONLY...")
print("-" * 80)

result = complete_data_pipeline(
    tickers=TICKERS,
    start_date=START_DATE,
    end_date=END_DATE,
    interval=INTERVAL,
    include_macro=True,
    include_sentiment=False,  # Disable sentiment for speed
    train_ratio=0.8,
    clip_percentiles=(0.01, 0.99),
    corr_threshold=0.95
)

train_df = result['train']
test_df = result['test']
scaler = result['scaler']
metadata = result['metadata']

print("\n✓ Pipeline completed successfully!")
print(f"  Train samples: {len(train_df)}")
print(f"  Test samples:  {len(test_df)}")
print(f"  Features:      {len(metadata['feature_cols'])}")

# Verification Tests
print("\n" + "=" * 80)
print("VERIFICATION TESTS")
print("=" * 80)

# Test 1: No temporal overlap
print("\n[Test 1] Checking for data leakage (temporal overlap)...")
train_end = pd.to_datetime(train_df['date'].max())
test_start = pd.to_datetime(test_df['date'].min())

if train_end < test_start:
    print(f"  ✓ PASS: No overlap")
    print(f"    Train ends:  {train_end.date()}")
    print(f"    Test starts: {test_start.date()}")
    print(f"    Gap:         {(test_start - train_end).days} days")
else:
    print(f"  ✗ FAIL: Overlap detected!")
    print(f"    Train ends:  {train_end.date()}")
    print(f"    Test starts: {test_start.date()}")

# Test 2: Feature consistency
print("\n[Test 2] Checking feature consistency...")
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)

if train_cols == test_cols:
    print(f"  ✓ PASS: Train and test have same {len(train_cols)} columns")
else:
    print(f"  ✗ FAIL: Column mismatch")
    print(f"    Train only: {train_cols - test_cols}")
    print(f"    Test only:  {test_cols - train_cols}")

# Test 3: Normalization
print("\n[Test 3] Checking normalization...")
non_feature_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
feature_cols = [c for c in train_df.columns if c not in non_feature_cols]

train_means = train_df[feature_cols].mean().mean()
train_stds = train_df[feature_cols].std().mean()

print(f"  Train mean: {train_means:.4f} (target: ~0.00)")
print(f"  Train std:  {train_stds:.4f} (target: ~1.00)")

if abs(train_means) < 0.1 and abs(train_stds - 1.0) < 0.2:
    print("  ✓ PASS: Training set properly normalized")
else:
    print("  ⚠ WARNING: Normalization may be off")

# Test 4: Macro indicators
print("\n[Test 4] Checking macro indicators...")
macro_keywords = ['treasury', 'fed_funds', 'vix', 'cpi', 'unemployment',
                  'ism', 'yield_curve', 'dxy', 'oil', 'spread', 'pce', 'baa']
macro_cols = [c for c in train_df.columns if any(kw in c.lower() for kw in macro_keywords)]

print(f"  Found {len(macro_cols)} macro indicator columns:")
for col in sorted(macro_cols)[:10]:  # Show first 10
    print(f"    - {col}")
if len(macro_cols) > 10:
    print(f"    ... and {len(macro_cols) - 10} more")

if len(macro_cols) >= 10:
    print("  ✓ PASS: Comprehensive macro coverage")
else:
    print(f"  ⚠ WARNING: Only {len(macro_cols)} macro features (expected 10+)")

# Test 5: Data quality
print("\n[Test 5] Checking data quality...")
train_na_pct = train_df[feature_cols].isna().sum().sum() / (len(train_df) * len(feature_cols)) * 100
test_na_pct = test_df[feature_cols].isna().sum().sum() / (len(test_df) * len(feature_cols)) * 100

print(f"  Train NaN: {train_na_pct:.2f}%")
print(f"  Test NaN:  {test_na_pct:.2f}%")

if train_na_pct < 1.0 and test_na_pct < 1.0:
    print("  ✓ PASS: Low missing data")
else:
    print("  ⚠ WARNING: High missing data percentage")

# Save results
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

train_df.to_parquet(PROCESSED_DATA_DIR / 'train.parquet', index=False)
test_df.to_parquet(PROCESSED_DATA_DIR / 'test.parquet', index=False)

with open(PROCESSED_DATA_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open(PROCESSED_DATA_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

print(f"\n✓ Saved to: {PROCESSED_DATA_DIR}")
print(f"  - train.parquet")
print(f"  - test.parquet")
print(f"  - scaler.pkl")
print(f"  - metadata.json")

# Optional: Test with sentiment (commented out by default)
print("\n" + "=" * 80)
print("OPTIONAL: SENTIMENT TEST")
print("=" * 80)
print("\nTo test sentiment integration, uncomment the code below and ensure:")
print("  1. You have a NewsAPI key in api_keys.json")
print("  2. You have installed: pip install newsapi-python nltk")
print("\nThen run the sentiment test.")

"""
# Uncomment to test sentiment:
print("\n[TEST 2] Running pipeline with MACRO + SENTIMENT...")
print("-" * 80)

result_sentiment = complete_data_pipeline(
    tickers=TICKERS,
    start_date=START_DATE,
    end_date=END_DATE,
    interval=INTERVAL,
    include_macro=True,
    include_sentiment=True,  # Enable sentiment
    sentiment_source='news',  # 'news', 'twitter', or 'combined'
    train_ratio=0.8
)

train_sentiment = result_sentiment['train']
test_sentiment = result_sentiment['test']

# Check sentiment columns
sentiment_cols = [c for c in train_sentiment.columns if 'sentiment' in c.lower()]
print(f"\nFound {len(sentiment_cols)} sentiment features:")
for col in sentiment_cols:
    print(f"  - {col}")

# Save sentiment version
train_sentiment.to_parquet(PROCESSED_DATA_DIR / 'train_with_sentiment.parquet', index=False)
test_sentiment.to_parquet(PROCESSED_DATA_DIR / 'test_with_sentiment.parquet', index=False)
print(f"\n✓ Saved sentiment-enhanced data to {PROCESSED_DATA_DIR}")
"""

print("\n" + "=" * 80)
print("ALL TESTS COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("  1. Review the data in data/processed_new/")
print("  2. Update your training notebook to use the new pipeline")
print("  3. Compare old vs new test metrics (new should be more realistic)")
print("  4. Optional: Enable sentiment after getting NewsAPI key")
print("\n")
