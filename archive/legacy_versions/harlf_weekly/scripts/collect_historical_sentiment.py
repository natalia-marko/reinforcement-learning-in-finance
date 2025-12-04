"""
Collect historical sentiment data for all tickers.

This script runs overnight to collect 5 years of sentiment data from multiple sources:
- News sentiment (FinBERT analysis)
- Social media (Reddit)
- Options market data
- Analyst recommendations
- Insider trading activity

Author: Multi-Hierarchical RL Portfolio System
"""

import sys
import os
from pathlib import Path

# Add project root to path (handle both running from scripts/ or project root)
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from helpers.sentiment_data import SentimentDataCollector
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path

# Configuration
TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']

# IMPORTANT: Free Finnhub tier = 60 calls/min
# Historical news = 1 call per day per ticker
# Recommendation: Start with recent data (last 3 months)
# For full 5-year history, need paid tier or run over multiple days
START_DATE = '2024-08-01'  # Last 3 months (reasonable for free tier)
END_DATE = '2025-11-06'

# Create output directory (use project root)
OUTPUT_DIR = project_root / 'data_hierarchical' / 'sentiment_raw'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize collector (use project root config)
print("Initializing Sentiment Data Collector...")
config_path = project_root / 'config' / 'api_keys.json'
collector = SentimentDataCollector(api_keys_file=str(config_path))

print(f"\nStarting data collection for {len(TICKERS)} tickers")
print(f"Date range: {START_DATE} to {END_DATE}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 70)

# Collect for each ticker
for i, ticker in enumerate(TICKERS, 1):
    print(f"\n{'='*70}")
    print(f"COLLECTING SENTIMENT DATA FOR {ticker} ({i}/{len(TICKERS)})")
    print(f"{'='*70}")

    try:
        # Collect news (with rate limiting)
        print("\n1. Collecting news...")
        news_df = collector.collect_news(ticker, START_DATE, END_DATE, analyze_sentiment=True)
        if not news_df.empty:
            news_df.to_csv(OUTPUT_DIR / f'{ticker}_news.csv', index=False)
            print(f"  ✅ Saved {len(news_df)} news articles")
        else:
            print(f"  ⚠️  No news data collected")

        # Wait to respect rate limits
        print("  Waiting 60 seconds (rate limit)...")
        time.sleep(60)

        # Collect social (recent data only)
        print("\n2. Collecting social data...")
        social_df = collector.collect_social(ticker, lookback_days=30)
        if not social_df.empty:
            social_df.to_csv(OUTPUT_DIR / f'{ticker}_social.csv', index=False)
            print(f"  ✅ Saved {len(social_df)} social mentions")
        else:
            print(f"  ⚠️  No social data collected")

        # Collect options (single snapshot)
        print("\n3. Collecting options data...")
        options_data = collector.collect_options(ticker)
        if options_data:
            pd.DataFrame([options_data]).to_csv(
                OUTPUT_DIR / f'{ticker}_options.csv',
                index=False
            )
            print(f"  ✅ Saved options snapshot")
        else:
            print(f"  ⚠️  No options data collected")

        # Collect analyst data
        print("\n4. Collecting analyst data...")
        analyst_recs, price_targets = collector.collect_analyst_data(ticker, lookback_months=12)

        if analyst_recs is not None and not analyst_recs.empty:
            analyst_recs.to_csv(OUTPUT_DIR / f'{ticker}_analyst_recs.csv', index=False)
            print(f"  ✅ Saved {len(analyst_recs)} analyst recommendation periods")
        else:
            print(f"  ⚠️  No analyst recommendations collected")

        if price_targets:
            pd.DataFrame([price_targets]).to_csv(
                OUTPUT_DIR / f'{ticker}_price_targets.csv',
                index=False
            )
            print(f"  ✅ Saved price targets")
        else:
            print(f"  ⚠️  No price targets collected")

        # Collect insider trading data
        print("\n5. Collecting insider trading data...")
        insider_trans, insider_sentiment = collector.collect_insider_data(ticker, lookback_months=6)

        if insider_trans is not None and not insider_trans.empty:
            insider_trans.to_csv(OUTPUT_DIR / f'{ticker}_insider_trans.csv', index=False)
            print(f"  ✅ Saved {len(insider_trans)} insider transactions")
        else:
            print(f"  ⚠️  No insider transactions collected")

        if insider_sentiment:
            pd.DataFrame([insider_sentiment]).to_csv(
                OUTPUT_DIR / f'{ticker}_insider_sentiment.csv',
                index=False
            )
            print(f"  ✅ Saved insider sentiment metrics")
        else:
            print(f"  ⚠️  No insider sentiment collected")

        print(f"\n✅ {ticker} complete!")

    except Exception as e:
        print(f"  ❌ Error collecting {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()

    # Rate limit between tickers
    if i < len(TICKERS):
        print(f"\n⏳ Waiting 120 seconds before next ticker...")
        time.sleep(120)

print("\n" + "="*70)
print("✅ ALL DATA COLLECTION COMPLETE!")
print("="*70)

# Print summary
print("\nData Collection Summary:")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Files created:")
for ticker in TICKERS:
    ticker_files = list(OUTPUT_DIR.glob(f'{ticker}_*.csv'))
    print(f"    {ticker}: {len(ticker_files)} files")

print("\nNext steps:")
print("  1. Verify data quality (see SENTIMENT_QUICKSTART.md)")
print("  2. Run feature engineering pipeline")
print("  3. Integrate with training scripts")
