# Real Sentiment Agent Implementation Plan

**Project**: Multi-Hierarchical RL Portfolio System
**Feature**: Real Sentiment Data Integration
**Timeline**: 2-3 weeks for full implementation
**Difficulty**: Advanced (requires API keys, data processing)

---

## Executive Summary

Replace the current fake "sentiment" agent with a real sentiment analysis system using:
- **News Sentiment**: FinBERT NLP analysis of financial news
- **Social Media**: Twitter/Reddit mentions and sentiment
- **Options Market**: Put/Call ratios, unusual options activity
- **Analyst Data**: Recommendations, price targets, upgrades/downgrades
- **Insider Trading**: Form 4 filings, insider buying/selling

**Expected Impact**: +0.5 to +1.5 Sharpe ratio improvement through alternative alpha signals.

---

## Phase 1: Infrastructure Setup (Week 1)

### 1.1 Install Required Dependencies

```bash
# NLP and ML
pip install transformers torch sentence-transformers

# Financial data APIs
pip install yfinance alpha_vantage finnhub-python polygon-api-client

# Web scraping (for free sources)
pip install beautifulsoup4 requests selenium praw

# Data processing
pip install pandas numpy scipy scikit-learn
```

### 1.2 API Key Setup

**Required API Keys** (mix of free and paid):

| Service | Purpose | Cost | Sign-up |
|---------|---------|------|---------|
| **Finnhub** | News, analyst ratings, insider trading | Free tier: 60 calls/min | finnhub.io |
| **Alpha Vantage** | News sentiment, company fundamentals | Free tier: 500 calls/day | alphavantage.co |
| **Polygon.io** | Options data, news | Free tier: 5 calls/min | polygon.io |
| **Reddit API** | Reddit sentiment (PRAW) | Free | reddit.com/prefs/apps |
| **Twitter API** | Twitter mentions (optional, expensive) | Paid ~$100/month | developer.twitter.com |

**API Keys File** (`config/api_keys.json`):
```json
{
  "finnhub": "YOUR_FINNHUB_KEY",
  "alpha_vantage": "YOUR_ALPHA_VANTAGE_KEY",
  "polygon": "YOUR_POLYGON_KEY",
  "reddit_client_id": "YOUR_REDDIT_CLIENT_ID",
  "reddit_client_secret": "YOUR_REDDIT_SECRET",
  "twitter_bearer_token": "OPTIONAL_TWITTER_TOKEN"
}
```

### 1.3 Create Sentiment Data Module

**File Structure**:
```
helpers/
├── sentiment_data.py          # NEW: Main sentiment data collection
├── sentiment_features.py       # NEW: Feature engineering from sentiment
├── sentiment_cache.py          # NEW: Caching to avoid repeated API calls
└── config.py                   # UPDATE: Add sentiment config
```

---

## Phase 2: News Sentiment (Week 1)

### 2.1 FinBERT Integration

**Model**: `ProsusAI/finbert` (SOTA financial sentiment model)

**Features to Create**:
```python
NEWS_SENTIMENT_FEATURES = [
    'news_sentiment_score',      # Mean sentiment [-1, 1]
    'news_sentiment_std',         # Sentiment volatility
    'news_volume_24h',            # Number of articles
    'news_volume_7d',             # Weekly article count
    'positive_news_pct',          # % of positive articles
    'negative_news_count',        # Count of negative articles
    'news_momentum',              # Change in sentiment
]
```

**Implementation**:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FinBERTAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()

    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of financial text."""
        inputs = self.tokenizer(text, return_tensors="pt",
                                padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Labels: negative, neutral, positive
        sentiment_score = probs[0][2].item() - probs[0][0].item()  # positive - negative

        return {
            'score': sentiment_score,  # [-1, 1]
            'positive_prob': probs[0][2].item(),
            'neutral_prob': probs[0][1].item(),
            'negative_prob': probs[0][0].item()
        }
```

**Data Sources**:
1. **Finnhub News API** (free tier: good coverage)
2. **Alpha Vantage News** (free tier: limited)
3. **Yahoo Finance RSS** (free, scraping needed)

---

## Phase 3: Social Media Sentiment (Week 2)

### 3.1 Reddit Sentiment (r/wallstreetbets, r/stocks)

**Features**:
```python
REDDIT_FEATURES = [
    'reddit_mentions',           # Number of mentions
    'reddit_sentiment',          # Average sentiment
    'reddit_upvote_ratio',       # Engagement metric
    'wsb_mentions',              # WallStreetBets specific
]
```

**Implementation**:
```python
import praw
from datetime import datetime, timedelta

class RedditSentimentCollector:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def get_ticker_sentiment(self, ticker: str, lookback_days: int = 7):
        """Get Reddit sentiment for a ticker."""
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        mentions = []

        for sub_name in subreddits:
            subreddit = self.reddit.subreddit(sub_name)

            # Search for ticker mentions
            for submission in subreddit.search(ticker, time_filter='week', limit=100):
                mentions.append({
                    'title': submission.title,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'upvote_ratio': submission.upvote_ratio,
                    'created_utc': submission.created_utc
                })

        return mentions
```

### 3.2 Twitter Sentiment (Optional - Expensive)

**Alternative**: Use free Twitter scraping tools like `snscrape` or `tweepy` academic tier.

---

## Phase 4: Options Market Sentiment (Week 2)

### 4.1 Put/Call Ratios and Implied Volatility

**Features**:
```python
OPTIONS_FEATURES = [
    'put_call_ratio',            # Put volume / Call volume
    'put_call_ratio_oi',         # Using open interest
    'iv_percentile',             # Implied vol percentile (52w)
    'iv_skew',                   # 25-delta put vs call IV
    'unusual_options_volume',    # Volume vs average
]
```

**Data Source**: Polygon.io or Yahoo Finance

**Implementation**:
```python
import yfinance as yf

def get_options_sentiment(ticker: str):
    """Calculate options-based sentiment."""
    stock = yf.Ticker(ticker)

    # Get options chain
    try:
        options_dates = stock.options
        if not options_dates:
            return None

        # Get nearest expiry
        nearest_expiry = options_dates[0]
        opt_chain = stock.option_chain(nearest_expiry)

        calls = opt_chain.calls
        puts = opt_chain.puts

        # Calculate put/call ratio
        put_volume = puts['volume'].sum()
        call_volume = calls['volume'].sum()
        pc_ratio = put_volume / (call_volume + 1e-10)

        # Calculate put/call ratio by open interest
        put_oi = puts['openInterest'].sum()
        call_oi = calls['openInterest'].sum()
        pc_ratio_oi = put_oi / (call_oi + 1e-10)

        return {
            'put_call_ratio': pc_ratio,
            'put_call_ratio_oi': pc_ratio_oi,
            'total_call_volume': call_volume,
            'total_put_volume': put_volume
        }
    except:
        return None
```

---

## Phase 5: Analyst & Insider Data (Week 3)

### 5.1 Analyst Recommendations

**Features**:
```python
ANALYST_FEATURES = [
    'analyst_rating_avg',        # Average rating (1-5 scale)
    'analyst_rating_change',     # Change in past month
    'num_analysts',              # Analyst coverage
    'price_target_return',       # (Target - Current) / Current
    'upgrade_count_30d',         # Recent upgrades
    'downgrade_count_30d',       # Recent downgrades
]
```

**Data Source**: Finnhub, Alpha Vantage

**Implementation**:
```python
import finnhub

def get_analyst_sentiment(ticker: str, api_key: str):
    """Get analyst recommendations."""
    finnhub_client = finnhub.Client(api_key=api_key)

    # Get recommendations
    recommendations = finnhub_client.recommendation_trends(ticker)

    if not recommendations:
        return None

    latest = recommendations[0]

    # Rating scale: 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell
    avg_rating = (
        latest['strongBuy'] * 1 +
        latest['buy'] * 2 +
        latest['hold'] * 3 +
        latest['sell'] * 4 +
        latest['strongSell'] * 5
    ) / (latest['strongBuy'] + latest['buy'] + latest['hold'] +
         latest['sell'] + latest['strongSell'] + 1e-10)

    return {
        'analyst_rating_avg': avg_rating,
        'num_analysts': sum([latest['strongBuy'], latest['buy'],
                            latest['hold'], latest['sell'], latest['strongSell']]),
        'strong_buy_count': latest['strongBuy'],
        'buy_count': latest['buy'],
        'hold_count': latest['hold']
    }
```

### 5.2 Insider Trading

**Features**:
```python
INSIDER_FEATURES = [
    'insider_buy_transactions',  # Count of insider buys
    'insider_sell_transactions', # Count of insider sells
    'insider_buy_value',         # Total value of buys
    'insider_sentiment',         # (Buys - Sells) / (Buys + Sells)
]
```

**Data Source**: Finnhub, SEC Edgar

---

## Phase 6: Feature Engineering Pipeline

### 6.1 New Sentiment Feature Set

**Final Feature List** (15 features):
```python
REAL_SENTIMENT_FEATURES = [
    # News Sentiment (4 features)
    'news_sentiment_score',      # [-1, 1]
    'news_volume_7d',            # Weekly article count
    'news_sentiment_momentum',   # 7d change in sentiment
    'negative_news_ratio',       # % negative articles

    # Social Sentiment (3 features)
    'reddit_mentions',           # Log-scaled mentions
    'reddit_sentiment',          # Average sentiment
    'social_momentum',           # Change in social activity

    # Options Sentiment (4 features)
    'put_call_ratio',            # Volume-based
    'put_call_ratio_oi',         # Open interest based
    'iv_percentile',             # Implied vol rank
    'unusual_options_activity',  # Anomaly detection

    # Analyst Sentiment (2 features)
    'analyst_rating_avg',        # 1-5 scale (inverted)
    'price_target_return',       # Expected return

    # Insider Sentiment (2 features)
    'insider_buy_ratio',         # Buys / (Buys + Sells)
    'insider_transaction_value'  # Net value (scaled)
]
```

### 6.2 Data Aggregation Strategy

**Weekly Aggregation**:
```python
def aggregate_sentiment_weekly(daily_data: pd.DataFrame, ticker: str) -> pd.Series:
    """Aggregate daily sentiment to weekly (Friday close)."""

    weekly = {
        # News: mean sentiment, sum volume
        'news_sentiment_score': daily_data['news_sentiment'].mean(),
        'news_volume_7d': daily_data['news_count'].sum(),

        # Social: mean sentiment, max mentions
        'reddit_sentiment': daily_data['reddit_sentiment'].mean(),
        'reddit_mentions': daily_data['reddit_mentions'].max(),

        # Options: last available value
        'put_call_ratio': daily_data['put_call_ratio'].iloc[-1],
        'iv_percentile': daily_data['iv_percentile'].iloc[-1],

        # Analyst: last available value (changes slowly)
        'analyst_rating_avg': daily_data['analyst_rating'].iloc[-1],
        'price_target_return': daily_data['price_target_return'].iloc[-1],
    }

    return pd.Series(weekly)
```

---

## Phase 7: Caching & Rate Limiting

### 7.1 Cache Strategy

**Problem**: API rate limits (60 calls/min for Finnhub)

**Solution**: Cache data locally

```python
import json
from pathlib import Path
from datetime import datetime, timedelta

class SentimentCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def get(self, ticker: str, date: str, data_type: str):
        """Get cached data if available and fresh."""
        cache_file = self.cache_dir / f"{ticker}_{data_type}_{date}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def set(self, ticker: str, date: str, data_type: str, data: dict):
        """Cache data."""
        cache_file = self.cache_dir / f"{ticker}_{data_type}_{date}.json"

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
```

---

## Phase 8: Integration with Existing System

### 8.1 Update Configuration

**File**: `helpers/config.py`

```python
# Add new sentiment feature list
class AgentFeatures:
    SENTIMENT = [
        # News Sentiment (4)
        'news_sentiment_score',
        'news_volume_7d',
        'news_sentiment_momentum',
        'negative_news_ratio',

        # Social Sentiment (3)
        'reddit_mentions',
        'reddit_sentiment',
        'social_momentum',

        # Options Sentiment (4)
        'put_call_ratio',
        'put_call_ratio_oi',
        'iv_percentile',
        'unusual_options_activity',

        # Analyst & Insider (4)
        'analyst_rating_avg',
        'price_target_return',
        'insider_buy_ratio',
        'insider_transaction_value'
    ]
```

### 8.2 Create New Data Prep Script

**File**: `scripts/prepare_sentiment_data.py`

```python
"""
Sentiment Data Preparation Script

Collects real sentiment data from multiple sources and prepares
features for the Sentiment Agent.

Usage:
    python scripts/prepare_sentiment_data.py --start-date 2020-01-01 --end-date 2025-11-04
"""

import argparse
from pathlib import Path
from helpers.sentiment_data import SentimentDataCollector
from helpers.sentiment_features import SentimentFeatureEngineer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--tickers', nargs='+', default=['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG'])
    args = parser.parse_args()

    # Initialize collectors
    collector = SentimentDataCollector(api_keys_file='config/api_keys.json')
    engineer = SentimentFeatureEngineer()

    # Collect data for each ticker
    for ticker in args.tickers:
        print(f"\nCollecting sentiment data for {ticker}...")

        # Collect from all sources
        news_data = collector.collect_news(ticker, args.start_date, args.end_date)
        social_data = collector.collect_social(ticker, args.start_date, args.end_date)
        options_data = collector.collect_options(ticker, args.start_date, args.end_date)
        analyst_data = collector.collect_analyst(ticker, args.start_date, args.end_date)
        insider_data = collector.collect_insider(ticker, args.start_date, args.end_date)

        # Engineer features
        features = engineer.create_features(
            news=news_data,
            social=social_data,
            options=options_data,
            analyst=analyst_data,
            insider=insider_data
        )

        # Save to cache
        features.to_csv(f'data_hierarchical/sentiment_raw/{ticker}_sentiment.csv')
        print(f"  ✅ Saved {len(features)} weeks of sentiment data")

if __name__ == '__main__':
    main()
```

---

## Phase 9: Testing & Validation

### 9.1 Data Quality Checks

```python
def validate_sentiment_data(features_df):
    """Validate sentiment feature quality."""
    checks = {
        'no_nulls': features_df.isnull().sum().sum() == 0,
        'no_inf': not np.isinf(features_df.select_dtypes(include=[np.number])).any().any(),
        'reasonable_ranges': all([
            features_df['news_sentiment_score'].between(-1, 1).all(),
            features_df['put_call_ratio'].between(0, 5).all(),
            features_df['analyst_rating_avg'].between(1, 5).all(),
        ]),
        'sufficient_coverage': len(features_df) >= 200,  # At least 200 weeks
    }

    return checks
```

### 9.2 Performance Comparison

**Experiment Design**:
```
1. Baseline: Current fake sentiment agent (Sharpe: -1.05)
2. Test: Real sentiment agent
3. Metric: Test set Sharpe ratio

Expected improvement: +1.5 to +2.5 Sharpe points
```

---

## Timeline & Milestones

### Week 1: Infrastructure & News
- [ ] Day 1-2: API setup, install dependencies
- [ ] Day 3-4: FinBERT integration, news collection
- [ ] Day 5: News feature engineering, caching

### Week 2: Social & Options
- [ ] Day 6-7: Reddit sentiment collector
- [ ] Day 8-9: Options data collection
- [ ] Day 10: Social/options feature engineering

### Week 3: Analyst/Insider & Integration
- [ ] Day 11-12: Analyst recommendations, insider trading
- [ ] Day 13-14: Full pipeline integration
- [ ] Day 15: Testing, validation, retraining

---

## Cost Estimation

**API Costs** (monthly):
- Finnhub free tier: $0 (60 calls/min, sufficient for weekly updates)
- Alpha Vantage free: $0 (500 calls/day)
- Polygon.io free: $0 (5 calls/min)
- Reddit API: $0 (free)
- Twitter API: $0-100 (optional, can skip)

**Compute Costs**:
- FinBERT inference: ~$0 (runs locally, small model)
- Total: **$0-100/month**

**Data Storage**:
- Cached sentiment data: ~100MB per year
- Total: Negligible

---

## Risk Mitigation

### Risk 1: API Rate Limits
**Mitigation**: Implement caching, batch requests, use free tiers

### Risk 2: Missing Data
**Mitigation**: Fill forward for missing values, use multiple data sources

### Risk 3: Model Overfitting
**Mitigation**: Cross-validate on multiple time periods, monitor out-of-sample Sharpe

### Risk 4: Data Quality
**Mitigation**: Validation checks, outlier detection, sanity tests

---

## Success Metrics

**Minimum Viable Product (MVP)**:
- [ ] 15 real sentiment features implemented
- [ ] Data collection for 5+ years (2020-2025)
- [ ] Test Sharpe > 0.5 (better than current -1.05)

**Target Performance**:
- [ ] Test Sharpe > 1.5
- [ ] Correlation with Technical agent < 0.3
- [ ] Consistent positive returns across market regimes

**Stretch Goals**:
- [ ] Test Sharpe > 2.0
- [ ] Outperform Technical agent standalone
- [ ] Add real-time sentiment updates

---

## Next Steps

1. **Get Approval**: Review this plan, confirm Option B
2. **Setup APIs**: Register for Finnhub, Alpha Vantage, Polygon, Reddit
3. **Start Coding**: Begin with Phase 1 (infrastructure)
4. **Iterative Development**: Build one data source at a time
5. **Test Early**: Validate each component before integration

---

**Ready to proceed?** Let me know when you have API keys, and I'll start building!
