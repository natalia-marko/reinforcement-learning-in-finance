"""
Sentiment Data Collection Module

Collects real sentiment data from multiple sources:
- News sentiment (FinBERT)
- Social media (Reddit, Twitter)
- Options market (put/call ratios)
- Analyst recommendations
- Insider trading

Author: Multi-Hierarchical RL Portfolio System
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Optional imports (install as needed)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  transformers not installed. Run: pip install transformers torch")

try:
    import finnhub
    HAS_FINNHUB = True
except ImportError:
    HAS_FINNHUB = False
    print("⚠️  finnhub not installed. Run: pip install finnhub-python")

try:
    import praw
    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False
    print("⚠️  praw not installed. Run: pip install praw")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("⚠️  yfinance not installed")


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

class APIKeyManager:
    """Manage API keys from config file."""

    def __init__(self, config_path: str = 'config/api_keys.json'):
        self.config_path = Path(config_path)
        self.keys = self._load_keys()

    def _load_keys(self) -> Dict:
        """Load API keys from JSON file."""
        if not self.config_path.exists():
            print(f"⚠️  API keys file not found: {self.config_path}")
            print("Creating template file...")
            self._create_template()
            return {}

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _create_template(self):
        """Create template API keys file."""
        self.config_path.parent.mkdir(exist_ok=True, parents=True)

        template = {
            "finnhub": "YOUR_FINNHUB_KEY_HERE",
            "alpha_vantage": "YOUR_ALPHA_VANTAGE_KEY_HERE",
            "polygon": "YOUR_POLYGON_KEY_HERE",
            "reddit_client_id": "YOUR_REDDIT_CLIENT_ID_HERE",
            "reddit_client_secret": "YOUR_REDDIT_CLIENT_SECRET_HERE",
            "reddit_user_agent": "sentiment_collector_v1.0",
            "twitter_bearer_token": "OPTIONAL_TWITTER_TOKEN_HERE"
        }

        with open(self.config_path, 'w') as f:
            json.dump(template, f, indent=2)

        print(f"✅ Template created at {self.config_path}")
        print("Please fill in your API keys.")

    def get(self, key: str) -> Optional[str]:
        """Get API key."""
        value = self.keys.get(key)
        if not value or 'YOUR_' in value.upper():
            return None
        return value


# ============================================================================
# FINBERT NEWS SENTIMENT
# ============================================================================

class FinBERTAnalyzer:
    """
    Financial news sentiment analysis using FinBERT.

    FinBERT is a BERT model fine-tuned on financial news for sentiment analysis.
    Paper: https://arxiv.org/abs/1908.10063
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed")

        print(f"Loading FinBERT model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        print("✅ FinBERT loaded")

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of financial text.

        Args:
            text: Financial news article or snippet

        Returns:
            Dictionary with sentiment scores
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT labels: [negative, neutral, positive]
        negative_prob = probs[0][0].item()
        neutral_prob = probs[0][1].item()
        positive_prob = probs[0][2].item()

        # Compute aggregate score
        sentiment_score = positive_prob - negative_prob  # [-1, 1]

        return {
            'score': sentiment_score,
            'positive': positive_prob,
            'neutral': neutral_prob,
            'negative': negative_prob,
            'label': ['negative', 'neutral', 'positive'][probs[0].argmax().item()]
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts (more efficient)."""
        return [self.analyze(text) for text in texts]


# ============================================================================
# NEWS DATA COLLECTOR
# ============================================================================

class NewsCollector:
    """Collect financial news from multiple sources."""

    def __init__(self, api_keys: APIKeyManager):
        self.api_keys = api_keys
        self.finnhub_client = None

        if HAS_FINNHUB and api_keys.get('finnhub'):
            self.finnhub_client = finnhub.Client(api_key=api_keys.get('finnhub'))

    def collect_finnhub_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """
        Collect news from Finnhub API.

        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of news articles
        """
        if not self.finnhub_client:
            print("⚠️  Finnhub client not initialized")
            return []

        # Convert dates to timestamps
        start_ts = datetime.strptime(start_date, '%Y-%m-%d')
        end_ts = datetime.strptime(end_date, '%Y-%m-%d')

        articles = []
        current_date = start_ts
        api_call_count = 0
        max_calls_per_minute = 55  # Stay under 60/min limit

        # Collect day by day (Finnhub API limitation)
        while current_date <= end_ts:
            next_date = current_date + timedelta(days=1)

            # Rate limiting: pause every 55 calls for 60 seconds
            if api_call_count >= max_calls_per_minute:
                print(f"  ⏳ Rate limit: waiting 60 seconds (collected {len(articles)} articles so far)...")
                import time as time_module
                time_module.sleep(60)
                api_call_count = 0

            try:
                news = self.finnhub_client.company_news(
                    ticker,
                    _from=current_date.strftime('%Y-%m-%d'),
                    to=next_date.strftime('%Y-%m-%d')
                )
                api_call_count += 1

                for article in news:
                    articles.append({
                        'date': datetime.fromtimestamp(article['datetime']),
                        'headline': article['headline'],
                        'summary': article['summary'],
                        'source': article['source'],
                        'url': article['url']
                    })

            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'API limit' in error_str:
                    print(f"  ⏳ Rate limit hit, waiting 60 seconds...")
                    import time as time_module
                    time_module.sleep(60)
                    api_call_count = 0
                    # Retry this date
                    continue
                else:
                    print(f"  ⚠️  Error for {current_date.date()}: {error_str}")

            current_date = next_date

        return articles


# ============================================================================
# REDDIT SENTIMENT COLLECTOR
# ============================================================================

class RedditCollector:
    """Collect sentiment from Reddit."""

    def __init__(self, api_keys: APIKeyManager):
        self.api_keys = api_keys
        self.reddit = None

        if HAS_PRAW:
            client_id = api_keys.get('reddit_client_id')
            client_secret = api_keys.get('reddit_client_secret')
            user_agent = api_keys.get('reddit_user_agent') or 'sentiment_collector_v1.0'

            if client_id and client_secret:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )

    def collect_mentions(
        self,
        ticker: str,
        subreddits: List[str] = ['wallstreetbets', 'stocks', 'investing'],
        lookback_days: int = 7
    ) -> List[Dict]:
        """
        Collect Reddit mentions of ticker.

        Args:
            ticker: Stock ticker
            subreddits: List of subreddit names
            lookback_days: Number of days to look back

        Returns:
            List of Reddit posts mentioning ticker
        """
        if not self.reddit:
            print("⚠️  Reddit client not initialized")
            return []

        mentions = []
        cutoff_time = datetime.now() - timedelta(days=lookback_days)

        for sub_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(sub_name)

                # Search for ticker
                for submission in subreddit.search(ticker, time_filter='week', limit=100):
                    created_time = datetime.fromtimestamp(submission.created_utc)

                    if created_time < cutoff_time:
                        continue

                    mentions.append({
                        'date': created_time,
                        'subreddit': sub_name,
                        'title': submission.title,
                        'text': submission.selftext,
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'upvote_ratio': submission.upvote_ratio
                    })

            except Exception as e:
                print(f"  Error collecting from r/{sub_name}: {str(e)}")

        return mentions


# ============================================================================
# OPTIONS DATA COLLECTOR
# ============================================================================

class OptionsCollector:
    """Collect options market data."""

    def __init__(self):
        if not HAS_YFINANCE:
            raise ImportError("yfinance not installed")

    def collect_options_metrics(self, ticker: str) -> Optional[Dict]:
        """
        Collect options-based sentiment metrics.

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary with options metrics
        """
        try:
            stock = yf.Ticker(ticker)

            # Get options chain
            options_dates = stock.options
            if not options_dates:
                return None

            # Use nearest expiry
            nearest_expiry = options_dates[0]
            opt_chain = stock.option_chain(nearest_expiry)

            calls = opt_chain.calls
            puts = opt_chain.puts

            # Calculate put/call ratios
            put_volume = puts['volume'].fillna(0).sum()
            call_volume = calls['volume'].fillna(0).sum()
            pc_ratio = put_volume / (call_volume + 1e-10)

            put_oi = puts['openInterest'].fillna(0).sum()
            call_oi = calls['openInterest'].fillna(0).sum()
            pc_ratio_oi = put_oi / (call_oi + 1e-10)

            # Get implied volatility
            call_iv = calls['impliedVolatility'].fillna(0).mean()
            put_iv = puts['impliedVolatility'].fillna(0).mean()

            return {
                'date': datetime.now(),
                'put_call_ratio': pc_ratio,
                'put_call_ratio_oi': pc_ratio_oi,
                'call_volume': call_volume,
                'put_volume': put_volume,
                'call_iv': call_iv,
                'put_iv': put_iv,
                'iv_skew': put_iv - call_iv
            }

        except Exception as e:
            print(f"  Error collecting options data: {str(e)}")
            return None


# ============================================================================
# ANALYST RECOMMENDATIONS COLLECTOR
# ============================================================================

class AnalystCollector:
    """Collect analyst recommendations and price targets."""

    def __init__(self, api_keys: APIKeyManager):
        self.api_keys = api_keys
        self.finnhub_client = None

        if HAS_FINNHUB and api_keys.get('finnhub'):
            self.finnhub_client = finnhub.Client(api_key=api_keys.get('finnhub'))

    def collect_recommendations(
        self,
        ticker: str,
        lookback_months: int = 12
    ) -> Optional[pd.DataFrame]:
        """
        Collect analyst recommendations.

        Args:
            ticker: Stock ticker
            lookback_months: Number of months to look back

        Returns:
            DataFrame with analyst recommendations
        """
        if not self.finnhub_client:
            print("⚠️  Finnhub client not initialized")
            return None

        try:
            # Get recommendation trends
            recommendations = self.finnhub_client.recommendation_trends(ticker)

            if not recommendations:
                return None

            # Convert to DataFrame
            rec_df = pd.DataFrame(recommendations)

            # Parse period to datetime
            rec_df['date'] = pd.to_datetime(rec_df['period'])

            # Filter by lookback period
            cutoff_date = datetime.now() - timedelta(days=lookback_months * 30)
            rec_df = rec_df[rec_df['date'] >= cutoff_date]

            # Calculate aggregate metrics
            rec_df['total_ratings'] = (
                rec_df['strongBuy'] + rec_df['buy'] +
                rec_df['hold'] + rec_df['sell'] + rec_df['strongSell']
            )

            # Calculate weighted score: strongBuy=2, buy=1, hold=0, sell=-1, strongSell=-2
            rec_df['rating_score'] = (
                rec_df['strongBuy'] * 2 + rec_df['buy'] * 1 +
                rec_df['hold'] * 0 + rec_df['sell'] * (-1) +
                rec_df['strongSell'] * (-2)
            ) / (rec_df['total_ratings'] + 1e-10)

            return rec_df[['date', 'strongBuy', 'buy', 'hold', 'sell',
                          'strongSell', 'total_ratings', 'rating_score']]

        except Exception as e:
            print(f"  Error collecting analyst recommendations: {str(e)}")
            return None

    def collect_price_targets(self, ticker: str) -> Optional[Dict]:
        """
        Collect analyst price targets.

        Args:
            ticker: Stock ticker

        Returns:
            Dictionary with price target data
        """
        if not self.finnhub_client:
            return None

        try:
            # Get price target consensus
            price_targets = self.finnhub_client.price_target(ticker)

            if not price_targets:
                return None

            # Get current price for comparison
            try:
                if HAS_YFINANCE:
                    current_price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
                else:
                    current_price = None
            except:
                current_price = None

            result = {
                'date': datetime.now(),
                'target_high': price_targets.get('targetHigh'),
                'target_low': price_targets.get('targetLow'),
                'target_mean': price_targets.get('targetMean'),
                'target_median': price_targets.get('targetMedian'),
                'num_analysts': price_targets.get('numberOfAnalysts', 0)
            }

            # Calculate upside if we have current price
            if current_price and result['target_mean']:
                result['upside_pct'] = ((result['target_mean'] - current_price) / current_price) * 100
            else:
                result['upside_pct'] = None

            return result

        except Exception as e:
            print(f"  Error collecting price targets: {str(e)}")
            return None


# ============================================================================
# INSIDER TRADING COLLECTOR
# ============================================================================

class InsiderTradingCollector:
    """Collect insider trading activity."""

    def __init__(self, api_keys: APIKeyManager):
        self.api_keys = api_keys
        self.finnhub_client = None

        if HAS_FINNHUB and api_keys.get('finnhub'):
            self.finnhub_client = finnhub.Client(api_key=api_keys.get('finnhub'))

    def collect_insider_transactions(
        self,
        ticker: str,
        lookback_months: int = 6
    ) -> Optional[pd.DataFrame]:
        """
        Collect insider trading transactions.

        Args:
            ticker: Stock ticker
            lookback_months: Number of months to look back

        Returns:
            DataFrame with insider transactions
        """
        if not self.finnhub_client:
            print("⚠️  Finnhub client not initialized")
            return None

        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_months * 30)

            # Get insider transactions
            transactions = self.finnhub_client.stock_insider_transactions(
                ticker,
                _from=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )

            if not transactions or 'data' not in transactions:
                return None

            # Convert to DataFrame
            insider_df = pd.DataFrame(transactions['data'])

            if insider_df.empty:
                return None

            # Parse transaction date
            insider_df['date'] = pd.to_datetime(insider_df['transactionDate'])

            # Calculate transaction value (shares * price)
            insider_df['transaction_value'] = (
                insider_df['share'].fillna(0) * insider_df['transactionPrice'].fillna(0)
            )

            # Classify transaction type
            insider_df['is_buy'] = insider_df['transactionCode'].str.contains('P', na=False)
            insider_df['is_sell'] = insider_df['transactionCode'].str.contains('S', na=False)

            return insider_df[['date', 'name', 'share', 'transactionPrice',
                              'transaction_value', 'transactionCode', 'is_buy', 'is_sell']]

        except Exception as e:
            print(f"  Error collecting insider transactions: {str(e)}")
            return None

    def aggregate_insider_sentiment(
        self,
        transactions_df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Aggregate insider transactions into sentiment metrics.

        Args:
            transactions_df: DataFrame from collect_insider_transactions()

        Returns:
            Dictionary with aggregated insider sentiment
        """
        if transactions_df is None or transactions_df.empty:
            return None

        try:
            # Calculate buy/sell volumes
            buy_transactions = transactions_df[transactions_df['is_buy']]
            sell_transactions = transactions_df[transactions_df['is_sell']]

            buy_value = buy_transactions['transaction_value'].sum()
            sell_value = sell_transactions['transaction_value'].sum()
            net_value = buy_value - sell_value

            buy_count = len(buy_transactions)
            sell_count = len(sell_transactions)
            total_count = buy_count + sell_count

            # Calculate metrics
            buy_ratio = buy_count / (total_count + 1e-10)
            net_value_normalized = net_value / (buy_value + sell_value + 1e-10)

            return {
                'date': datetime.now(),
                'buy_transactions': buy_count,
                'sell_transactions': sell_count,
                'total_transactions': total_count,
                'buy_ratio': buy_ratio,
                'buy_value': buy_value,
                'sell_value': sell_value,
                'net_value': net_value,
                'net_value_ratio': net_value_normalized,
                # Sentiment interpretation: positive if more buying, negative if more selling
                'insider_sentiment_score': (buy_ratio - 0.5) * 2  # Scale to [-1, 1]
            }

        except Exception as e:
            print(f"  Error aggregating insider sentiment: {str(e)}")
            return None


# ============================================================================
# MAIN COLLECTOR CLASS
# ============================================================================

class SentimentDataCollector:
    """
    Main class to collect all sentiment data from multiple sources.

    Data Sources:
        - News sentiment (FinBERT analysis)
        - Social media (Reddit, Twitter)
        - Options market (put/call ratios, IV)
        - Analyst recommendations and price targets
        - Insider trading activity

    Usage:
        collector = SentimentDataCollector(api_keys_file='config/api_keys.json')

        # Collect news with sentiment analysis
        news = collector.collect_news('AAPL', '2024-01-01', '2024-12-31')

        # Collect social media mentions
        social = collector.collect_social('AAPL', lookback_days=7)

        # Collect options market data
        options = collector.collect_options('AAPL')

        # Collect analyst data
        analyst_recs, price_targets = collector.collect_analyst_data('AAPL')

        # Collect insider trading data
        insider_trans, insider_sentiment = collector.collect_insider_data('AAPL')
    """

    def __init__(self, api_keys_file: str = 'config/api_keys.json'):
        print("Initializing Sentiment Data Collector...")
        self.api_keys = APIKeyManager(api_keys_file)

        # Initialize collectors
        self.news_collector = NewsCollector(self.api_keys)
        self.reddit_collector = RedditCollector(self.api_keys)
        self.options_collector = OptionsCollector() if HAS_YFINANCE else None
        self.analyst_collector = AnalystCollector(self.api_keys)
        self.insider_collector = InsiderTradingCollector(self.api_keys)

        # Initialize FinBERT (heavy, load on demand)
        self.finbert = None

        print("✅ Sentiment Data Collector ready")

    def _ensure_finbert(self):
        """Load FinBERT if not already loaded."""
        if self.finbert is None and HAS_TRANSFORMERS:
            self.finbert = FinBERTAnalyzer()

    def collect_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        analyze_sentiment: bool = True
    ) -> pd.DataFrame:
        """
        Collect news and analyze sentiment.

        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            analyze_sentiment: Run FinBERT analysis

        Returns:
            DataFrame with news and sentiment
        """
        print(f"\nCollecting news for {ticker}...")

        # Collect news articles
        articles = self.news_collector.collect_finnhub_news(ticker, start_date, end_date)

        if not articles:
            print(f"  ⚠️  No news found for {ticker}")
            return pd.DataFrame()

        print(f"  ✅ Collected {len(articles)} articles")

        # Analyze sentiment
        if analyze_sentiment:
            print("  Analyzing sentiment with FinBERT...")
            self._ensure_finbert()

            for article in articles:
                # Analyze headline + summary
                text = f"{article['headline']}. {article['summary']}"
                sentiment = self.finbert.analyze(text)
                article['sentiment_score'] = sentiment['score']
                article['sentiment_label'] = sentiment['label']

        return pd.DataFrame(articles)

    def collect_social(
        self,
        ticker: str,
        lookback_days: int = 7
    ) -> pd.DataFrame:
        """Collect social media sentiment."""
        print(f"\nCollecting social sentiment for {ticker}...")

        mentions = self.reddit_collector.collect_mentions(ticker, lookback_days=lookback_days)

        if not mentions:
            print(f"  ⚠️  No social mentions found for {ticker}")
            return pd.DataFrame()

        print(f"  ✅ Collected {len(mentions)} mentions")

        return pd.DataFrame(mentions)

    def collect_options(self, ticker: str) -> Optional[Dict]:
        """Collect options market data."""
        if not self.options_collector:
            return None

        return self.options_collector.collect_options_metrics(ticker)

    def collect_analyst_data(
        self,
        ticker: str,
        lookback_months: int = 12
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Collect analyst recommendations and price targets.

        Args:
            ticker: Stock ticker
            lookback_months: Number of months to look back

        Returns:
            Tuple of (recommendations_df, price_targets_dict)
        """
        print(f"\nCollecting analyst data for {ticker}...")

        # Collect recommendations
        recommendations = self.analyst_collector.collect_recommendations(
            ticker, lookback_months=lookback_months
        )

        if recommendations is not None and not recommendations.empty:
            print(f"  ✅ Collected {len(recommendations)} recommendation periods")
        else:
            print(f"  ⚠️  No analyst recommendations found")

        # Collect price targets
        price_targets = self.analyst_collector.collect_price_targets(ticker)

        if price_targets:
            print(f"  ✅ Collected price targets from {price_targets['num_analysts']} analysts")
        else:
            print(f"  ⚠️  No price targets found")

        return recommendations, price_targets

    def collect_insider_data(
        self,
        ticker: str,
        lookback_months: int = 6
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Collect insider trading transactions and sentiment.

        Args:
            ticker: Stock ticker
            lookback_months: Number of months to look back

        Returns:
            Tuple of (transactions_df, sentiment_dict)
        """
        print(f"\nCollecting insider trading data for {ticker}...")

        # Collect transactions
        transactions = self.insider_collector.collect_insider_transactions(
            ticker, lookback_months=lookback_months
        )

        if transactions is not None and not transactions.empty:
            print(f"  ✅ Collected {len(transactions)} insider transactions")

            # Aggregate sentiment
            sentiment = self.insider_collector.aggregate_insider_sentiment(transactions)

            if sentiment:
                print(f"  ✅ Insider sentiment score: {sentiment['insider_sentiment_score']:.3f}")
        else:
            print(f"  ⚠️  No insider transactions found")
            sentiment = None

        return transactions, sentiment


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    """Test sentiment data collection."""

    print("="*70)
    print("TESTING SENTIMENT DATA COLLECTION")
    print("="*70)

    # Initialize collector (use correct path whether run from root or helpers/)
    import os
    # Check parent directory FIRST to avoid using template in helpers/config/
    if os.path.exists('../config/api_keys.json'):
        api_keys_path = '../config/api_keys.json'
    elif os.path.exists('config/api_keys.json'):
        api_keys_path = 'config/api_keys.json'
    else:
        api_keys_path = 'config/api_keys.json'  # Will create template

    collector = SentimentDataCollector(api_keys_file=api_keys_path)

    # Test with Apple
    ticker = 'AAPL'

    # Test news collection (small sample)
    print("\n1. Testing news collection...")
    news_df = collector.collect_news(ticker, '2024-11-01', '2024-11-05', analyze_sentiment=True)
    if not news_df.empty:
        print(f"\nSample news:")
        print(news_df[['date', 'headline', 'sentiment_score']].head())

    # Test social collection
    print("\n2. Testing social media collection...")
    social_df = collector.collect_social(ticker, lookback_days=7)
    if not social_df.empty:
        print(f"\nSample mentions:")
        print(social_df[['date', 'subreddit', 'title', 'score']].head())

    # Test options collection
    print("\n3. Testing options data collection...")
    options_data = collector.collect_options(ticker)
    if options_data:
        print(f"\nOptions metrics:")
        print(f"  Put/Call Ratio: {options_data['put_call_ratio']:.3f}")
        print(f"  Put/Call OI: {options_data['put_call_ratio_oi']:.3f}")
        print(f"  IV Skew: {options_data['iv_skew']:.3f}")

    # Test analyst data collection
    print("\n4. Testing analyst data collection...")
    analyst_recs, price_targets = collector.collect_analyst_data(ticker, lookback_months=12)
    if analyst_recs is not None and not analyst_recs.empty:
        print(f"\nSample analyst recommendations:")
        print(analyst_recs[['date', 'strongBuy', 'buy', 'hold', 'sell', 'rating_score']].head())
    if price_targets:
        print(f"\nPrice targets:")
        print(f"  Target Mean: ${price_targets['target_mean']:.2f}")
        print(f"  Target High: ${price_targets['target_high']:.2f}")
        print(f"  Target Low: ${price_targets['target_low']:.2f}")
        if price_targets['upside_pct']:
            print(f"  Upside: {price_targets['upside_pct']:.2f}%")

    # Test insider trading collection
    print("\n5. Testing insider trading data collection...")
    insider_trans, insider_sentiment = collector.collect_insider_data(ticker, lookback_months=6)
    if insider_trans is not None and not insider_trans.empty:
        print(f"\nSample insider transactions:")
        print(insider_trans[['date', 'name', 'share', 'transactionPrice', 'is_buy', 'is_sell']].head())
    if insider_sentiment:
        print(f"\nInsider sentiment metrics:")
        print(f"  Buy Transactions: {insider_sentiment['buy_transactions']}")
        print(f"  Sell Transactions: {insider_sentiment['sell_transactions']}")
        print(f"  Buy Ratio: {insider_sentiment['buy_ratio']:.3f}")
        print(f"  Sentiment Score: {insider_sentiment['insider_sentiment_score']:.3f}")

    print("\n" + "="*70)
    print("✅ Testing complete!")
    print("="*70)
