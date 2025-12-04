import pandas as pd
import numpy as np
import praw
from datetime import datetime, timedelta
import time
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# IPO DATES DICTIONARY
# =============================================================================

IPO_DATES = {
    'aem':  '1978-10',     # Agnico Eagle Mines: Oct 1978 (Toronto; US ADR since 1996)
    'rddt': '2024-03',     # Reddit IPO: March 2024
    'nvda': '1999-01',     # NVIDIA IPO: Jan 1999
    'smr':  '2022-05',     # NuScale Power (SMR): May 2022
    'rgti': '2022-03',     # Rigetti Computing: March 2022 (SPAC close)
    'mu':   '1984-06',     # Micron IPO: June 1984
    'amd':  '1979-09',     # AMD IPO: September 1979
    'app':  '2021-04',     # Applovin IPO: April 2021
    'mrvl': '2000-06',     # Marvell IPO: June 2000
    'msft': '1986-03',     # Microsoft IPO: March 1986
    'qbts': '2022-08',     # D-Wave (QBTS): August 2022
    'pltr': '2020-10',     # Palantir: October 2020
    'asml': '1995-03',     # ASML: March 1995
    'goog': '2004-08',     # Google IPO: August 2004
    'ionq': '2021-10',     # IonQ (SPAC): October 2021
    'ai':   '2020-12',     # C3.ai IPO: December 2020
    'arbe': '2021-10',     # Arbe Robotics: October 2021
    'veru': '1999-04',     # Veru Inc.: April 1999 (as "Urologix", later renamed, but tradable)
    'qqq':  '1999-03',     # QQQ ETF: March 1999
}

# =============================================================================
# CORRELATION ANALYSIS FUNCTIONS
# =============================================================================

def comprehensive_correlation_analysis(merged_df, sentiment_features, target_col='log_return', 
                                     ticker_col='ticker', min_observations=10, 
                                     significance_level=0.05, verbose=True):
    """
    Perform comprehensive correlation analysis between sentiment features and target variable.
    
    Args:
        merged_df (pd.DataFrame): DataFrame with sentiment and target data
        sentiment_features (list): List of sentiment column names to analyze
        target_col (str): Name of target variable column (default: 'log_return')
        ticker_col (str): Name of ticker/identifier column (default: 'ticker')
        min_observations (int): Minimum observations required per ticker (default: 10)
        significance_level (float): P-value threshold for significance (default: 0.05)
        verbose (bool): Print detailed results (default: True)
    
    Returns:
        dict: Dictionary containing:
            - portfolio_correlations: Portfolio-level correlation results
            - ticker_correlations: Individual ticker correlation results
            - summary: Summary statistics and validation results
            - correlation_variables: Stored correlation values for visualization
    """
    
    # Clean data for analysis
    required_cols = sentiment_features + [target_col]
    clean_data = merged_df.dropna(subset=required_cols).copy()
    
    if verbose:
        retention_pct = len(clean_data) / len(merged_df) * 100
        print(f"Analysis dataset: {len(clean_data):,} complete observations ({retention_pct:.1f}% retention)")
    
    # =============================================================================
    # PORTFOLIO-LEVEL CORRELATIONS
    # =============================================================================
    
    portfolio_correlations = []
    correlation_vars = {}
    
    if verbose:
        print(f"\nPortfolio-Level Correlations:")
    
    for feature in sentiment_features:
        # Pearson correlation
        corr_p, p_val_p = pearsonr(clean_data[feature], clean_data[target_col])
        # Spearman correlation (more robust)
        corr_s, p_val_s = spearmanr(clean_data[feature], clean_data[target_col])
        
        portfolio_correlations.append({
            'Feature': feature,
            'Pearson_Corr': corr_p,
            'Pearson_P': p_val_p,
            'Spearman_Corr': corr_s,
            'Spearman_P': p_val_s,
            'R_Squared': corr_p**2,
            'Significant': p_val_p < significance_level
        })
        
        # Store for visualization
        correlation_vars[f'{feature}_corr'] = corr_p
        correlation_vars[f'{feature}_p'] = p_val_p
        
        if verbose:
            sig_p = "***" if p_val_p < significance_level else ""
            print(f"{feature.upper()}: r={corr_p:.4f} (p={p_val_p:.4f}) {sig_p}")
    
    # =============================================================================
    # INDIVIDUAL TICKER ANALYSIS
    # =============================================================================
    
    ticker_results = {}
    
    if verbose:
        print(f"\nIndividual Ticker Analysis:")
    
    for feature in sentiment_features:
        ticker_correlations = []
        significant_count = 0
        
        for ticker in merged_df[ticker_col].unique():
            ticker_data = merged_df[merged_df[ticker_col] == ticker].dropna(subset=required_cols)
            
            if len(ticker_data) >= min_observations:
                # Use Spearman for individual tickers (more robust)
                corr, p_val = spearmanr(ticker_data[feature], ticker_data[target_col])
                ticker_correlations.append({
                    'ticker': ticker, 
                    'correlation': corr, 
                    'p_value': p_val,
                    'n_observations': len(ticker_data),
                    'significant': p_val < significance_level
                })
                if p_val < significance_level:
                    significant_count += 1
        
        ticker_df = pd.DataFrame(ticker_correlations)
        ticker_results[feature] = ticker_df
        
        if len(ticker_df) > 0 and verbose:
            print(f"{feature.upper()}: {significant_count}/{len(ticker_df)} tickers significant")
            
            # Show top 3 strongest correlations
            top_3 = ticker_df.nlargest(3, 'correlation')
            strongest = [f"{row['ticker']}({row['correlation']:.3f})" for _, row in top_3.iterrows()]
            print(f"  Strongest: {', '.join(strongest)}")
    
    # =============================================================================
    # SUMMARY AND VALIDATION
    # =============================================================================
    
    significant_portfolio = sum(1 for x in portfolio_correlations if x['Significant'])
    strongest_feature = max(portfolio_correlations, key=lambda x: abs(x['Pearson_Corr']))
    
    # Calculate average individual ticker significance rates
    avg_ticker_significance = {}
    for feature in sentiment_features:
        if feature in ticker_results and len(ticker_results[feature]) > 0:
            sig_rate = ticker_results[feature]['significant'].mean() * 100
            avg_ticker_significance[feature] = sig_rate
    
    summary = {
        'total_observations': len(clean_data),
        'retention_rate': len(clean_data) / len(merged_df),
        'significant_portfolio_features': significant_portfolio,
        'total_features_tested': len(sentiment_features),
        'strongest_feature': strongest_feature['Feature'],
        'strongest_correlation': strongest_feature['Pearson_Corr'],
        'strongest_r_squared': strongest_feature['R_Squared'],
        'avg_ticker_significance_rates': avg_ticker_significance,
        'portfolio_significance_rate': significant_portfolio / len(sentiment_features)
    }
    
    if verbose:
        print(f"\nValidation Results:")
        print(f"Significant measures: {significant_portfolio}/{len(sentiment_features)} portfolio-level")
        print(f"Strongest correlation: {strongest_feature['Feature']} (r={strongest_feature['Pearson_Corr']:.4f})")
        print(f"Effect size: {strongest_feature['R_Squared']:.1%} variance explained")
        
        if avg_ticker_significance:
            print(f"\nIndividual Ticker Significance Rates:")
            for feature, rate in avg_ticker_significance.items():
                print(f"  {feature}: {rate:.1f}% of tickers significant")
    
    return {
        'portfolio_correlations': portfolio_correlations,
        'ticker_correlations': ticker_results,
        'summary': summary,
        'correlation_variables': correlation_vars,
        'clean_data': clean_data
    }

# =============================================================================
# REDDIT SENTIMENT ANALYSIS PIPELINE
# =============================================================================

# Configuration
PORTFOLIO_TICKERS = [
    'AEM', 'AI', 'AMD', 'APP', 'ARBE', 'ASML', 'GOOG', 'IONQ', 'MRVL', 'MSFT', 
    'MU', 'NVDA', 'PLTR', 'QBTS', 'QQQ', 'RDDT', 'RGTI', 'SMR', 'VERU'
]

START_DATE = datetime(2015, 1, 1)
END_DATE = datetime.now()

COMPANY_NAMES = {
    'NVDA': ['nvidia'], 'AMD': ['amd'], 'MSFT': ['microsoft'],
    'GOOG': ['google', 'alphabet'], 'AI': ['c3.ai'], 'ASML': ['asml'],
    'MU': ['micron'], 'MRVL': ['marvell'], 'IONQ': ['ionq'],
    'RGTI': ['rigetti'], 'QBTS': ['d-wave'], 'PLTR': ['palantir'],
    'AEM': ['agnico eagle'], 'VERU': ['veru'], 'APP': ['applovin'],
    'ARBE': ['arbe'], 'RDDT': ['reddit'], 'SMR': ['nuscale'],
    'QQQ': ['qqq', 'nasdaq']
}

SUBREDDIT_TIERS = {
    'tier1': ['SecurityAnalysis', 'investing', 'ValueInvesting'],
    'tier2': ['stocks', 'StockMarket', 'investing'],
    'tier3': ['wallstreetbets', 'pennystocks', 'SecurityAnalysis']
}

MIN_POST_SCORE = 3
POSTS_PER_TICKER = 200

def setup_reddit_api():
    """Initialize Reddit API connection"""
    try:
        reddit = praw.Reddit(
            client_id="pLqfk1M1ymfj3ih1NrVFlA",
            client_secret="_hl1434FeTi9kgv_GXAi5tBLoCaLIQ",
            user_agent="FinancialSentimentAnalysis/3.0"
        )
        print("✅ Reddit API connected")
        return reddit
    except Exception as e:
        print(f"❌ Reddit API failed: {e}")
        return None

def setup_finbert():
    """Initialize FinBERT sentiment analyzer"""
    try:
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        print("✅ FinBERT loaded")
        return finbert_pipeline
    except Exception as e:
        print(f"❌ FinBERT failed: {e}")
        return None

def analyze_sentiment_enhanced(text, finbert_analyzer):
    """Enhanced sentiment analysis using standard FinBERT scoring"""
    if not text or len(text.strip()) < 10:
        return 0.0, 0.33, 0.33, 0.33
    
    try:
        text = text[:512]
        result = finbert_analyzer(text)[0]
        label = result['label'].lower()
        score = result['score']
        
        # Standard FinBERT scoring without amplification
        if label == 'positive':
            tone = score
            pos = score
            neu = (1 - score) / 2
            neg = (1 - score) / 2
        elif label == 'negative':
            tone = -score
            pos = (1 - score) / 2
            neu = (1 - score) / 2
            neg = score
        else:  # neutral
            tone = 0.0
            pos = (1 - score) / 2
            neu = score
            neg = (1 - score) / 2
        
        return tone, pos, neu, neg
    except Exception as e:
        return 0.0, 0.33, 0.33, 0.33

def is_ticker_mentioned(text, ticker):
    """Check if ticker is mentioned in text"""
    if not text:
        return False
    
    text_lower = text.lower()
    ticker_lower = ticker.lower()
    
    if f"${ticker_lower}" in text_lower:
        return True
    
    if ticker in COMPANY_NAMES:
        for name in COMPANY_NAMES[ticker]:
            if name in text_lower:
                return True
    
    return False

def get_search_subreddits(ticker):
    """Get comprehensive subreddit list for ticker with multiple strategies"""
    search_order = []
    
    # Strategy 1: Direct ticker subreddit (r/AMD, r/NVDA, etc.)
    search_order.append(ticker.lower())
    
    # Strategy 2: Company-specific subreddits
    if ticker in COMPANY_NAMES:
        for name in COMPANY_NAMES[ticker]:
            search_order.append(name)
    
    # Strategy 3: Alternative ticker formats
    search_order.append(ticker.upper())
    search_order.append(f"${ticker.lower()}")
    
    # Strategy 4: General financial subreddits (tiered by quality)
    for tier in ['tier1', 'tier2', 'tier3']:
        search_order.extend(SUBREDDIT_TIERS[tier])
    
    # Strategy 5: Additional relevant subreddits
    additional_subs = [
        'investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting',
        'StockMarket', 'wallstreetbets', 'pennystocks', 'dividends',
        'options', 'finance', 'economics', 'business', 'investments', 'trading',
        'financialindependence', 'personalfinance'
    ]
    search_order.extend(additional_subs)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_subs = []
    for sub in search_order:
        if sub not in seen:
            seen.add(sub)
            unique_subs.append(sub)
    
    return unique_subs

def collect_reddit_posts(reddit, ticker, finbert_analyzer):
    """Collect and analyze Reddit posts for a specific ticker"""
    posts_data = []
    subreddits = get_search_subreddits(ticker)
    
    print(f"  Collecting posts for {ticker} from {len(subreddits)} subreddits...")
    
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            # Try multiple search strategies
            search_queries = [
                f"${ticker.lower()}",  # $amd
                f"${ticker.upper()}",  # $AMD
                ticker.lower(),        # amd
                ticker.upper(),        # AMD
                f"{ticker} stock", f"{ticker} earnings", f"{ticker} news"
            ]
            
            # Add company name searches
            if ticker in COMPANY_NAMES:
                for name in COMPANY_NAMES[ticker]:
                    search_queries.append(name)
            
            subreddit_posts = []
            for query in search_queries:
                try:
                    # Try different time filters to get maximum historical data
                    for time_filter in ['all', 'year', 'month']:
                        try:
                            posts = subreddit.search(query, limit=100, sort='relevance', time_filter=time_filter)
                            subreddit_posts.extend(list(posts))
                            break  # If successful, don't try other time filters
                        except Exception as time_error:
                            continue
                except Exception as query_error:
                    continue
            
            # Process posts from this subreddit
            for post in subreddit_posts:
                if len(posts_data) >= POSTS_PER_TICKER:
                    break
                
                # Skip if post is too old (before 2015) or too new (future)
                post_date = datetime.utcfromtimestamp(post.created_utc)
                if post_date < START_DATE or post_date > END_DATE:
                    continue
                
                if getattr(post, 'score', 0) < MIN_POST_SCORE:
                    continue
                
                full_text = f"{post.title} {getattr(post, 'selftext', '')}"
                
                if not is_ticker_mentioned(full_text, ticker):
                    continue
                
                tone, pos, neu, neg = analyze_sentiment_enhanced(full_text, finbert_analyzer)
                
                posts_data.append({
                    'post_id': post.id,
                    'ticker': ticker,
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'full_text': full_text[:800],
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'post_date': post_date,
                    'sentiment_compound': tone,
                    'sentiment_positive': pos,
                    'sentiment_neutral': neu,
                    'sentiment_negative': neg
                })
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"    Error with subreddit {subreddit_name}: {e}")
            continue
    
    print(f"    Collected {len(posts_data)} posts for {ticker}")
    return posts_data

def monthly_aggregate(df_raw):
    """Aggregate raw posts into monthly sentiment data"""
    if df_raw.empty:
        return pd.DataFrame()
    
    df = df_raw.copy()
    df['year'] = df['post_date'].dt.year
    df['month'] = df['post_date'].dt.month
    df['year_month'] = df['post_date'].dt.strftime('%Y-%m')
    df['engagement'] = df['score'] * (df['num_comments'] + 1)
    df['weight'] = np.maximum(df['engagement'], 1.0)
    
    monthly_data = []
    
    for (ticker, year, month), group in df.groupby(['ticker', 'year', 'month']):
        article_count = len(group)
        mean_tone = group['sentiment_compound'].mean()
        mean_positive = group['sentiment_positive'].mean()
        mean_negative = group['sentiment_negative'].mean()
        
        weights = group['weight']
        weighted_tone = np.average(group['sentiment_compound'], weights=weights)
        weighted_positive = np.average(group['sentiment_positive'], weights=weights)
        weighted_negative = np.average(group['sentiment_negative'], weights=weights)
        
        avg_score = group['score'].mean()
        total_engagement = group['engagement'].sum()
        avg_engagement = group['engagement'].mean()
        subreddits_used = group['subreddit'].nunique()
        
        monthly_data.append({
            'ticker': ticker,
            'year': year,
            'month': month,
            'year_month': f"{year}-{month:02d}",
            'article_count': article_count,
            'mean_tone': mean_tone,
            'mean_positive': mean_positive,
            'mean_negative': mean_negative,
            'weighted_tone': weighted_tone,
            'weighted_positive': weighted_positive,
            'weighted_negative': weighted_negative,
            'avg_score': avg_score,
            'total_engagement': total_engagement,
            'avg_engagement': avg_engagement,
            'subreddits_used': subreddits_used
        })
    
    return pd.DataFrame(monthly_data)

def run_reddit_pipeline():
    """Main pipeline execution function"""
    print("Reddit Sentiment Analysis Pipeline")
    print("=" * 50)
    
    reddit = setup_reddit_api()
    finbert_analyzer = setup_finbert()
    
    if reddit is None or finbert_analyzer is None:
        print("❌ Pipeline initialization failed")
        return None, None, None
    
    print(f"\nCollecting Reddit data for {len(PORTFOLIO_TICKERS)} tickers...")
    print("=" * 50)
    
    all_posts = []
    ticker_summary = []
    
    for i, ticker in enumerate(PORTFOLIO_TICKERS, 1):
        print(f"\n[{i}/{len(PORTFOLIO_TICKERS)}] Processing {ticker}...")
        
        try:
            posts_data = collect_reddit_posts(reddit, ticker, finbert_analyzer)
            all_posts.extend(posts_data)
            
            ticker_summary.append({
                'ticker': ticker,
                'posts_collected': len(posts_data),
                'subreddits_used': len(set([p['subreddit'] for p in posts_data])),
                'avg_sentiment': np.mean([p['sentiment_compound'] for p in posts_data]) if posts_data else 0
            })
            
        except Exception as e:
            print(f"    ❌ Error processing {ticker}: {e}")
            continue
    
    print(f"\n✅ Data collection complete: {len(all_posts)} total posts collected")
    
    if all_posts:
        df_raw = pd.DataFrame(all_posts)
        df_raw['post_date'] = pd.to_datetime(df_raw['post_date'])
        df_monthly = monthly_aggregate(df_raw)
        
        print(f"\nProcessing complete:")
        print(f"  Raw posts: {len(df_raw):,}")
        print(f"  Monthly observations: {len(df_monthly):,}")
        print(f"  Tickers covered: {df_monthly['ticker'].nunique()}")
        print(f"  Date range: {df_monthly['year_month'].min()} to {df_monthly['year_month'].max()}")
        
        # Export data
        os.makedirs('data', exist_ok=True)
        df_raw.to_csv('data/reddit_sentiment_raw.csv', index=False)
        df_monthly.to_csv('data/reddit_sentiment_monthly.csv', index=False)
        
        summary_df = pd.DataFrame(ticker_summary)
        summary_df.to_csv('data/reddit_sentiment_summary.csv', index=False)
        
        print(f"\n✅ Files exported:")
        print(f"  Raw data: data/reddit_sentiment_raw.csv ({len(df_raw):,} posts)")
        print(f"  Monthly data: data/reddit_sentiment_monthly.csv ({len(df_monthly):,} observations)")
        print(f"  Summary: data/reddit_sentiment_summary.csv ({len(ticker_summary)} tickers)")
        
        # Data quality analysis
        high_quality = (df_monthly['article_count'] >= 10).sum()
        medium_quality = ((df_monthly['article_count'] >= 5) & (df_monthly['article_count'] < 10)).sum()
        low_quality = (df_monthly['article_count'] < 5).sum()
        
        print(f"\nData Quality Analysis:")
        print(f"  High quality months (≥10 posts): {high_quality} ({high_quality/len(df_monthly)*100:.1f}%)")
        print(f"  Medium quality months (5-9 posts): {medium_quality} ({medium_quality/len(df_monthly)*100:.1f}%)")
        print(f"  Low quality months (<5 posts): {low_quality} ({low_quality/len(df_monthly)*100:.1f}%)")
        
        return df_raw, df_monthly, summary_df
    else:
        print("❌ No data collected")
        return None, None, None


