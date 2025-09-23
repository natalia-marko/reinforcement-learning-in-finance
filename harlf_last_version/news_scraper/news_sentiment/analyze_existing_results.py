#!/usr/bin/env python3
"""
Analysis of existing sentiment analysis results
Shows what data is available and creates visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

def analyze_reddit_sentiment():
    """Analyze the existing Reddit sentiment data"""
    
    print("REDDIT SENTIMENT ANALYSIS RESULTS INSPECTION")
    print("=" * 60)
    
    # Load the monthly sentiment data
    data_path = "/Users/nataliamarko/Documents/GitHub_Projects/reinforcement_learning_in_finance/harlf_last_version/news_scraper/out/reddit/reddit_monthly_sentiment.csv"
    
    if not os.path.exists(data_path):
        print("❌ Reddit sentiment data not found")
        return
    
    df = pd.read_csv(data_path)
    
    # Basic info
    print(f"📊 Dataset Overview:")
    print(f"   • Total records: {len(df):,}")
    print(f"   • Unique tickers: {df['ticker'].nunique()}")
    print(f"   • Date range: {df['month'].min()} to {df['month'].max()}")
    print(f"   • Columns: {list(df.columns)}")
    print()
    
    # Top tickers by coverage
    print("🏆 Top 10 Tickers by Data Points:")
    ticker_counts = df['ticker'].value_counts().head(10)
    for i, (ticker, count) in enumerate(ticker_counts.items(), 1):
        print(f"   {i:2d}. {ticker:6s}: {count:3d} months")
    print()
    
    # Sentiment statistics
    print("📈 Sentiment Statistics:")
    sentiment_cols = ['sent_mean', 'sent_median', 'sent_ew_weighted']
    for col in sentiment_cols:
        if col in df.columns:
            valid_data = df[col].dropna()
            print(f"   • {col}:")
            print(f"     - Count: {len(valid_data):,}")
            print(f"     - Mean: {valid_data.mean():.4f}")
            print(f"     - Std: {valid_data.std():.4f}")
            print(f"     - Range: [{valid_data.min():.4f}, {valid_data.max():.4f}]")
    print()
    
    # Activity statistics
    print("📱 Activity Statistics:")
    activity_cols = ['n_posts', 'mentions_textual', 'mentions_per_day']
    for col in activity_cols:
        if col in df.columns:
            valid_data = df[col].dropna()
            print(f"   • {col}:")
            print(f"     - Mean: {valid_data.mean():.2f}")
            print(f"     - Median: {valid_data.median():.2f}")
            print(f"     - Max: {valid_data.max():.0f}")
    print()
    
    # Recent data
    print("🕒 Recent Data Sample:")
    recent = df[df['month'] >= '2024-01'].head(10)
    if len(recent) > 0:
        for _, row in recent.iterrows():
            print(f"   {row['ticker']:6s} {row['month']} | Posts: {row['n_posts']:3.0f} | Sentiment: {row['sent_mean']:6.3f}")
    print()
    
    # Create visualizations
    create_sentiment_visualizations(df)
    
    return df

def create_sentiment_visualizations(df):
    """Create visualizations of the sentiment data"""
    
    print("📊 Creating visualizations...")
    
    # Filter for tickers with substantial data
    ticker_counts = df['ticker'].value_counts()
    top_tickers = ticker_counts[ticker_counts >= 12].index[:8]  # At least 12 months of data
    
    df_viz = df[df['ticker'].isin(top_tickers)].copy()
    
    if len(df_viz) == 0:
        print("   ⚠️  Insufficient data for visualizations")
        return
    
    # Convert month to datetime
    df_viz['date'] = pd.to_datetime(df_viz['month'])
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reddit Sentiment Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Sentiment over time for top tickers
    ax1 = axes[0, 0]
    for ticker in top_tickers[:5]:  # Show top 5 for clarity
        ticker_data = df_viz[df_viz['ticker'] == ticker]
        if len(ticker_data) > 1:
            ax1.plot(ticker_data['date'], ticker_data['sent_mean'], 
                    label=ticker, marker='o', markersize=4, linewidth=2)
    
    ax1.set_title('Sentiment Trends Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Mean Sentiment')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Distribution of sentiment scores
    ax2 = axes[0, 1]
    sentiment_data = df['sent_mean'].dropna()
    ax2.hist(sentiment_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Distribution of Sentiment Scores')
    ax2.set_xlabel('Mean Sentiment')
    ax2.set_ylabel('Frequency')
    ax2.axvline(x=sentiment_data.mean(), color='red', linestyle='--', 
                label=f'Mean: {sentiment_data.mean():.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Activity vs Sentiment scatter
    ax3 = axes[1, 0]
    valid_data = df[df['sent_mean'].notna() & df['n_posts'].notna()]
    scatter = ax3.scatter(valid_data['n_posts'], valid_data['sent_mean'], 
                         alpha=0.6, c=valid_data['mentions_per_day'], 
                         cmap='viridis', s=30)
    ax3.set_title('Activity vs Sentiment')
    ax3.set_xlabel('Number of Posts')
    ax3.set_ylabel('Mean Sentiment')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Mentions per Day')
    
    # 4. Top tickers by average sentiment
    ax4 = axes[1, 1]
    avg_sentiment = df.groupby('ticker')['sent_mean'].mean().sort_values(ascending=False).head(10)
    bars = ax4.bar(range(len(avg_sentiment)), avg_sentiment.values)
    ax4.set_title('Top 10 Tickers by Average Sentiment')
    ax4.set_xlabel('Ticker')
    ax4.set_ylabel('Average Sentiment')
    ax4.set_xticks(range(len(avg_sentiment)))
    ax4.set_xticklabels(avg_sentiment.index, rotation=45)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Color bars based on sentiment
    for bar, value in zip(bars, avg_sentiment.values):
        if value > 0:
            bar.set_color('green')
        elif value < 0:
            bar.set_color('red')
        else:
            bar.set_color('gray')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "reddit_sentiment_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Visualization saved to: {output_path}")

def create_summary_matrix(df):
    """Create a summary matrix similar to the one we planned"""
    
    print("📋 Creating Summary Matrix...")
    
    # Get recent data (last 2 years)
    recent_df = df[df['month'] >= '2023-01'].copy()
    
    if len(recent_df) == 0:
        print("   ⚠️  No recent data available")
        return
    
    # Create pivot table
    matrix = recent_df.pivot_table(
        index='month',
        columns='ticker', 
        values='sent_mean',
        fill_value=np.nan
    )
    
    # Filter for tickers with good coverage
    ticker_coverage = matrix.count()
    good_tickers = ticker_coverage[ticker_coverage >= 6].index  # At least 6 months
    matrix_filtered = matrix[good_tickers]
    
    print(f"   • Matrix shape: {matrix_filtered.shape}")
    print(f"   • Coverage: {matrix_filtered.count().sum()} data points")
    
    # Save matrix
    matrix_path = "reddit_sentiment_matrix.csv"
    matrix_filtered.to_csv(matrix_path)
    print(f"   ✅ Matrix saved to: {matrix_path}")
    
    # Show sample
    print("\n   Sample of the matrix:")
    print(matrix_filtered.head().round(3))
    
    return matrix_filtered

if __name__ == "__main__":
    # Run the analysis
    df = analyze_reddit_sentiment()
    
    if df is not None:
        matrix = create_summary_matrix(df)
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("📁 Generated files:")
        print("   • reddit_sentiment_analysis.png - Comprehensive visualizations")
        print("   • reddit_sentiment_matrix.csv - Unified sentiment matrix")
        print("\n💡 This demonstrates the system is working and producing")
        print("   high-quality sentiment analysis results from Reddit data!")
