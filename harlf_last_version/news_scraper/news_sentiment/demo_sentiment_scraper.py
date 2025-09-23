"""
Demo script for sentiment analysis pipeline
Shows basic usage of the sentiment scraping system
"""

from sentiment_scraper_utils import SentimentScraper, get_asset_definitions
import pandas as pd

def run_demo():
    """Run a simple demonstration of the sentiment analysis system"""
    
    print("Financial News Sentiment Analysis Demo")
    print("=" * 50)
    
    # Initialize scraper
    scraper = SentimentScraper(output_dir="demo_results")
    scraper.load_finbert()
    
    # Get asset definitions
    assets = get_asset_definitions()
    
    # Demo with a single asset and recent month
    demo_asset = "SP_500"
    demo_year = 2024
    demo_month = 9  # September
    
    print(f"\nDemo: Analyzing sentiment for {demo_asset} in {demo_month:02d}/{demo_year}")
    print(f"Using terms: {assets[demo_asset][:2]}")  # Use first 2 terms
    
    # Run sentiment analysis
    sentiment_score = scraper.scrape_monthly_sentiment(
        asset_name=demo_asset,
        terms=assets[demo_asset][:2],  # Limit terms for demo
        year=demo_year,
        month=demo_month
    )
    
    print(f"\nResult: Sentiment score = {sentiment_score:.4f}")
    
    # Test individual sentiment analysis
    print("\n" + "="*50)
    print("Individual Sentiment Analysis Test:")
    
    test_sentences = [
        "The S&P 500 soared to new heights as investor confidence reached record levels.",
        "Markets crashed today amid fears of economic recession and banking crisis.",
        "The Federal Reserve maintained interest rates at current levels as expected."
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        label, score = scraper.analyze_sentiment(sentence)
        print(f"\nSentence {i}: {sentence[:60]}...")
        print(f"Sentiment: {label.upper()} (confidence: {score:.3f})")
    
    print("\nDemo completed! Check 'demo_results' directory for output files.")

if __name__ == "__main__":
    run_demo()
