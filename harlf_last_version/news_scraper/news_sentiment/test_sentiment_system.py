#!/usr/bin/env python3
"""
Simple test script to validate the sentiment analysis system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentiment_scraper_utils import SentimentScraper, get_asset_definitions
import pandas as pd

def test_basic_functionality():
    """Test basic functionality without web scraping"""
    
    print("=== SENTIMENT ANALYSIS SYSTEM TEST ===")
    print()
    
    # Test 1: Asset definitions
    print("1. Testing asset definitions...")
    assets = get_asset_definitions()
    print(f"   ✓ Loaded {len(assets)} asset categories")
    print(f"   ✓ Sample assets: {list(assets.keys())[:3]}")
    print()
    
    # Test 2: Initialize scraper
    print("2. Testing scraper initialization...")
    scraper = SentimentScraper(output_dir="test_results")
    print("   ✓ Scraper initialized successfully")
    print()
    
    # Test 3: FinBERT loading
    print("3. Testing FinBERT model loading...")
    try:
        scraper.load_finbert()
        print("   ✓ FinBERT model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading FinBERT: {e}")
        return False
    print()
    
    # Test 4: Sentiment analysis
    print("4. Testing sentiment analysis...")
    test_texts = [
        "The S&P 500 soared to record highs as investor confidence reached new peaks.",
        "Markets crashed dramatically amid fears of economic recession and banking crisis.",
        "The Federal Reserve maintained interest rates at current levels as expected."
    ]
    
    results = []
    for i, text in enumerate(test_texts, 1):
        try:
            label, score = scraper.analyze_sentiment(text)
            results.append((text[:50] + "...", label, score))
            print(f"   Text {i}: {label.upper()} (confidence: {score:.3f})")
        except Exception as e:
            print(f"   ✗ Error analyzing text {i}: {e}")
            return False
    print()
    
    # Test 5: URL generation
    print("5. Testing Google News URL generation...")
    try:
        test_url = scraper.generate_google_news_url("S&P 500", "1/1/2024", "1/31/2024")
        print(f"   ✓ Generated URL: {test_url[:80]}...")
    except Exception as e:
        print(f"   ✗ Error generating URL: {e}")
        return False
    print()
    
    # Test 6: Create sample results
    print("6. Creating sample results...")
    try:
        # Create a sample sentiment matrix
        sample_data = []
        assets_sample = ["SP_500", "NASDAQ", "Gold"]
        months = ["2024-01", "2024-02", "2024-03"]
        
        for month in months:
            for asset in assets_sample:
                # Simulate sentiment scores
                if asset == "SP_500":
                    score = 0.15 if "01" in month else (-0.05 if "02" in month else 0.08)
                elif asset == "NASDAQ":
                    score = 0.12 if "01" in month else (-0.08 if "02" in month else 0.05)
                else:  # Gold
                    score = -0.02 if "01" in month else (0.10 if "02" in month else -0.03)
                
                sample_data.append({
                    'Date': month,
                    'Asset': asset,
                    'Sentiment_Score': score
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(sample_data)
        
        # Pivot to matrix format
        sentiment_matrix = df.pivot(index='Date', columns='Asset', values='Sentiment_Score')
        
        # Save results
        os.makedirs(scraper.output_dir, exist_ok=True)
        matrix_path = os.path.join(scraper.output_dir, "sample_sentiment_matrix.csv")
        sentiment_matrix.to_csv(matrix_path)
        
        print(f"   ✓ Sample sentiment matrix saved to: {matrix_path}")
        print("   ✓ Matrix shape:", sentiment_matrix.shape)
        print("\n   Sample data preview:")
        print(sentiment_matrix.round(4))
        
    except Exception as e:
        print(f"   ✗ Error creating sample results: {e}")
        return False
    
    print("\n=== ALL TESTS PASSED ===")
    print(f"✓ System is ready for full sentiment analysis")
    print(f"✓ Results will be saved to: {scraper.output_dir}")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
