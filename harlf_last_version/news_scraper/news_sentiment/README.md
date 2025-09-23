# Financial News Sentiment Analysis Pipeline

A streamlined system for scraping financial news and analyzing sentiment using FinBERT, designed for integration with trading strategies.

## Overview

This pipeline implements the following workflow:
1. **Asset Definition**: Define financial assets and their search terms
2. **News Scraping**: Scrape Google News for each asset and time period
3. **Content Extraction**: Extract article titles and content (first 500 words)
4. **Sentiment Analysis**: Analyze sentiment using FinBERT model
5. **Score Calculation**: Calculate monthly sentiment scores: (P_positive - P_negative) / N
6. **Data Export**: Generate unified sentiment matrix and visualizations

## Key Features

- **KISS Principle**: Clean, modular, and maintainable code
- **Rate Limiting**: Respects API constraints with intelligent delays
- **Error Handling**: Robust error handling for web scraping
- **FinBERT Integration**: Uses specialized financial sentiment model
- **Visualization**: Clean matplotlib/seaborn charts
- **CSV Output**: Easy integration with other systems

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage

```python
from sentiment_scraper_utils import SentimentScraper, get_asset_definitions

# Initialize scraper
scraper = SentimentScraper(output_dir="results")
scraper.load_finbert()

# Get predefined assets
assets = get_asset_definitions()

# Analyze sentiment for S&P 500 in January 2024
sentiment_score = scraper.scrape_monthly_sentiment(
    asset_name="SP_500",
    terms=assets["SP_500"][:3],  # Use first 3 terms
    year=2024,
    month=1
)

print(f"Sentiment score: {sentiment_score:.4f}")
```

### 2. Run Demo Script

```bash
python demo_sentiment_scraper.py
```

### 3. Full Pipeline (Notebook)

Open and run `sentiment_analysis_pipeline.ipynb` for the complete workflow with visualizations.

## Asset Configuration

The system includes predefined search terms for major financial assets:

- **SP_500**: S&P 500, SPX, S&P 500 Index, etc.
- **NASDAQ**: NASDAQ, NASDAQ Composite, Nasdaq 100, etc.
- **Dow_Jones**: Dow Jones, DJIA, Dow Jones Industrial Average, etc.
- **Gold**: Gold, Gold Price, Gold Market, Precious Metals, etc.
- **Oil**: Oil, Crude Oil, WTI, Brent Crude, etc.

And many more including international indices (CAC 40, FTSE 100, Nikkei 225, etc.)

## Output Files

The system generates:

1. **Monthly CSV files**: Individual sentiment results per month
2. **Unified sentiment matrix**: Complete time series for all assets
3. **Summary statistics**: Mean, std, min, max for each asset
4. **Visualizations**: Time series plots, correlations, distributions

## Sentiment Score Formula

```
Sentiment_Score = (Sum_Positive_Scores - Sum_Negative_Scores) / Total_Articles
```

Where:
- Positive scores come from articles classified as "positive" by FinBERT
- Negative scores come from articles classified as "negative" by FinBERT
- Neutral articles contribute 0 to the calculation

## Rate Limiting

The system includes intelligent rate limiting:
- Random delays between 2-4 seconds between requests
- Respects Google's terms of service
- Handles request failures gracefully

## Error Handling

Robust error handling for:
- Network timeouts and connection errors
- Invalid URLs or missing content
- FinBERT model loading issues
- File I/O operations

## Extending the System

### Add New Assets

```python
def get_custom_assets():
    return {
        "Bitcoin": ["Bitcoin", "BTC", "Bitcoin price", "Cryptocurrency"],
        "Tesla": ["Tesla", "TSLA", "Tesla stock", "Tesla Motors"]
    }
```

### Custom Sentiment Analysis

```python
# Override sentiment analysis method
class CustomScraper(SentimentScraper):
    def analyze_sentiment(self, text):
        # Your custom sentiment logic here
        return label, score
```

## Performance Notes

- Processing ~10 articles per term per month
- FinBERT model requires ~2GB GPU memory (optional)
- Full pipeline (2003-2024, all assets) takes several hours due to rate limiting
- Start with recent years and key assets for testing

## Integration with Trading Strategies

The unified sentiment matrix can be directly used as features in:
- Reinforcement learning trading agents
- Traditional quantitative strategies
- Risk management systems
- Market timing models

## Files Structure

```
├── sentiment_scraper_utils.py      # Core utility functions
├── sentiment_analysis_pipeline.ipynb  # Complete workflow notebook
├── demo_sentiment_scraper.py       # Simple demonstration
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## License

This project follows standard academic/research usage guidelines.
