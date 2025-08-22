# News Scraper - Financial Sentiment Analysis System

A comprehensive financial news sentiment analysis system focused on portfolio-specific sentiment extraction from GDELT Global Knowledge Graph (GKG) data, with extensible architecture for additional news sources.

## Project Overview

This system analyzes sentiment for a portfolio of technology and growth stocks using GDELT V2Tone data, providing monthly sentiment aggregations with confidence scoring and quality filtering. The analysis covers over 10 years of historical data (2015-2025) with sophisticated pattern matching and domain filtering.

## Project Structure

```
news_scraper/
├── README.md                                    # This file
├── v2tone_precomputed_analysis.ipynb           # Main sentiment analysis notebook
├── data/
│   ├── portfolio_holdings.csv                  # Portfolio positions and quantities
│   └── v2tone_built_in_domain_quality_filter.csv # Pre-computed sentiment data
├── sql/
│   └── gdelt_queries.sql                       # BigQuery SQL for GDELT data extraction
├── google_news.py                              # Google News RSS collector (stub)
├── reddit.py                                   # Reddit sentiment collector (stub)
└── yahoo_finance.py                           # Yahoo Finance scraper (stub)
```

## Core Features

### 1. GDELT V2Tone Sentiment Analysis
- **Historical Coverage**: 2015-2025 monthly sentiment data
- **Portfolio Focus**: 21 technology and growth stocks including NVDA, AMD, MSFT, GOOGL
- **Quality Filtering**: Multi-tier domain filtering with 40+ trusted financial sources
- **Confidence Scoring**: Article count-based confidence levels (high/medium/low)
- **Statistical Validation**: Comprehensive V2Tone data validation and cleaning

### 2. Portfolio-Centric Analysis
- **Complete Coverage**: Includes all portfolio assets regardless of data quality
- **Confidence Weighting**: Data quality-based weighting for analysis
- **Benchmark Comparison**: Statistical comparison with QQQ and SPY sentiment
- **Risk Assessment**: Quality flags and confidence scores for investment decisions

### 3. Data Quality Management
- **Coverage Analysis**: Detailed data availability assessment per ticker
- **Missing Data Handling**: Smart interpolation and filling strategies based on data quality
- **Source Validation**: Domain-based quality scoring and filtering
- **Statistical Testing**: Comprehensive statistical analysis with effect size calculations

## Portfolio Assets

The system analyzes sentiment for the following portfolio (quantities in `data/portfolio_holdings.csv`):

**Core Holdings:**
- NVDA (133 shares) - NVIDIA Corporation
- RDDT (200 shares) - Reddit Inc
- SMR (145 shares) - NuScale Power Corporation
- RGTI (675 shares) - Rigetti Computing Inc
- PLTR (40 shares) - Palantir Technologies Inc

**Technology Stocks:**
- MSFT (20 shares) - Microsoft Corporation
- AMD (30 shares) - Advanced Micro Devices Inc
- GOOGL (16 shares) - Alphabet Inc
- MU (50 shares) - Micron Technology Inc
- MRVL (80 shares) - Marvell Technology Inc

**Emerging Tech:**
- AI (315 shares) - C3.ai Inc
- IONQ (108 shares) - IonQ Inc
- QBTS (620 shares) - D-Wave Quantum Inc
- ARBE (3050 shares) - Arbe Robotics Ltd
- APP (16 shares) - AppLovin Corporation
- ASML (6 shares) - ASML Holding NV

## Data Sources and Quality

### Primary Source: GDELT Global Knowledge Graph
- **Data Range**: February 2015 - August 2025
- **Update Frequency**: Monthly aggregations
- **Quality Sources**: 40+ tier-1 financial domains including Reuters, Bloomberg, WSJ
- **Validation**: Multi-step V2Tone validation with word count minimums

### Quality Tier Classification:

**Tier 1 - Institutional Grade:**
- Reuters, Bloomberg, Wall Street Journal, Financial Times

**Tier 2 - Professional Financial Media:**
- CNBC, MarketWatch, Barron's, Forbes, TheStreet

**Tier 3 - Established Financial Platforms:**
- Yahoo Finance, SeekingAlpha, Morningstar, Fool.com

**Tier 4 - Major News with Business Sections:**
- CNN, BBC, New York Times, Washington Post

## Key Analysis Results

### Portfolio vs Benchmarks (Statistical Significance p < 0.0001):
- **Portfolio sentiment significantly higher than QQQ** (Cohen's d = 0.893)
- **Portfolio sentiment significantly higher than SPY** (Cohen's d = 3.263)
- **Moderate correlation with QQQ** (r = 0.473)
- **Weak correlation with SPY** (r = 0.153)

### Top Performers by Sentiment:
1. **AMD**: 0.549 average sentiment (100% coverage)
2. **NVDA**: 0.517 average sentiment (100% coverage)
3. **MRVL**: 0.509 average sentiment (98.4% coverage)
4. **MSFT**: 0.274 average sentiment (100% coverage)

### Data Quality Distribution:
- **10 assets** with >90% data coverage (excellent quality)
- **2 assets** with 80-90% coverage (good quality)
- **9 assets** with <80% coverage (requires careful handling)

## Usage

### 1. Main Analysis Notebook
```bash
jupyter notebook v2tone_precomputed_analysis.ipynb
```

The notebook provides:
- Data quality assessment and coverage analysis
- Sentiment trend visualizations
- Statistical comparison with benchmarks
- Company-specific performance metrics
- Portfolio-weighted sentiment analysis

### 2. GDELT Data Extraction
```sql
-- Use the query in sql/gdelt_queries.sql for fresh data extraction
-- Modify date ranges as needed for different periods
```

### 3. Portfolio Updates
Update `data/portfolio_holdings.csv` with current positions:
```csv
Ticker,Amount
NVDA,133
RDDT,200
...
```

## Technical Implementation

### Sentiment Scoring
- **V2Tone Processing**: GDELT's multi-dimensional sentiment scores
- **FinBERT-style Normalization**: (Positive - Negative) / 100 scale
- **Confidence Levels**: Based on article count thresholds
- **Quality Weighting**: Domain-based source reliability scoring

### Missing Data Strategy
- **High Quality (>95% coverage)**: Minimal interpolation
- **Good Quality (80-95%)**: Linear interpolation for small gaps
- **Medium Quality (50-80%)**: Company median imputation with flags
- **Low Quality (<50%)**: Careful handling with confidence weighting

### Statistical Methods
- **Paired t-tests** for benchmark comparisons
- **Pearson and Spearman correlations** for relationship analysis
- **Cohen's d** for effect size measurement
- **Confidence-weighted analysis** for portfolio decisions

## Future Enhancements

### Planned Implementations:
1. **Google News RSS Integration** - Recent news complement to GDELT
2. **Reddit Sentiment Analysis** - Social media sentiment from r/wallstreetbets
3. **Yahoo Finance Integration** - Earnings and financial report sentiment
4. **Real-time Updates** - Daily sentiment monitoring
5. **Advanced ML Models** - Deep learning sentiment classification

### Extension Points:
- Additional news sources via modular collector pattern
- Custom sentiment models beyond V2Tone
- Real-time alert system for sentiment changes
- Integration with trading signals and portfolio optimization

## Dependencies

```python
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
jupyter>=1.0.0
```

## Data Schema

### Input Data (v2tone_built_in_domain_quality_filter.csv):
```python
{
    "ticker": str,                    # Stock symbol
    "year": int,                      # Year
    "month": int,                     # Month
    "article_count": int,             # Number of articles
    "tone_mean": float,               # Average sentiment tone
    "positive_mean": float,           # Average positive score
    "negative_mean": float,           # Average negative score
    "finbert_sentiment": float,       # Normalized sentiment [-1, 1]
    "confidence_level": str,          # high/medium/low/insufficient
    "sample_urls": list,              # Sample article URLs
    "sources_used": list              # Source domains used
}
```

## License

This project is part of a broader reinforcement learning in finance research initiative. Please ensure compliance with GDELT terms of use for commercial applications.

## Contact

For questions about the analysis methodology, data quality issues, or enhancement requests, please refer to the notebook documentation or create an issue in the project repository.