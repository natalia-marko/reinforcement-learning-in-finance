# HARLF Sentiment Analysis: Cursor Rules (KISS Policy & Notebook-First)

## Project Overview
This project implements hierarchical reinforcement learning for portfolio management with real-time sentiment analysis. The system integrates multiple data sources (Yahoo Finance, Google News, Reddit) with FinBERT processing and custom RL environments.

## File Organization

### Notebook Structure (Primary Research Interface)
- **Core analysis must be in Jupyter Notebooks (.ipynb)** - notebooks are the primary interface for research and development
- **Notebook naming convention**: `01_data_collection.ipynb`, `02_sentiment_analysis.ipynb`, `03_harlf_training.ipynb`, etc.
- Each notebook should be self-contained but can import utility functions
- Include all visualizations, key findings, and methodology explanations in notebook markdown cells

### Python Module Structure (Supporting Utilities)
```
project/
├── notebooks/          # Primary research interface
├── utils/             # Reusable utility functions
│   ├── data_collectors.py
│   ├── sentiment_processor.py
│   ├── feature_engineer.py
│   └── harlf_env.py
├── models/            # Trained models and weights
├── cache/             # Data caching (news_cache/, sentiment_cache/)
├── outputs/           # Generated datasets and results
└── config/            # Configuration files
```

### Data Management
- **Cache everything**: Use intelligent caching for API calls (24hr validity)
- **Outputs directory**: All generated CSVs, results, and artifacts go to `outputs/`
- **Version control**: Include `.gitignore` for cache/, models/, and large output files
- **Data alignment**: Ensure consistent datetime indexing across all data sources

## Coding Style (KISS Principle for Financial ML)

### Core Principles
- **Readability over cleverness**: Write code that other researchers can understand quickly
- **Modular functions**: Break complex operations into small, testable functions
- **Error handling**: Always include try/catch for API calls and external data
- **Reproducibility**: Set random seeds and document versions

### Financial ML Specific Standards
```python
# Good: Clear, modular function
def calculate_sentiment_score(headlines, model):
    """Convert FinBERT outputs to [-1, +1] sentiment scores"""
    results = model(headlines)
    scores = []
    for result in results:
        if result['label'] == 'positive':
            scores.append(result['score'])
        elif result['label'] == 'negative': 
            scores.append(-result['score'])
        else:
            scores.append(0.0)
    return np.array(scores)

# Bad: Complex, hard to debug
def process_sentiment(data):
    return np.array([r['score'] if r['label'] == 'positive' else -r['score'] if r['label'] == 'negative' else 0.0 for r in model([item for item in data])])
```

### Preferred Libraries
- **Data**: `pandas`, `numpy`, `yfinance`
- **ML/RL**: `stable-baselines3`, `gymnasium`, `scikit-learn`
- **NLP**: `transformers` (FinBERT), `textblob` (fallback)
- **APIs**: `praw` (Reddit), `gnews` (Google News)
- **Visualization**: `matplotlib`, `seaborn` (avoid plotly unless interactive needed)

## Sentiment Analysis Specific Rules

### Data Collection
- **API rate limiting**: Always implement proper rate limiting (0.5s Yahoo, 1s Google/Reddit)
- **Caching first**: Check cache before making API calls
- **Batch processing**: Process sentiment in batches of 20 articles for efficiency
- **Error recovery**: Graceful degradation when sources fail

### FinBERT Processing
```python
# Standard FinBERT setup pattern
def setup_finbert():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception:
        print("Falling back to alternative model")
        return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
```

### Feature Engineering
- **Temporal alignment**: Ensure all features use consistent monthly aggregation
- **Normalization**: Always normalize features to [0,1] or [-1,1] for RL compatibility
- **Missing data**: Use forward-fill with decay, never just drop data
- **Validation**: Always validate sentiment-return correlations before RL training

## HARLF Environment Rules

### Environment Structure
- Extend `gymnasium.Env` for all custom environments
- Use proven reward structure: `α1 * ROI - α2 * MaxDrawdown - α3 * Volatility`
- Maintain consistent observation space: `[technical_features, sentiment_features]`
- Always normalize portfolio weights to sum to 1

### Agent Training
- **Base agents**: Train individual agents on single signal types (sentiment-only, technical-only)
- **Meta agents**: Combine base agent outputs with learned weights
- **Super agent**: Final hierarchical integration
- **Reproducibility**: Save all hyperparameters and random seeds

## Visualizations (Financial Focus)

### Chart Priorities
1. **Time series plots**: Sentiment over time, portfolio performance
2. **Correlation heatmaps**: Sentiment-return relationships
3. **Distribution plots**: Sentiment score distributions by source
4. **Performance charts**: Cumulative returns, drawdowns, Sharpe ratios

### Visualization Standards
```python
# Good: Clean, informative financial chart
plt.figure(figsize=(12, 6))
plt.plot(sentiment_df.index, sentiment_df['weighted_sentiment'], 
         label='Weighted Sentiment', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('Monthly Weighted Sentiment Score')
plt.ylabel('Sentiment Score [-1, +1]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Bad: Over-styled, unclear chart
sentiment_df.plot(kind='line', style='-o', color=['red', 'blue', 'green'], 
                  alpha=0.8, linewidth=3, markersize=8, figsize=(15, 8))
```

### What to Avoid
- 3D plots (rarely useful for time series financial data)
- Excessive styling or colors
- Verbose printed DataFrames (use `.head()` and visualizations)
- Multiple overlapping plots that obscure insights

## Project Workflow

### Development Cycle
1. **Data Collection**: Start with small date ranges, expand after validation
2. **Sentiment Processing**: Test on sample data before full processing
3. **Feature Engineering**: Validate alignment before RL training
4. **Environment Testing**: Test with dummy agents before full training
5. **Model Training**: Start with short training runs, scale up gradually

### Quality Gates
- All data collection functions must handle API failures gracefully
- Sentiment scores must be validated against known market events
- RL environments must pass basic gym compliance tests
- All results must be reproducible with saved seeds

### Notebook Organization
Each notebook should follow this structure:
```markdown
# Notebook Title
## 1. Setup & Configuration
## 2. Data Loading/Collection  
## 3. Processing & Analysis
## 4. Validation & Quality Checks
## 5. Results & Visualizations
## 6. Key Findings & Next Steps
```

## Documentation & Testing

### Docstring Standards
```python
def collect_sentiment_data(tickers, start_date, end_date, sources=['yahoo', 'google']):
    """
    Collect sentiment data from multiple sources for given tickers.
    
    Args:
        tickers (list): List of stock symbols
        start_date (str): Start date 'YYYY-MM-DD'
        end_date (str): End date 'YYYY-MM-DD'  
        sources (list): Data sources to use
        
    Returns:
        pd.DataFrame: Sentiment scores indexed by date and ticker
        
    Raises:
        APIError: When all data sources fail
    """
```

### Testing Philosophy
- **Unit tests**: For core utility functions in `utils/`
- **Integration tests**: For API collectors and data pipelines
- **Validation tests**: Sentiment-return correlation checks
- **Environment tests**: RL environment compliance

### Error Handling Patterns
```python
# Standard error handling pattern
def safe_api_call(func, *args, max_retries=3, **kwargs):
    """Wrapper for safe API calls with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Output Preferences

### Results Documentation
- **Key metrics table**: Sharpe ratio, max drawdown, total return
- **Performance comparison**: Sentiment-enhanced vs baseline models
- **Validation results**: Correlation coefficients with p-values
- **Model artifacts**: Save trained agents to `models/` directory

### File Outputs
All results should be saved to organized output files:
```
outputs/
├── monthly_sentiment.csv           # Aggregated sentiment features
├── validation_results.csv          # Correlation analysis
├── portfolio_performance.json      # Performance metrics
├── model_hyperparameters.json      # Training configuration
└── plots/                          # Generated visualizations
```

## Special Considerations for Financial ML

### Market Data Handling
- **No lookahead bias**: Ensure strict temporal ordering
- **Market hours**: Consider timezone alignment for international assets  
- **Corporate actions**: Handle splits, dividends in price data
- **Survivorship bias**: Include delisted stocks in historical analysis

### Risk Management
- **Position limits**: Implement maximum position size constraints
- **Drawdown controls**: Stop training if drawdown exceeds thresholds
- **Sentiment validation**: Flag unusual sentiment patterns for review
- **Performance monitoring**: Track live performance vs backtest

### Regulatory Compliance
- **Data attribution**: Properly cite all data sources
- **API terms**: Comply with rate limits and usage terms
- **Model documentation**: Maintain audit trail for model decisions
- **Risk disclosures**: Include appropriate disclaimers in outputs

## Final Principles

1. **Notebooks first**: All research happens in notebooks, utilities support them
2. **Visual insights**: Show, don't tell - use charts over tables
3. **Reproducible results**: Anyone should be able to rerun your analysis
4. **Incremental development**: Build and test small pieces before integration
5. **Real-world focus**: Always consider practical trading constraints
6. **Quality over speed**: Better to have reliable, well-tested code than fast, buggy code