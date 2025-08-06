# HARLF Sentiment Analysis for Portfolio Management

A comprehensive sentiment analysis pipeline for hierarchical reinforcement learning (HARLF) portfolio management, combining real-time news sentiment with classical technical analysis.

## 🎯 Project Overview

This project implements a production-ready sentiment analysis system that integrates multiple data sources (Yahoo Finance, Google News, Reddit) with FinBERT processing to create sentiment features for reinforcement learning portfolio management.

### Key Features

- **Multi-Source Data Collection**: Yahoo Finance, Google News, Reddit APIs
- **Production-Grade FinBERT**: Financial sentiment analysis with fallback mechanisms
- **Intelligent Caching**: 24-hour cache validity with compression
- **HARLF Integration**: Sentiment features for reinforcement learning environments
- **Validation Framework**: Sentiment-return correlation analysis
- **Notebook-First Research**: All analysis in Jupyter notebooks

## 📁 Project Structure

```
harlf_v3/
├── notebooks/                    # Primary research interface
│   ├── 01_data_collection.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   └── 03_harlf_training.ipynb
├── utils/                       # Reusable utility functions
│   ├── data_collectors.py
│   ├── sentiment_processor.py
│   └── feature_engineer.py
├── config/                      # Configuration files
│   └── sentiment_config.py
├── cache/                       # Data caching
│   ├── news_cache/
│   └── sentiment_cache/
├── outputs/                     # Generated results
├── models/                      # Trained models
├── portfolio_holdings.csv       # Portfolio data
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd harlf_v3

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Analysis

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. 01_data_collection.ipynb
# 2. 02_sentiment_analysis.ipynb
# 3. 03_harlf_training.ipynb
```

### 3. Configuration

Edit `config/sentiment_config.py` to customize:
- Data sources and weights
- FinBERT model settings
- HARLF reward function parameters
- Cache settings

## 📊 Data Pipeline

### Phase 1: Data Collection
- **Multi-source collection**: Yahoo Finance, Google News, Reddit
- **Rate limiting**: Respects API limits (0.5s Yahoo, 1s Google/Reddit)
- **Intelligent caching**: 24-hour validity with JSON compression
- **Error handling**: Exponential backoff and graceful degradation

### Phase 2: Sentiment Processing
- **FinBERT integration**: Production-ready with fallback models
- **Batch processing**: 20 articles simultaneously for efficiency
- **Sentiment normalization**: Convert to [-1, +1] range for RL compatibility
- **Source weighting**: Volume-based + quality weighting

### Phase 3: Feature Engineering
- **Temporal alignment**: Consistent monthly aggregation
- **HARLF compatibility**: Technical + sentiment feature fusion
- **Validation framework**: Sentiment-return correlation analysis
- **Missing data handling**: Forward-fill with decay

## 🧠 HARLF Integration

### Environment Structure
```python
class HARLFSentimentEnv(gym.Env):
    """Enhanced HARLF environment with real sentiment data"""
    
    def __init__(self, price_data, technical_features, sentiment_features):
        # Observation: [technical_features, sentiment_features]
        # Action: Portfolio weights (sum to 1)
        # Reward: α1*ROI - α2*MaxDrawdown - α3*Volatility
```

### Agent Architecture
- **Base Agents**: Sentiment-only RL for isolation testing
- **Meta Agents**: Technical + sentiment fusion
- **Super Agent**: Multi-modal signal integration

## 📈 Key Metrics

### Sentiment Analysis
- **Monthly sentiment scores**: Weighted aggregation across sources
- **Sentiment-return correlation**: Current and predictive correlations
- **Statistical significance**: T-test validation
- **Data quality metrics**: Article counts, date ranges, confidence scores

### Portfolio Performance
- **Risk-adjusted returns**: Sharpe ratio, Sortino ratio
- **Drawdown analysis**: Maximum drawdown, recovery periods
- **Volatility metrics**: Rolling volatility, VaR
- **Sentiment divergence**: Early warning system

## 🔧 Configuration

### Data Sources
```python
SENTIMENT_CONFIG = {
    'sources': {
        'yahoo': {'enabled': True, 'rate_limit': 0.5, 'weight': 0.4},
        'google': {'enabled': True, 'rate_limit': 1.0, 'weight': 0.4},
        'reddit': {'enabled': True, 'rate_limit': 1.0, 'weight': 0.2}
    }
}
```

### FinBERT Settings
```python
'finbert': {
    'model': 'ProsusAI/finbert',
    'fallback_model': 'yiyanghkust/finbert-tone',
    'batch_size': 20,
    'confidence_threshold': 0.5
}
```

### HARLF Parameters
```python
'harlf': {
    'alpha1': 1.0,  # ROI weight
    'alpha2': 2.0,  # Max Drawdown penalty
    'alpha3': 0.5   # Volatility penalty
}
```

## 📋 Portfolio Holdings

Current portfolio includes 18 tech stocks:
- **AI/ML**: NVDA, AMD, AI, IONQ
- **Semiconductors**: MU, MRVL, ASML
- **Software**: MSFT, GOOGL, RDDT
- **Clean Energy**: SMR, PLUG
- **Biotech**: VERU, INGM
- **Others**: AEM, CHYM, RGTI, ARBE

## 🎨 Visualization Examples

### Sentiment Analysis
- Time series plots of sentiment over time
- Correlation heatmaps with returns
- Distribution plots of sentiment scores
- Source-weighted sentiment aggregation

### Portfolio Performance
- Cumulative returns with sentiment overlay
- Drawdown analysis with sentiment signals
- Risk-return scatter plots
- Feature importance analysis

## 🔍 Validation Framework

### Sentiment-Return Correlation
- **Current correlation**: Same-month sentiment vs returns
- **Predictive correlation**: Sentiment vs next-month returns
- **Statistical significance**: T-test validation
- **Robustness checks**: Cross-validation, bootstrap

### Data Quality
- **Article counts**: Per source, per ticker
- **Date coverage**: Missing data analysis
- **Confidence scores**: FinBERT confidence distribution
- **Source reliability**: Weighted aggregation validation

## 🚨 Error Handling

### API Failures
- Exponential backoff retry mechanism
- Graceful degradation when sources fail
- Cache fallback for offline operation
- Comprehensive error logging

### Data Quality Issues
- Missing data handling with forward-fill
- Anomaly detection for unusual sentiment patterns
- Confidence threshold filtering
- Temporal alignment validation

## 📚 Research Methodology

### KISS Principle
- **Keep It Simple**: Clear, readable code over clever solutions
- **Modular design**: Small, testable functions
- **Standard libraries**: pandas, numpy, scikit-learn
- **Visual insights**: Charts over verbose text output

### Notebook-First Approach
- **Primary research interface**: All analysis in Jupyter notebooks
- **Self-contained workflows**: Each notebook is complete
- **Visual documentation**: Charts and explanations in markdown
- **Reproducible results**: Saved seeds and configurations

## 🔮 Future Enhancements

### Planned Features
- **Real-time processing**: Live sentiment monitoring
- **Advanced NLP**: Entity recognition, topic modeling
- **Ensemble methods**: Multiple sentiment models
- **Alternative data**: Earnings calls, SEC filings
- **International markets**: Multi-currency support

### Performance Optimization
- **Parallel processing**: Multi-core sentiment analysis
- **Distributed caching**: Redis integration
- **GPU acceleration**: CUDA-optimized FinBERT
- **Streaming data**: Real-time feature updates

## 🤝 Contributing

1. **Follow KISS principle**: Write clear, simple code
2. **Notebook-first**: Add analysis in Jupyter notebooks
3. **Document changes**: Update README and docstrings
4. **Test thoroughly**: Validate sentiment-return correlations
5. **Visual insights**: Include charts and explanations

## 📄 License

This project is for research purposes. Please ensure compliance with:
- API terms of service for data sources
- Model licensing for FinBERT
- Data attribution requirements
- Risk disclosure statements

## 📞 Support

For questions or issues:
1. Check the notebooks for examples
2. Review configuration settings
3. Validate data quality metrics
4. Test with sample data first

---

**Note**: This is a research project. Past performance does not guarantee future results. Always validate sentiment signals before trading decisions. 