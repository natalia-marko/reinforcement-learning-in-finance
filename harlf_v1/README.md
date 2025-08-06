# Hierarchical Reinforcement Learning Framework (HARLF) for Asset Investing

## Project Overview

This project implements a hierarchical reinforcement learning framework for asset allocation, inspired by the HARLF approach. The system combines market data analysis with sentiment analysis to make intelligent portfolio decisions.

## Project Structure

```
reinforcement_learning_in_finance/
├── 01_data_collection.ipynb          # Milestone 1: Market data collection
├── 02_sentiment_analysis.ipynb       # Milestone 2: News sentiment pipeline
├── 03_feature_engineering.ipynb      # Milestone 3: Feature creation
├── 04_rl_environment.ipynb           # Milestone 4: RL environment setup
├── 05_hierarchical_agents.ipynb      # Milestone 5: Agent architecture
├── 06_training_evaluation.ipynb      # Milestone 6: Training and backtesting
├── data/                             # Data storage
│   ├── raw/                          # Raw downloaded data
│   └── processed/                    # Cleaned and processed data
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd reinforcement_learning_in_finance
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

## Milestones

### ✅ Milestone 1: Data Collection
- **File:** `01_data_collection.ipynb`
- **Objective:** Collect market data for 14 global assets
- **Assets:** SPY, QQQ, IWM, EFA, EEM, FXI, VGK, EWG, EWJ, GLD, USO, SLV, GBTC, ETHE
- **Period:** 2003-2024
- **Output:** Cleaned daily returns data

### 🔄 Milestone 2: Sentiment Analysis (Next)
- **File:** `02_sentiment_analysis.ipynb`
- **Objective:** Process news articles for sentiment scores
- **Model:** FinBERT for financial text classification
- **Output:** Monthly sentiment scores per asset

### 📋 Upcoming Milestones
- **Milestone 3:** Feature Engineering
- **Milestone 4:** RL Environment Design
- **Milestone 5:** Hierarchical Agent Architecture
- **Milestone 6:** Training and Evaluation

## Data Science Approach

**KISS Policy (Keep It Simple, Stupid):**
- Clear, well-documented code
- Comprehensive markdown explanations
- Simple but effective implementations
- Reproducible results with fixed seeds

**Key Principles:**
- Modular design with separate notebooks for each milestone
- Extensive error handling and data validation
- Clear documentation of assumptions and limitations
- Focus on real-world applicability

## Usage

1. **Start with Milestone 1:** Run `01_data_collection.ipynb` to collect market data
2. **Follow the sequence:** Each milestone builds on the previous one
3. **Check outputs:** Verify data quality and intermediate results
4. **Customize:** Modify parameters and assets as needed

## Data Sources

- **Market Data:** Yahoo Finance (yfinance)
- **News Data:** NewsAPI (planned for Milestone 2)
- **Sentiment Model:** FinBERT (planned for Milestone 2)

## Performance Metrics

The final system will be evaluated against:
- Equal-weight portfolio
- S&P 500 benchmark
- Risk-parity portfolio
- Single monolithic RL agent

## Contributing

1. Follow the KISS principle
2. Add comprehensive markdown documentation
3. Include error handling and validation
4. Test with different market conditions

## License

[Add your license here]

## Contact

[Add your contact information here] 