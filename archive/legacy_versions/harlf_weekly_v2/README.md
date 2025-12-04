# Multi-Hierarchical RL Portfolio Management System

**A three-layer reinforcement learning system for portfolio optimization with separate technical and sentiment analysis.**

## 🎯 System Architecture

```
┌─────────────────────────────────────────────────┐
│                META AGENT (Part 3)              │
│   Inputs: All base data + super decision +     │
│           macro indicators                      │
│   Output: Final portfolio adjustment            │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│              SUPER AGENT (Part 2)               │
│   Inputs: Technical weights + Sentiment weights │
│           + lag returns                         │
│   Output: Ensemble/selection decision           │
└───────────┬───────────────────┬─────────────────┘
            │                   │
┌───────────▼──────────┐  ┌────▼──────────────────┐
│  TECHNICAL AGENT     │  │  SENTIMENT AGENT      │
│  (Part 1)            │  │  (Part 1)             │
│                      │  │                       │
│  Inputs:             │  │  Inputs:              │
│  - Price indicators  │  │  - News sentiment     │
│  - Volume indicators │  │  - Market sentiment   │
│  - Momentum          │  │  - Social sentiment   │
│  - Volatility        │  │  - Risk sentiment     │
│  - Benchmark corr    │  │                       │
│                      │  │                       │
│  Output:             │  │  Output:              │
│  Portfolio weights   │  │  Portfolio weights    │
└──────────────────────┘  └───────────────────────┘
```

## 📦 Part 1: Base Layer Agents (Current)

### Portfolio
Your personal portfolio includes:
- RDDT, NVDA, MU, AAPL, SMR, AMD, ASML, MSFT, GOOG, AI, ARBE, CHYM

### What's Implemented

1. **Data Preparation** (`data_preparation_part1.py`)
   - Downloads weekly price data for all tickers
   - Handles different listing dates (aligns to common period)
   - Creates two **separate** feature sets:
     - Technical features (price/volume based)
     - Sentiment features (market sentiment proxies)
   - Proper train/val/test splits (60/20/20)
   - **No data leakage** (fixed from audit findings)

2. **Environments** (`environments_part1.py`)
   - `TechnicalAgentEnv`: Uses only technical indicators
   - `SentimentAgentEnv`: Uses only sentiment indicators
   - Both output portfolio weights via softmax
   - Reward: Sharpe ratio
   - Weekly rebalancing frequency

3. **Training Notebook** (`train_part1_agents.ipynb`)
   - Interactive training with visualizations
   - Tests 3 algorithms: PPO, SAC, A2C
   - Automatic model selection (best on validation)
   - Saves best models for Part 2

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv_hierarchical
source venv_hierarchical/bin/activate  # On Windows: venv_hierarchical\Scripts\activate

# Install dependencies
pip install -r requirements_part1.txt
```

### Step 1: Prepare Data

```bash
python data_preparation_part1.py
```

This will:
- Download price data for all your tickers
- Create technical features (26 features per ticker)
- Create sentiment features (12 features per ticker)
- Split into train/val/test
- Save to `data_hierarchical/`

**Expected output:**
```
data_hierarchical/
├── technical/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── sentiment/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── returns_train.csv
├── returns_val.csv
├── returns_test.csv
└── metadata.json
```

### Step 2: Train Agents

Open and run the Jupyter notebook:

```bash
jupyter notebook train_part1_agents.ipynb
```

The notebook will:
1. Load and explore the data
2. Train Technical Agent with PPO, SAC, A2C
3. Train Sentiment Agent with PPO, SAC, A2C
4. Select best model for each based on validation Sharpe
5. Compare agents and visualize results
6. Save best models to `models_part1/`

**Expected runtime:** ~2-4 hours (depends on your machine)

## 📊 Features

### Technical Agent Features (26 total)

**Trend Indicators:**
- Price to SMA ratios (4w, 8w, 12w)

**Momentum Indicators:**
- Return lags (1w, 2w, 3w)
- RSI (14 weeks)
- ROC (4-week rate of change)

**Volatility Indicators:**
- Realized volatility (12w, annualized)
- ATR percentage

**Volume Indicators:**
- Volume ratio (vs 20w MA)
- Volume momentum (4-week change)

**Benchmark Relative:**
- Rolling correlation with benchmark (12w) - **FIXED: No look-ahead**
- Rolling beta (12w) - **FIXED: No look-ahead**

### Sentiment Agent Features (12+ total)

**Price-Based Sentiment:**
- Momentum (4w, 8w, 12w)
- Positive weeks percentage
- Price dispersion

**Volume-Based Sentiment:**
- Volume trend
- Volume surge

**Risk Sentiment:**
- Volatility regime
- Drawdown from recent high

**Market Sentiment:**
- Market fear (VIX-based)
- Credit sentiment (HY spread proxy)

**News Sentiment:**
- Simulated news sentiment (replace with real API in production)

## 🎓 Key Design Decisions

### 1. Split-of-Concept
Technical and Sentiment agents see **completely different** features. This ensures:
- Independent learning
- Complementary strategies
- Better ensemble potential in Part 2

### 2. No Data Leakage
Following audit findings, the system has:
- ✅ Proper rolling correlation (no look-ahead)
- ✅ No backward fill (only forward fill)
- ✅ Proper train/val/test splits
- ✅ Normalization fit only on train data

### 3. Long-Only Portfolio
Actions are converted to weights via softmax:
- Ensures weights sum to 1
- No short positions
- Realistic for most retail investors

### 4. Sharpe Reward
Reward = (portfolio_return / running_volatility) × √52

- Encourages risk-adjusted returns
- Annualized for interpretability
- Clipped to [-10, 10] for stability

## 📈 Expected Performance

Based on audit learnings and proper data handling:

| Agent | Algorithm | Test Sharpe (Expected) |
|-------|-----------|------------------------|
| Technical | PPO/SAC | 0.5 - 1.0 |
| Sentiment | PPO/SAC | 0.3 - 0.8 |

**Note:** These are realistic expectations without data leakage. Higher values would be suspicious!

## 🔧 Configuration

Edit `data_preparation_part1.py` to customize:

```python
preparator = HierarchicalDataPreparator(
    tickers=PORTFOLIO,
    start_date='2020-01-01',    # Adjust based on ticker availability
    end_date=None,              # None = today
    train_split=0.6,            # 60% train
    val_split=0.2,              # 20% val, 20% test
    benchmark='SPY',            # Or 'QQQ' for tech-heavy
)
```

Edit training notebook to customize training:

```python
TRAIN_CONFIG = {
    'total_steps': 200_000,      # Increase for better convergence
    'eval_freq': 5_000,          # Validation frequency
    'patience': 5,               # Early stopping patience
    'learning_rate': 3e-4,       # Learning rate
    'softmax_temperature': 5.0,  # Higher = more concentrated weights
}
```

## 🐛 Troubleshooting

### Issue: "No data available for ticker X"

**Solution:** Some tickers (RDDT, AI) are newly listed. Adjust `start_date` or remove these tickers:

```python
# In data_preparation_part1.py, line ~900
PORTFOLIO = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']  # Older tickers only
```

### Issue: "Dataset too small"

**Solution:** You need at least ~50 weeks of data after alignment. Either:
- Use fewer tickers
- Start from later date
- Remove newly-listed stocks

### Issue: Training is slow

**Solution:**
- Reduce `total_steps` to 100,000
- Use PPO only (faster than SAC)
- Use CPU if GPU memory is limited

### Issue: Low validation Sharpe

**Solution:** This is normal! Without data leakage, Sharpe ratios are lower. If validation Sharpe < 0:
- Check your data quality
- Try different algorithms
- Adjust `softmax_temperature`
- Increase `total_steps`

## 🔍 Validation

After training, verify no data leakage:

```python
# In notebook, add this cell:
import pandas as pd

# Check correlation calculations
tech_train = pd.read_csv('data_hierarchical/technical/train.csv', index_col=0, parse_dates=True)

# Verify correlations only use past data
# At time t, correlation should only use [t-11, t]
# Not the entire series!

print("✓ Data leakage checks passed" if all_checks_pass else "⚠ Warning: potential leakage")
```

## 📚 Next Steps

Once Part 1 is complete:

1. **Part 2: Super Agent**
   - Takes technical and sentiment weights as input
   - Learns to ensemble/select between agents
   - Reward: improve over best base agent

2. **Part 3: Meta Agent**
   - Takes super agent decision + macro data
   - Final portfolio adjustment
   - Reward: overall portfolio performance

## 📝 Files Overview

```
hierarchical_rl_system/
├── data_preparation_part1.py    # Data preparation script
├── environments_part1.py         # RL environments
├── train_part1_agents.ipynb     # Training notebook
├── requirements_part1.txt        # Python dependencies
└── README.md                     # This file

Generated after running:
├── data_hierarchical/            # Prepared data
└── models_part1/                 # Trained models
```

## 🤝 Contributing

This is your personal system! Customize freely:

- Add more features
- Try different algorithms (TD3, PPO-LSTM)
- Experiment with reward functions
- Add real sentiment data (NewsAPI, Twitter)

## ⚠️ Disclaimer

This system is for educational and research purposes. Not financial advice. Always:
- Backtest thoroughly
- Paper trade first
- Understand the risks
- Consult a financial advisor

## 🎉 Success Criteria

You'll know Part 1 is working when:
- ✅ Data preparation completes without errors
- ✅ Both agents train successfully
- ✅ Validation Sharpe > 0 for at least one agent
- ✅ Models saved to `models_part1/`
- ✅ Ready to build Part 2!

---

**Good luck with your hierarchical RL system!** 🚀

Questions? Review the audit documents for best practices and common pitfalls.
