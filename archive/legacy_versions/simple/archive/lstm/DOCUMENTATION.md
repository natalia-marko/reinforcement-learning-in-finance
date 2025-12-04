# RL Portfolio Management System - Documentation

## Overview
A reinforcement learning system for dynamic portfolio rebalancing using tech stocks, implementing temporal integrity safeguards and realistic market constraints.

## Quick Start
```bash
# 1. Setup environment
conda activate demand_forecast_env
pip install sb3-contrib gymnasium pandas numpy yfinance ta

# 2. Prepare data (creates train/test split)
python data_eng.py

# 3. Train model (walk-forward validation)
python train_lstm.py

# 4. Visualize results
python plot_learning_curves.py

# 5. Backtest on held-out data
python backtest.py
```

## System Architecture

### Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `data_eng.py` | Data preparation | Raw price download, feature engineering with expanding windows, train/test split |
| `rl_system.py` | Environment | ImprovedPortfolioEnv with multi-scale features, stable rewards |
| `train_lstm.py` | Training pipeline | Walk-forward validation, temporal integrity, early stopping |
| `backtest.py` | Evaluation | True out-of-sample testing, realistic costs |
| `config.py` | Configuration | Centralized parameters, paths, hyperparameters |

### Data Flow
```
Raw Prices (2014-2025)
    ↓
Train/Val Split (80%) | Test Split (20% held out)
    ↓                      ↓
Feature Engineering    [NEVER TOUCHED]
    ↓                      ↓
Walk-Forward Training     ↓
    ↓                      ↓
Best Model Selection      ↓
    ↓                      ↓
    └──────────────→ Final Backtest
```

## Critical Design Decisions

### 1. Temporal Integrity (No Data Leakage)
- **Expanding Windows**: All normalization uses `expanding().shift(1)` - only past data
- **56-Week Gap**: Between train/validation sets (matches longest feature window)
- **Per-Fold Features**: Recalculated for each fold using only available data
- **Test Set Isolation**: 20% held out, saved separately, never accessed during training

### 2. Feature Engineering
```python
FEATURE_WINDOWS = [4, 13, 26]  # weeks

# Multi-scale captures:
# 4w:  Short-term momentum, earnings reactions
# 13w: Quarterly cycles, sector rotation  
# 26w: Medium trends, institutional rebalancing
# 52w: Annual cycles, mean reversion
```

**113 Total Features**:
- Returns & momentum (multi-scale)
- Volatility (4w, 13w, 52w)
- Technical indicators (RSI, MACD, Bollinger)
- Relative strength vs market
- Drawdown metrics
- Calendar features (seasonality)

### 3. Portfolio Environment

**State Space**:
- Market features (113 dims)
- Current portfolio weights (7 dims)  
- Recent performance metrics (4 dims)
- Optional: Correlation features

**Action Space**:
- Continuous weights [0, 1] for 7 assets
- Softmax transformation with 40% concentration limit
- Minimum trade filter (2% threshold)

**Reward Function**:
```python
reward = 0.4 * sharpe_component +
         0.3 * sortino_component +  
         0.2 * diversification_bonus +
         0.1 * transaction_penalty
```

### 4. Training Configuration

**Walk-Forward Validation**:
```
Fold 1: Train[0:123] → Gap[123:179] → Val[179:302]
Fold 2: Train[0:246] → Gap[246:302] → Val[302:425]  
Fold 3: Train[0:369] → Gap[369:425] → Val[425:495]
```

**Hyperparameters**:
- Algorithm: RecurrentPPO with LSTM
- Learning rate: 3e-4
- Batch size: 128
- Training steps: 100k per fold
- Early stopping: 10 evaluations patience

**Transaction Costs**:
- Base: 30 bps
- Slippage: 20 bps
- Total: 50 bps per trade

## File Structure
```
project/
├── config.py                 # Configuration parameters
├── data_eng.py              # Data preparation pipeline
├── rl_system.py             # ImprovedPortfolioEnv class
├── train_lstm.py            # Training with walk-forward
├── backtest.py              # Out-of-sample evaluation
├── plot_learning_curves.py  # Visualization
│
├── raw_price_data.csv       # Training prices (495 weeks)
├── qqq_benchmark.csv        # Benchmark data
├── test_set_NEVER_TOUCH.csv # Hold-out test (123 weeks)
│
├── best_model/              # Trained models
│   ├── fold_0/             
│   ├── fold_1/
│   ├── fold_2/
│   └── best_overall_model.zip
│
└── logs/                    # Training logs
    ├── train_fold_*.csv
    └── fold_*/evaluations.npz
```

## Expected Performance

### Realistic Targets
| Metric | Expected | Warning if |
|--------|----------|------------|
| Annual Return | 8-12% above risk-free | >30% |
| Sharpe Ratio | 0.5-0.8 | >1.5 |
| Max Drawdown | -20% to -30% | <-10% |
| Win Rate | 55-60% | >70% |
| Turnover | 200-400% annual | >800% |

### Baseline Comparisons
- **Equal Weight**: ~10-15% annual return
- **Buy & Hold QQQ**: ~12-18% annual return
- **Your RL Strategy**: Should beat equal-weight by 2-5%

## Common Issues & Solutions

### Issue: "LEAKAGE DETECTED"
**Cause**: Expanding window statistics differ when calculated on different data lengths  
**Solution**: This is expected and safe - the validation uses tolerance of 1%

### Issue: ImportError for PortfolioEnv
**Cause**: Class is named `ImprovedPortfolioEnv`  
**Solution**: `from rl_system import ImprovedPortfolioEnv as PortfolioEnv`

### Issue: Training rewards plateau quickly
**Cause**: Learning rate too high or reward scaling issues  
**Solution**: Reduce learning rate to 1e-4, check reward normalization

### Issue: Validation performance worse than training
**Cause**: Overfitting or insufficient gap  
**Solution**: Increase GAP to 52+ weeks, reduce model complexity

## Validation Checklist

Before deploying with real money:

- [ ] Backtest Sharpe < 1.0 (realistic)
- [ ] Test performance worse than validation (expected)
- [ ] Transaction costs reduce returns by 15-25%
- [ ] Drawdowns exceed 20% (market reality)
- [ ] Equal-weight baseline computed for comparison
- [ ] 2008/2020 crash periods tested
- [ ] Paper trading for 6+ months
- [ ] Monte Carlo simulation (1000+ paths)
- [ ] Slippage analysis on live quotes
- [ ] Risk limits implemented (position limits, stop-losses)

## Key Improvements from Original

1. **Fixed Data Leakage**: Expanding windows with shift(1)
2. **Proper Test Set**: 20% completely held out
3. **Realistic Costs**: 50bps total (was 10bps)
4. **Stable Rewards**: Differential Sharpe with tanh scaling
5. **Correct Returns**: Consistent simple return calculation
6. **Better Gap**: 56 weeks between train/val (was 8)
7. **Regime Detection**: Adapts to market conditions
8. **Correlation Features**: Portfolio risk awareness

## Next Steps

1. **Simplify Architecture**: Consider removing LSTM (overkill for monthly rebalancing)
2. **Add Baselines**: Implement momentum, mean-reversion strategies
3. **Ensemble Methods**: Combine multiple strategies
4. **Risk Overlays**: Add stop-losses, volatility targeting
5. **Alternative Features**: Volume, sentiment, macro indicators

## References

- **Jegadeesh & Titman (1993)**: Returns to Buying Winners and Selling Losers
- **Marcos López de Prado (2018)**: Advances in Financial Machine Learning
- **Stable Baselines3 Docs**: https://stable-baselines3.readthedocs.io/

## Support

For issues or questions, check:
1. Error messages in console
2. Training logs in `logs/` directory  
3. Feature validation warnings (usually safe)
4. Compare against baseline performance

---

**Remember**: A simple strategy that works beats a complex one that doesn't. Start simple, prove it works, then add complexity.
