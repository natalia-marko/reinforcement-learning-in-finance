# Portfolio Rebalancing with Reinforcement Learning

## Notebook Pipeline

This directory contains a complete pipeline for training and testing RL agents for portfolio rebalancing.

### 📊 01_data_preparation.ipynb
**Status**: ✅ Complete
**Purpose**: Prepare and clean financial data for training

**Output**:
- `data/processed/train.parquet` - Training data (2020-01-01 to 2023-12-20)
- `data/processed/test.parquet` - Test data (2023-12-27 to 2024-12-25)
- `data/processed/metadata.json` - Feature names, tickers, etc.

**Data**:
- **Tickers**: NVDA, MU, AAPL, AMD, ASML, MSFT, GOOG
- **Features**: 135 technical + macro features per asset
- **Frequency**: Weekly resampling

---

### 🤖 02_train_baseline.ipynb
**Status**: ✅ Complete (refactored)
**Purpose**: Train PPO agents using walk-forward validation

**Key Features**:
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Architecture**: SimpleActor [256, 128] layers
- **Validation**: 3-fold walk-forward (expanding window)
- **Objective**: Maximize Sharpe ratio
- **Constraints**: Max 40% per asset, 2.5 bps transaction costs

**Output**:
- `models/baseline/fold_*/best_model.zip` - Trained models
- `models/baseline/fold_results.csv` - Validation metrics
- `models/baseline/fold_*/logs/` - Training logs

**Imports**: Uses standardized components from `rl_system.py`:
- `PortfolioEnv` - Trading environment
- `PortfolioMonitor` - Stats tracking wrapper
- `TrainingLogger` - Comprehensive logging callback
- `create_walk_forward_folds` - Validation splits

---

### 📈 03_backtests.ipynb
**Status**: ✅ NEW - Just created!
**Purpose**: Comprehensive out-of-sample backtesting

**Test Period**:
- 2023-12-27 to 2024-12-25 (53 weeks, ~1 year)
- Completely held-out from training

**Strategies Tested**:
1. **RL Agent** - Best model from walk-forward validation
2. **Equal Weight (1/N)** - Naive diversification baseline
3. **Buy & Hold** - Zero rebalancing
4. **Risk Parity** - Volatility-weighted
5. **Momentum** - Trend-following (12-week lookback)
6. **Minimum Variance** - Risk-minimizing

**Sections**:
1. **Setup & Imports** - Configuration and data loading
2. **Load Models** - Best fold model selection
3. **Baseline Strategies** - 5 benchmark implementations
4. **Backtest Execution** - Run all strategies
5. **Performance Metrics** - 20+ comprehensive metrics
6. **Visualizations**:
   - Cumulative returns comparison
   - RL agent weight evolution
   - Risk-return scatter
   - Drawdown analysis

**Metrics Calculated**:
- **Returns**: Total, annualized, weekly
- **Risk**: Volatility, downside deviation, max drawdown, VaR, CVaR
- **Risk-adjusted**: Sharpe, Sortino, Calmar, Omega ratios
- **Trading**: Turnover, transaction costs, win rate

**Output**:
- `results/backtests/backtest_summary.csv` - All metrics
- `results/backtests/trajectory_*.csv` - Weight/return trajectories
- `results/backtests/backtest_report.json` - Summary report
- `results/backtests/*.png` - Publication-ready plots

---

## Usage

### Quick Start
```bash
# 1. Prepare data
jupyter notebook 01_data_preparation.ipynb

# 2. Train RL agents
jupyter notebook 02_train_baseline.ipynb

# 3. Backtest and compare
jupyter notebook 03_backtests.ipynb
```

### Expected Runtime
- **01_data_preparation**: ~5 minutes
- **02_train_baseline**: ~2-4 hours (3 folds, 200k steps each)
- **03_backtests**: ~5-10 minutes

---

## Code Organization

All core components are centralized in `../rl_system.py`:

```python
from rl_system import (
    # Environment
    PortfolioEnv,           # Gym environment for portfolio rebalancing

    # Monitoring
    PortfolioMonitor,       # Wrapper to capture stats before reset
    TrainingLogger,         # Comprehensive logging with early stopping

    # Model
    SimpleActor,            # Neural network architecture

    # Validation
    create_walk_forward_folds,  # Split data chronologically
    print_fold_summary           # Display fold info
)
```

**Benefits**:
- Single source of truth
- No code duplication
- Easy maintenance
- Consistent behavior across notebooks

---

## Results Interpretation

### Success Criteria
✅ RL agent achieves **positive Sharpe ratio**
✅ Outperforms at least **equal weight baseline**
✅ Reasonable **transaction costs** (< 5% of returns)
✅ Stable performance (no extreme volatility)

### Realistic Expectations
- RL may not beat ALL baselines in ALL metrics
- Some periods of underperformance are expected
- Transaction costs impact returns significantly
- 1-year test period is relatively short

---

## Next Steps

Based on backtest results:

### If RL Agent Performs Well:
1. **Hyperparameter tuning** - Grid search on learning rate, entropy coef
2. **Architecture experiments** - Deeper networks, attention mechanisms
3. **Ensemble methods** - Combine multiple fold models
4. **Production deployment** - Risk management, monitoring

### If RL Agent Struggles:
1. **Feature engineering** - Add sentiment, fundamentals
2. **Reward function** - Try different objectives (returns, CVaR)
3. **Environment changes** - Different rebalancing frequencies
4. **Algorithm alternatives** - Try SAC, TD3, or A2C

---

## File Structure

```
notebooks/
├── 01_data_preparation.ipynb          # Data pipeline
├── 02_train_baseline.ipynb            # RL training
├── 03_backtests.ipynb                 # Out-of-sample testing ⭐ NEW
├── BACKTEST_PLAN.md                   # Detailed design doc
└── README.md                          # This file

../rl_system.py                         # Core components
../data/processed/                      # Prepared data
../models/baseline/                     # Trained models
../results/backtests/                   # Backtest outputs
```

---

## Dependencies

```bash
# Core ML
stable-baselines3>=2.0.0
torch>=2.0.0
gymnasium>=0.28.0

# Data
pandas>=2.0.0
numpy>=1.24.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Statistics
scipy>=1.10.0
```

---

## Notes

- All notebooks use reproducible seeds (42)
- Trained models are saved in ZIP format
- Logs are saved to CSV for easy analysis
- Plots are saved at 300 DPI for publication

---

## Support

For issues or questions:
1. Check the plan documents (`BACKTEST_PLAN.md`)
2. Review inline comments in notebooks
3. Verify data preparation completed successfully
4. Ensure training finished before backtesting

---

**Last Updated**: November 17, 2024
**Version**: 1.0
**Status**: Production Ready ✅
