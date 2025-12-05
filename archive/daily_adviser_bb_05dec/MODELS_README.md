# Model Training Documentation

## Overview

The daily adviser uses three specialized PPO (Proximal Policy Optimization) agents trained on different market regimes:

| Model | Purpose | Training Regime | Reward Function |
|-------|---------|----------------|-----------------|
| `agent_bull.zip` | Growth allocation | Low volatility periods | Dynamic Sortino Ratio |
| `agent_bear.zip` | Risk protection | High volatility periods | Dynamic Sortino Ratio |
| `agent_sniper.zip` | Precision timing | All market conditions | Dynamic Sortino Ratio |

## Model Specifications

### Architecture
- **Algorithm**: PPO (Proximal Policy Optimization) from Stable-Baselines3
- **Policy**: MlpPolicy (Multi-Layer Perceptron)
- **Observation Space**: (60, 7, 3) - 60-day lookback, 7 assets, 3 features per asset
- **Action Space**: 8 continuous values (7 assets + cash, converted to weights via softmax)

### Training Assets (In Order)
1. NVDA (NVIDIA)
2. MU (Micron Technology)
3. AAPL (Apple)
4. AMD (Advanced Micro Devices)
5. ASML (ASML Holding)
6. MSFT (Microsoft)
7. GOOG (Google/Alphabet)
8. CASH

⚠️ **CRITICAL**: Asset order must match exactly between training and production.

## Features

Each asset has 3 features at each timestep:

### 1. Normalized Volatility
```python
roll_std = log_returns.rolling(20).std()
roll_mean = roll_std.rolling(252).mean()
roll_std_dev = roll_std.rolling(252).std()
norm_vol = (roll_std - roll_mean) / (roll_std_dev + 1e-8)
```
- **Range**: Clipped to [-3, 3]
- **Interpretation**: Z-score of current 20-day volatility relative to 252-day history

### 2. RSI (Relative Strength Index)
**Training** (simplified):
```python
delta = prices.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / (loss + 1e-8)
rsi = 100 - (100 / (1 + rs))
norm_rsi = (rsi - 50) / 100  # Scale to [-0.5, 0.5]
```

**Production** (pandas_ta):
```python
rsi = ta.rsi(prices, length=14)  # Uses Wilder smoothing
norm_rsi = (rsi - 50) / 100
```

⚠️ **Note**: Minor distribution difference exists. Training uses SMA-based RSI, production uses Wilder smoothing (per training code comment).

### 3. Market Correlation
```python
market_returns = qqq_benchmark.pct_change()
correlation = log_returns.rolling(60).corr(market_returns)
```
- **Range**: [-1, 1]
- **Interpretation**: 60-day rolling correlation with QQQ

## Training Process

### 1. Data Preparation
- **Source**: Yahoo Finance
- **Period**: See training notebooks for specific date ranges
- **Benchmark**: QQQ (as proxy for SPY in feature engineering)

### 2. Environment
- **Type**: Custom Gym environment `PortfolioRebalanceEnv`
- **Episode Length**: Variable (walks through historical data)
- **Rebalance Frequency**: Monthly (20 trading days)
- **Transaction Costs**: 10 basis points (0.10%)
- **Initial Balance**: $100,000

### 3. Reward Function
```python
def calculate_reward(daily_returns):
    # Dynamic risk aversion based on market volatility
    market_vol = benchmark[-20:].std()
    risk_aversion = 2.0 + (market_vol - 0.015) * 100 if market_vol > 0.015 else 2.0
    
    # Sortino ratio (penalize only downside)
    downside_returns = returns[returns < 0]
    downside_dev = std(downside_returns) if len(downside_returns) > 0 else 0.001
    
    reward = sum(returns) - (risk_aversion * downside_dev)
    return reward
```

**Key Features:**
- Adaptive risk aversion (increases in volatile markets)
- Downside-focused (only penalizes negative volatility)
- Encourages returns while managing drawdowns

### 4. Training Hyperparameters
See `archive/legacy_versions/simple/03_tuning.ipynb` for Optuna-optimized values.

Standard settings:
- **Learning Rate**: ~3e-4
- **Batch Size**: 64
- **N Epochs**: 10
- **Clip Range**: 0.2
- **Total Timesteps**: Varies by agent (see notebooks)

## Training Notebooks

### Location
`/Users/nataliamarko/Documents/GitHub_Projects/reinforcement_learning_in_finance/archive/legacy_versions/simple/`

### Files

#### 1. `01_ai_rebalancer.ipynb`
- Data fetching and preprocessing
- Feature engineering implementation
- Environment setup and testing

#### 2. `02_board_of_directors.ipynb`
- Training of all three agents (Bull, Bear, Sniper)
- Ensemble strategy implementation
- Backtesting framework
- **Output**: `agent_bull.zip`, `agent_bear.zip`, `agent_sniper.zip`

#### 3. `03_tuning.ipynb`
- Hyperparameter optimization with Optuna
- Threshold tuning (volatility triggers, inertia)
- Performance validation

#### Supporting Code:
- `rl_system.py` - Core classes (FeatureEngineer, Environment, Board)
- `config.py` - Transaction costs and risk parameters

## Model Files

### Current Version
- **Location**: `daily_adviser/` directory
- **Training Date**: See notebooks for details
- **File Sizes**: ~2.1 MB each (compressed)

### File Format
Standard Stable-Baselines3 `.zip` format containing:
- Policy network weights
- Value network weights  
- Optimizer state
- Normalization statistics (if applicable)

### Loading Models
```python
from stable_baselines3 import PPO

bull_agent = PPO.load("agent_bull")
bear_agent = PPO.load("agent_bear")
sniper_agent = PPO.load("agent_sniper")
```

## Retraining

### When to Retrain
- Asset composition changes
- Market regime shift not captured by current models
- Feature engineering modifications
- Significant underperformance vs. backtest

### How to Retrain

1. **Navigate to training directory:**
   ```bash
   cd archive/legacy_versions/simple/
   ```

2. **Update data:**
   - Modify date ranges in `01_ai_rebalancer.ipynb`
   - Ensure latest market data is fetched

3. **Run training notebooks in order:**
   ```bash
   jupyter notebook 01_ai_rebalancer.ipynb   # Prepare data
   jupyter notebook 02_board_of_directors.ipynb  # Train agents
   jupyter notebook 03_tuning.ipynb          # Optimize thresholds
   ```

4. **Copy new models to production:**
   ```bash
   cp agent_*.zip ../../daily_adviser/
   ```

5. **Update configuration** in `daily_advisor.py` if thresholds changed

### Validation Checklist
- [ ] Backtest performance > buy-and-hold
- [ ] Sharpe ratio > 1.5
- [ ] Max drawdown < 25%
- [ ] Models load without errors
- [ ] Feature shapes match (60, 7, 3)
- [ ] Weights sum to 1.0
- [ ] Transaction costs accounted for

## Performance Metrics

### Training Performance
Refer to training notebooks for detailed metrics.

Expected characteristics:
- **Sharpe Ratio**: > 1.5 (annualized)
- **Win Rate**: 55-65%
- **Max Drawdown**: < 20-25%
- **Turnover**: Moderate (inertia threshold manages this)

### Production Monitoring
Track these metrics in production:
- Actual vs. predicted returns
- Regime detection accuracy
- Trading frequency
- Transaction cost impact

## Troubleshooting

### Model Loading Errors

**Error**: `FileNotFoundError: agent_bull.zip`

**Solution**: Copy models from archive:
```bash
cp archive/legacy_versions/simple/agent_*.zip daily_adviser/
```

### Prediction Shape Mismatch

**Error**: `ValueError: observation shape mismatch`

**Cause**: Feature engineering changed or asset order different

**Solution**:
1. Verify asset order matches training: `['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']`
2. Check feature shape is (60, 7, 3)
3. Retrain if assets changed

### Performance Degradation

**Symptoms**: 
- Frequent losses
- High turnover
- Regime misdetection

**Diagnosis**:
1. Check if market conditions are outside training distribution
2. Review feature drift (RSI, correlation patterns)
3. Validate data quality

**Solutions**:
- Increase retraining frequency
- Expand training data to include recent market conditions
- Adjust thresholds via `03_tuning.ipynb`

## Version Control

### Model Versioning
Consider tracking:
- Training date
- Data date range
- Performance metrics
- Hyperparameters

Suggested naming: `agent_bull_YYYYMMDD.zip`

### Rollback Procedure
Keep previous model versions:
```bash
cp agent_bull.zip agent_bull_backup_$(date +%Y%m%d).zip
```

If needed, restore:
```bash
cp agent_bull_backup_20251201.zip agent_bull.zip
```

## Future Improvements

### Potential Enhancements
1. **Walk-Forward Validation**: Retrain periodically on rolling windows
2. **Online Learning**: Update models with recent data
3. **Expanded Assets**: Include more sectors/asset classes
4. **Alternative Features**: Add fundamental data, sentiment signals
5. **Ensemble Weighting**: Learn optimal agent mixing vs. hardcoded ratios

### Research Questions
- Does RSI calculation difference impact significantly?
- Can we reduce training data requirements?
- Is monthly rebalancing optimal?

## References

- **Training Code**: `archive/legacy_versions/simple/rl_system.py`
- **Documentation**: `archive/legacy_versions/simple/RL_system_docum.md`
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Original Paper**: Proximal Policy Optimization (Schulman et al., 2017)
