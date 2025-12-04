# Part 1: Base Agents Training

## Overview
Train individual base agents using technical and sentiment features. Experiment with different reward functions to find the best approach for your data.

---

## 📁 Files Structure

### Core Training Files
- `environments_part1.py` - RL environments (Technical & Sentiment)
- `train_part1_agents.ipynb` - Original training (EMA Sharpe)
- `retrain_base_agents.ipynb` - Retraining with visualizations

### Reward Function Approaches
- `ema_sharpe_approach.py` - EMA-based Sharpe (baseline)
- `differential_sharpe_approach.py` - Differential Sharpe
- `multi_objective.py` - Multi-objective with penalties

### Comparison & Tuning
- `compare_reward_functions.py` - Compare all approaches
- `compare_reward_functions.ipynb` - Interactive comparison
- `tune_multi_objective.py` - Hyperparameter tuning

---

## 🎯 Three Reward Function Approaches

### 1. EMA-Based Sharpe (Current Baseline)
**File:** `ema_sharpe_approach.py` / `environments_part1.py`

**Reward Function:**
```python
# Rolling statistics with EMA
ema_mean = (1-α) × ema_mean + α × return
ema_var = (1-α) × ema_var + α × (return - ema_mean)²

# Sharpe-like reward
reward = (return / √ema_var) × √52
```

**Current Results:**
- Technical Agent: Test Sharpe 1.78 (PPO)
- Sentiment Agent: Test Sharpe 1.92 (SAC)

**Pros:**
- Simple and interpretable
- Proven performance
- No hyperparameter tuning needed
- Fast convergence

**Cons:**
- No explicit diversification incentive
- No turnover control
- May learn concentrated strategies

---

### 2. Differential Sharpe Ratio
**File:** `differential_sharpe_approach.py`

**Reward Function:**
```python
# Online gradient ascent for Sharpe
n_effective = 1 + decay × n_effective
δ = return - mean
mean += (1 - decay) × δ
var = decay × var + (1 - decay) × δ²

# Differential reward
reward = (return × std - 0.5 × mean × δ) / (std² + ε)
reward = clip(reward × √52, -10, 10)
```

**Pros:**
- Theoretically grounded
- Direct Sharpe optimization
- Adaptive to changing markets
- No hyperparameters to tune

**Cons:**
- Complex implementation
- Sensitive to initialization
- May be unstable early in training

**Reference:** Moody et al. (1998)

---

### 3. Multi-Objective (Optimized)
**File:** `multi_objective.py`

**Reward Function:**
```python
reward = return_scale × return 
         - λ_vol × |return - mean_return|
         - λ_conc × max(0, max_weight - 0.3)
         - λ_turn × Σ|weights_t - weights_{t-1}|
```

**Optimized Configurations:**

**Technical Agent:**
```python
return_scale = 5.0            # Less aggressive
volatility_penalty = 0.15     # More stability
concentration_penalty = 0.75  # More diversification
turnover_penalty = 0.015      # Less trading
```

**Sentiment Agent:**
```python
return_scale = 3.0            # Much less aggressive
volatility_penalty = 0.2      # Strong penalty
concentration_penalty = 1.0   # Force diversification
turnover_penalty = 0.02       # Minimize rebalancing
```

**Pros:**
- Explicit risk control
- Forces diversification
- Transaction cost awareness
- Interpretable hyperparameters
- Better generalization

**Cons:**
- Requires tuning
- More complex
- Slower training

**Reference:** Moody & Saffell (2001)

---

## 🚀 Usage

### Option 1: Quick Start (EMA Sharpe - Proven)
```bash
jupyter notebook retrain_base_agents.ipynb
```
Run all cells. Uses default EMA Sharpe reward (proven to work).

### Option 2: Compare All Approaches
```bash
# Interactive comparison
jupyter notebook compare_reward_functions.ipynb

# Or script
python compare_reward_functions.py
```

### Option 3: Tune Multi-Objective
```bash
# Quick test (15 minutes)
python tune_multi_objective.py --quick

# Full grid search (60 minutes)
python tune_multi_objective.py --grid --technical
python tune_multi_objective.py --grid --sentiment
```

### Option 4: Train Specific Approach
```bash
# EMA Sharpe (baseline)
# Just use train_part1_agents.ipynb or retrain_base_agents.ipynb

# Differential Sharpe
python differential_sharpe_approach.py

# Multi-Objective (optimized)
python multi_objective.py
```

---

## 📊 Expected Performance Comparison

| Approach | Technical (Test) | Sentiment (Test) | Avg |
|----------|------------------|------------------|-----|
| EMA Sharpe | 1.78 | 1.92 | **1.85** |
| Differential Sharpe | ~1.7-1.9 | ~1.8-2.0 | **~1.85** |
| Multi-Objective (optimized) | **~2.0-2.2** | **~1.4-1.6** | **~1.8** |

**Note:** Multi-objective optimized for lower generalization gap, not necessarily highest test Sharpe.

---

## 📁 Output

### Models
- `../models_part1/` - EMA Sharpe models (baseline)
- `../models_diff_sharpe/` - Differential Sharpe models
- `../models_multi_objective/` - Multi-objective models

### Results & Visualizations
- `reward_function_comparison.json` - Detailed results
- `../plots/` - All visualizations
- Training plots in notebooks

---

## 🎛️ Tuning Multi-Objective

### Penalty Effects

**Return Scale** (`return_scale`):
- Higher (10+): Aggressive, prioritizes returns
- Lower (3-5): Conservative, prioritizes stability
- Recommended: 5.0 (technical), 3.0 (sentiment)

**Volatility Penalty** (`volatility_penalty`):
- Higher (0.2+): Lower volatility portfolios
- Lower (0.0-0.05): Allows higher volatility
- Recommended: 0.15 (technical), 0.2 (sentiment)

**Concentration Penalty** (`concentration_penalty`):
- Higher (1.0+): Forces diversification
- Lower (0.0-0.25): Allows concentration
- Threshold: 30% per asset
- Recommended: 0.75 (technical), 1.0 (sentiment)

**Turnover Penalty** (`turnover_penalty`):
- Higher (0.02+): Buy-and-hold strategy
- Lower (0.0-0.005): Active rebalancing
- Recommended: 0.015 (technical), 0.02 (sentiment)

### Tuning Workflow

1. **Start with defaults** - Run and observe results
2. **Diagnose issues:**
   - High Val-Test gap → Increase penalties
   - Low test Sharpe → Decrease penalties or increase return_scale
   - Negative Sharpe → Much stronger penalties
3. **Quick test** - Try `tune_multi_objective.py --quick`
4. **Fine-tune** - Adjust based on results
5. **Full search** - Run grid search if needed

See `../docs/TUNING_GUIDE.md` for detailed instructions.

---

## 🔬 Environments

### `environments_part1.py`

**Classes:**
- `TechnicalAgentEnv` - Portfolio environment using technical indicators
- `SentimentAgentEnv` - Portfolio environment using sentiment indicators

**Features:**
- Continuous action space (portfolio weights via softmax)
- EMA-based Sharpe ratio reward (default)
- Episode-based training
- Rolling volatility tracking

**Creating Environments:**
```python
from environments_part1 import TechnicalAgentEnv, SentimentAgentEnv

tech_train = TechnicalAgentEnv('../data_hierarchical', 'train')
tech_val = TechnicalAgentEnv('../data_hierarchical', 'val')
tech_test = TechnicalAgentEnv('../data_hierarchical', 'test')
```

---

## 📈 Training Configuration

### Default (EMA Sharpe)
```python
TRAIN_CONFIG = {
    'total_steps': 300_000,
    'eval_freq': 5_000,
    'patience': 5,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'rolling_vol_window': 12,
    'softmax_temperature': 3.0,
}
```

### Algorithms
- **PPO** (Proximal Policy Optimization) - Stable, on-policy
- **SAC** (Soft Actor-Critic) - Sample-efficient, off-policy
- **A2C** (Advantage Actor-Critic) - Fast, on-policy

---

## 🎓 Recommendation

### For Your First Run:
**Use EMA Sharpe (default in notebooks)** - Proven to work, simple, no tuning needed

### For Experimentation:
**Run comparison notebook** - See all approaches side-by-side

### For Optimization:
**Try Multi-Objective** - Best for production with explicit risk controls

### For Research:
**Differential Sharpe** - Theoretically interesting, adaptive approach

---

## 📊 Evaluation Metrics

### Sharpe Ratio
```
Sharpe = (Mean Return - Risk-Free Rate) / Std Deviation
```
- Good: > 1.0
- Excellent: > 2.0

### Generalization Gap
```
Gap = Validation Sharpe - Test Sharpe
```
- Good: < 0.5
- Warning: 0.5 - 1.0
- Overfitting: > 1.0

### Key Checks
- Test Sharpe > 1.5 = Good performance
- Val-Test gap < 1.0 = Low overfitting
- Consistent across splits = Robust

---

## 🔄 Workflow

### Workflow 1: Quick & Proven
1. Run `retrain_base_agents.ipynb`
2. Review visualizations
3. Use best models → Part 2

### Workflow 2: Compare & Choose
1. Run `compare_reward_functions.ipynb`
2. Analyze performance across approaches
3. Select winner
4. Use best models → Part 2

### Workflow 3: Deep Optimization
1. Run baseline (EMA Sharpe)
2. Run `tune_multi_objective.py --quick`
3. Analyze results
4. Full grid search if needed
5. Use best models → Part 2

---

## 📚 References

1. Moody, J., & Saffell, M. (2001). *Learning to trade via direct reinforcement*. IEEE transactions on neural Networks, 12(4), 875-889.

2. Moody, J., Wu, L., Liao, Y., & Saffell, M. (1998). *Performance functions and reinforcement learning for trading systems and portfolios*. Journal of Forecasting, 17(5‐6), 441-470.

---

## 🎯 Next Steps

After training and selecting best approach:
1. Review test performance
2. Analyze portfolio allocations
3. Check generalization gaps
4. Save best models
5. Proceed to `../03_part2_super_agent/`

---

**Tip:** Start simple with EMA Sharpe, then experiment with other approaches if you want better risk control or generalization.
