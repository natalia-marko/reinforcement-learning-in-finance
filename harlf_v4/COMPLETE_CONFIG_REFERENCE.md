# Complete Configuration Reference

## All Available Parameters

### Base Agents Configuration

```python
# Sentiment Environment
sentiment_env_config = {
    'vol_window': 3,           # Rolling window for Sharpe-like reward (3-15)
    'transaction_cost': 0.001, # Transaction cost per trade (0.0005-0.002)
    'risk_lambda': 0.5,        # Legacy risk penalty weight (0.2-1.0)
}

# Technical Environment
technical_env_config = {
    'vol_window': 3,           # Rolling window for Sharpe-like reward (3-15)
}
```

**Reward Function**: `Sharpe-like = mean_return / volatility`

### Hierarchical Agents Configuration

```python
# Super Agent
super_agent_config = {
    'alpha1': 1.0,             # Weight for log returns (0.5-2.0)
    'alpha2': 0.5,             # Weight for max drawdown penalty (0.2-1.5)
    'alpha3': 0.5,             # Weight for volatility penalty (0.2-1.5)
    'exploration_bias': 0.01,  # Exploration bonus (0.001-0.02)
}

# Meta Agent
meta_agent_config = {
    'alpha1': 1.0,             # Weight for log returns (0.5-2.0)
    'alpha2': 0.5,             # Weight for max drawdown penalty (0.2-1.5)
    'alpha3': 0.5,             # Weight for volatility penalty (0.2-1.5)
    'exploration_bias': 0.001, # Exploration bonus (0.0001-0.01)
}
```

**Reward Function**: `reward = alpha1*log_returns - alpha2*mdd - alpha3*vol + exploration_bias`

### Regime Indicators Configuration

```python
regime_config = {
    'enabled': True,           # Enable/disable regime indicators
    'window': 6,               # SMA window for regime detection (3-12 months)
}
```

### RL Algorithm Hyperparameters

```python
# PPO Configuration
ppo_config = {
    'learning_rate': 3e-4,     # Learning rate (1e-4 to 5e-4)
    'n_steps': 2048,           # Steps per update (1024-4096)
    'batch_size': 64,          # Batch size (32-256)
    'n_epochs': 10,            # Epochs per update (5-20)
    'ent_coef': 0.01,          # Entropy coefficient (0.0-0.1)
}

# SAC Configuration
sac_config = {
    'learning_rate': 3e-4,     # Learning rate (1e-4 to 5e-4)
    'buffer_size': 20000,      # Replay buffer size (10000-100000)
    'learning_starts': 100,    # Steps before learning (50-200)
    'batch_size': 128,         # Batch size (64-256)
    'ent_coef': 'auto',        # Entropy coefficient ('auto' or 0.05-0.5)
}
```

### Training Configuration

```python
training_config = {
    'train_ratio': 0.60,       # Training data proportion (0.5-0.7)
    'val_ratio': 0.20,         # Validation data proportion (0.15-0.25)
    'timesteps': 100000,       # Total training timesteps (50k-500k)
    'algorithm': 'both',       # Algorithm: 'ppo', 'sac', or 'both'
}
```

## Risk Profiles Quick Reference

### Conservative Profile

**Goal**: Minimize risk, stable returns

```python
sentiment_env_config = {'vol_window': 10, 'transaction_cost': 0.002}
technical_env_config = {'vol_window': 10}
super_agent_config = {'alpha1': 1.0, 'alpha2': 1.0, 'alpha3': 1.0, 'exploration_bias': 0.005}
meta_agent_config = {'alpha1': 1.0, 'alpha2': 1.0, 'alpha3': 1.0, 'exploration_bias': 0.0005}
regime_config = {'enabled': True, 'window': 12}
ppo_config = {'learning_rate': 1e-4, 'ent_coef': 0.005}
training_config = {'timesteps': 150000}
```

**Characteristics**:
- High penalties for drawdown and volatility
- Longer windows for smoothness
- Lower exploration
- More training for stability

### Aggressive Profile

**Goal**: Maximize returns, accept higher risk

```python
sentiment_env_config = {'vol_window': 3, 'transaction_cost': 0.0005}
technical_env_config = {'vol_window': 3}
super_agent_config = {'alpha1': 1.5, 'alpha2': 0.2, 'alpha3': 0.2, 'exploration_bias': 0.02}
meta_agent_config = {'alpha1': 1.5, 'alpha2': 0.2, 'alpha3': 0.2, 'exploration_bias': 0.002}
regime_config = {'enabled': True, 'window': 3}
ppo_config = {'learning_rate': 5e-4, 'ent_coef': 0.02}
training_config = {'timesteps': 100000}
```

**Characteristics**:
- Low penalties for drawdown and volatility
- Emphasize returns (alpha1 = 1.5)
- Shorter windows for fast response
- Higher exploration
- Lower transaction costs

### Balanced Profile (Default)

**Goal**: Balance risk and returns

```python
sentiment_env_config = {'vol_window': 3, 'transaction_cost': 0.001}
technical_env_config = {'vol_window': 3}
super_agent_config = {'alpha1': 1.0, 'alpha2': 0.5, 'alpha3': 0.5, 'exploration_bias': 0.01}
meta_agent_config = {'alpha1': 1.0, 'alpha2': 0.5, 'alpha3': 0.5, 'exploration_bias': 0.001}
regime_config = {'enabled': True, 'window': 6}
ppo_config = {'learning_rate': 3e-4, 'ent_coef': 0.01}
training_config = {'timesteps': 100000}
```

**Characteristics**:
- Balanced risk-return tradeoff
- Medium windows
- Moderate exploration
- Standard training time

## Parameter Tuning Guidelines

### vol_window (Base Agents)
- **3-5**: Fast response, more volatile signals
- **6-8**: Balanced (recommended)
- **10-15**: Smooth, stable signals

### alpha1, alpha2, alpha3 (Hierarchical Agents)
- **alpha1** (returns weight): 
  - Conservative: 1.0
  - Balanced: 1.0-1.2
  - Aggressive: 1.5-2.0
  
- **alpha2** (drawdown penalty):
  - Conservative: 1.0-1.5
  - Balanced: 0.5-0.7
  - Aggressive: 0.2-0.3
  
- **alpha3** (volatility penalty):
  - Conservative: 1.0-1.5
  - Balanced: 0.5-0.7
  - Aggressive: 0.2-0.3

### exploration_bias
- **Super Agent**: 0.005-0.02 (higher for more exploration)
- **Meta Agent**: 0.0005-0.002 (typically 10x lower than super)

### regime_config.window
- **3-4 months**: Fast regime detection, more noise
- **6-9 months**: Balanced (recommended)
- **12+ months**: Stable regimes, slow to adapt

### ent_coef (Exploration)
- **0.0**: No exploration (pure exploitation)
- **0.005-0.01**: Balanced (recommended for PPO)
- **0.02-0.05**: High exploration
- **'auto'**: Let SAC auto-tune (recommended for SAC)

### timesteps (Training)
- **50,000**: Quick test/debugging
- **100,000**: Standard training
- **200,000**: Better generalization
- **500,000**: Production quality

## Usage Examples

### Example 1: Using Notebook Default

```python
# Cell 3 - Configuration already has defaults
# Just run the cell as-is

# Cell 4 - Load data

# Cell 6 - Create regime indicators
# Uses regime_config from Cell 3

# Cell 7 - Training
# Automatically uses all configs from Cell 3
```

### Example 2: Load Preset Configuration

```python
# Cell 3 - Replace with preset
from experiment_configs import conservative

sentiment_env_config = conservative['sentiment_env_config']
technical_env_config = conservative['technical_env_config']
super_agent_config = conservative['super_agent_config']
meta_agent_config = conservative['meta_agent_config']
regime_config = conservative['regime_config']
ppo_config = conservative['ppo_config']
sac_config = conservative['sac_config']
training_config = conservative['training_config']
```

### Example 3: Custom Configuration

```python
# Cell 3 - Edit specific parameters
sentiment_env_config['vol_window'] = 7
super_agent_config['alpha2'] = 0.8  # More conservative
training_config['timesteps'] = 200000  # More training
```

### Example 4: Mix Profiles

```python
# Cell 3 - Mix and match
from experiment_configs import conservative, aggressive

# Conservative base agents
sentiment_env_config = conservative['sentiment_env_config']
technical_env_config = conservative['technical_env_config']

# But aggressive hierarchical agents
super_agent_config = aggressive['super_agent_config']
meta_agent_config = aggressive['meta_agent_config']
```

## Pre-defined Experiment Profiles

All available in `experiment_configs.py`:

1. **quick_test**: Fast training (50k steps) for debugging
2. **conservative**: Low-risk, stable returns
3. **aggressive**: High-return, accept higher risk
4. **ppo_only**: Only train PPO (faster)
5. **sac_only**: Only train SAC (better for continuous)
6. **production**: Long training (500k steps) for best quality

## Complete Configuration Template

```python
# Copy this to Cell 3 and customize

# Base Agents
sentiment_env_config = {
    'vol_window': 3,
    'transaction_cost': 0.001,
    'risk_lambda': 0.5,
}

technical_env_config = {
    'vol_window': 3,
}

# Hierarchical Agents
super_agent_config = {
    'alpha1': 1.0,
    'alpha2': 0.5,
    'alpha3': 0.5,
    'exploration_bias': 0.01,
}

meta_agent_config = {
    'alpha1': 1.0,
    'alpha2': 0.5,
    'alpha3': 0.5,
    'exploration_bias': 0.001,
}

# Regime Indicators
regime_config = {
    'enabled': True,
    'window': 6,
}

# RL Algorithms
ppo_config = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'ent_coef': 0.01,
}

sac_config = {
    'learning_rate': 3e-4,
    'buffer_size': 20000,
    'learning_starts': 100,
    'batch_size': 128,
    'ent_coef': 'auto',
}

# Training
training_config = {
    'train_ratio': 0.60,
    'val_ratio': 0.20,
    'timesteps': 100000,
    'algorithm': 'both',
}
```

## Summary

All parameters are now in **one place** (Notebook Cell 3):

✓ Base agent reward parameters
✓ Hierarchical agent reward parameters
✓ Regime indicator settings
✓ PPO hyperparameters
✓ SAC hyperparameters
✓ Training configuration

**No need to edit .py files!** Just configure Cell 3 and run.

