# Configuration Guide

## Overview

All parameters are now configurable from the notebook for easy experimentation. No need to edit .py files anymore!

## What Can You Configure?

### 1. Environment Reward Parameters

#### Sentiment Environment
```python
sentiment_env_config = {
    'vol_window': 3,           # Rolling window for Sharpe-like reward
    'transaction_cost': 0.001, # Transaction cost per trade (0.1%)
    'risk_lambda': 0.5,        # Risk penalty weight
}
```

**Parameters explained**:
- `vol_window`: Number of recent periods to calculate volatility
  - Smaller (3-5): Faster response, more reactive
  - Larger (10-15): Smoother signals, more stable
- `transaction_cost`: Cost per rebalance (0.001 = 0.1%)
  - Lower: Encourages more trading
  - Higher: Encourages less trading
- `risk_lambda`: Penalty weight for volatility (legacy parameter)

#### Technical Environment
```python
technical_env_config = {
    'vol_window': 3,  # Rolling window for Sharpe-like reward
}
```

### 2. RL Algorithm Hyperparameters

#### PPO (Proximal Policy Optimization)
```python
ppo_config = {
    'learning_rate': 3e-4,     # How fast the model learns
    'n_steps': 2048,           # Steps collected before update
    'batch_size': 64,          # Mini-batch size
    'n_epochs': 10,            # Training epochs per update
    'ent_coef': 0.01,          # Exploration bonus
}
```

**Key parameters**:
- `learning_rate`: Lower = more stable but slower; Higher = faster but riskier
- `ent_coef`: Controls exploration
  - 0.0: No exploration (pure exploitation)
  - 0.01-0.02: Balanced
  - 0.05-0.1: High exploration

#### SAC (Soft Actor-Critic)
```python
sac_config = {
    'learning_rate': 3e-4,     # How fast the model learns
    'buffer_size': 20000,      # Replay buffer size
    'learning_starts': 100,    # Steps before learning starts
    'batch_size': 128,         # Batch size for training
    'ent_coef': 'auto',        # Auto-tune exploration
}
```

**Key parameters**:
- `buffer_size`: Larger = more diverse experiences but more memory
- `ent_coef`: 
  - 'auto': Let SAC figure it out (recommended)
  - Float (0.1-0.5): Manual control

### 3. Training Configuration
```python
training_config = {
    'train_ratio': 0.60,       # % of data for training
    'val_ratio': 0.20,         # % of data for validation
    'timesteps': 100000,       # Total training steps
    'algorithm': 'both',       # 'ppo', 'sac', or 'both'
}
```

## How to Use in Notebook

### Method 1: Direct Configuration (Current Setup)

Just edit the values in the configuration cell:

```python
# Cell 3 in notebook
sentiment_env_config = {
    'vol_window': 5,  # Change from 3 to 5
    'transaction_cost': 0.002,  # Change from 0.001 to 0.002
}

training_config = {
    'timesteps': 200000,  # Change from 100000 to 200000
    'algorithm': 'ppo',   # Change from 'both' to 'ppo'
}
```

Then run the training cell - it will automatically use these values!

### Method 2: Load Preset Configurations

Use the pre-defined experiment configurations:

```python
# At the top of your notebook
from experiment_configs import conservative, aggressive, production

# Load a preset
config = conservative

# Unpack it
sentiment_env_config = config['sentiment_env_config']
technical_env_config = config['technical_env_config']
ppo_config = config['ppo_config']
sac_config = config['sac_config']
training_config = config['training_config']
```

### Method 3: Mix and Match

Combine different presets:

```python
from experiment_configs import conservative, aggressive

sentiment_env_config = conservative['sentiment_env_config']  # Conservative sentiment
technical_env_config = aggressive['technical_env_config']    # Aggressive technical
ppo_config = production['ppo_config']                        # Production quality PPO
```

## Pre-defined Experiment Profiles

Located in `experiment_configs.py`:

1. **`quick_test`**: Fast training for debugging (50k steps)
2. **`conservative`**: Low-risk focus, longer windows, more training
3. **`aggressive`**: High-return focus, shorter windows, more exploration
4. **`ppo_only`**: Train only PPO (faster)
5. **`sac_only`**: Train only SAC (better for continuous actions)
6. **`production`**: Long training for best results (500k steps)

## Tuning Tips

### If your agents are not learning:
- Increase `timesteps` (100k → 200k)
- Increase `ent_coef` for more exploration (0.01 → 0.02)
- Try different `learning_rate` (3e-4 → 5e-4 or 1e-4)

### If your agents are too volatile:
- Increase `vol_window` (3 → 7 or 10)
- Decrease `learning_rate` (3e-4 → 1e-4)
- Decrease `ent_coef` (0.01 → 0.005)

### If training is too slow:
- Use `algorithm='ppo'` instead of 'both'
- Decrease `timesteps` (100k → 50k)
- Decrease `buffer_size` for SAC (20k → 10k)

### For better generalization:
- Increase `vol_window` for smoother signals
- Use longer training (`timesteps` = 200k-500k)
- Try `conservative` or `production` configs

## Example Workflow

```python
# 1. Quick experiment to test if it works
from experiment_configs import quick_test
# ... load quick_test config ...
# Run training (takes ~10 minutes)

# 2. If it works, try a longer run
training_config['timesteps'] = 200000
# Run again (~30-40 minutes)

# 3. Compare different profiles
# Try conservative, aggressive, and production
# See which performs best on validation set

# 4. Fine-tune the winner
# Adjust specific parameters based on results
sentiment_env_config['vol_window'] = 8  # Custom value
ppo_config['ent_coef'] = 0.015          # Custom value
# Final training run
```

## Comparing Results

After training, compare using validation Sharpe ratios:

```python
for name, res in results.items():
    val_sharpe = res['val_metrics']['sharpe_ratio']
    test_sharpe = res['test_metrics']['sharpe_ratio']
    print(f"{name}: Val Sharpe = {val_sharpe:.3f}, Test Sharpe = {test_sharpe:.3f}")
```

Good signs:
- Validation Sharpe > 1.0
- Test Sharpe close to Validation Sharpe (not much overfitting)
- Positive returns with low drawdowns

## Summary

**You can now**:
- Change all parameters in one place (notebook cell 3)
- No need to edit .py files anymore
- Try different configurations quickly
- Use pre-defined experiment profiles
- Mix and match configurations

**Files involved**:
- `harlf_system.ipynb`: Your main notebook with config cell
- `agent_wrapper.py`: Updated to accept parameters
- `experiment_configs.py`: Pre-defined configurations
- `reward_config_reference.py`: Parameter reference

**Next steps**:
1. Run with default config to establish baseline
2. Try different presets from `experiment_configs.py`
3. Fine-tune based on results
4. Document your best configuration for future use

