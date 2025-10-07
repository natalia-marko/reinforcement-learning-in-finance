# What's New - Configuration System Update

## Summary

All parameters are now configurable from the notebook! No more editing .py files to change reward functions or RL hyperparameters.

## Changes Made

### 1. Refactored Reward Functions

#### Base Agents (Sentiment & Technical)
- **Old**: Simple log returns as reward
- **New**: Sharpe-like ratio (mean_return / volatility)
- **Benefit**: Encourages consistent, risk-adjusted performance

#### Super & Meta Agents
- **Old**: Simple portfolio returns
- **New**: Complex multi-component reward
  ```
  reward = alpha1*log_returns - alpha2*mdd - alpha3*vol + exploration_bias
  ```
- **Benefit**: Balances returns, risk, and exploration

### 2. Centralized Configuration

#### Updated Files
- **`agent_wrapper.py`**
  - `train_agent()` now accepts `ppo_params` and `sac_params`
  - `train_and_evaluate_with_split()` now accepts:
    - `sentiment_env_params`
    - `technical_env_params`
    - `ppo_params`
    - `sac_params`

- **`harlf_system.ipynb`**
  - Added new **Configuration Cell** (Cell 3)
  - All parameters defined in one place
  - Training cell automatically uses these configs

- **Environment Files** (sentiment_enviroment.py, technical_enviroment.py, etc.)
  - Added parameter support to `__init__` methods
  - Maintain backward compatibility with defaults

### 3. New Files Created

#### `CONFIGURATION_GUIDE.md`
Comprehensive guide explaining:
- What each parameter does
- How to configure from notebook
- Tuning tips
- Example workflows

#### `experiment_configs.py`
Pre-defined configurations:
- `quick_test`: Fast debugging (50k steps)
- `conservative`: Low-risk profile
- `aggressive`: High-return profile
- `ppo_only`: PPO algorithm only
- `sac_only`: SAC algorithm only
- `production`: Production-quality (500k steps)

#### `reward_config_reference.py`
Quick reference for parameter values

#### `reward_function_refactoring_summary.md`
Technical details on reward function changes

## How to Use

### Quick Start

1. Open `harlf_system.ipynb`
2. Edit Cell 3 (Configuration) with your desired parameters
3. Run all cells
4. Done!

### Example: Change Vol Window

```python
# In notebook Cell 3
sentiment_env_config = {
    'vol_window': 7,  # Changed from 3 to 7
    'transaction_cost': 0.001,
}
```

### Example: Use Preset Configuration

```python
# In notebook Cell 3
from experiment_configs import conservative

# Load conservative profile
sentiment_env_config = conservative['sentiment_env_config']
technical_env_config = conservative['technical_env_config']
ppo_config = conservative['ppo_config']
sac_config = conservative['sac_config']
training_config = conservative['training_config']
```

## Parameters You Can Now Control

### Environment (Reward Function)
- `vol_window`: Rolling window for risk calculation
- `transaction_cost`: Trading costs
- `risk_lambda`: Risk penalty weight
- `alpha1`, `alpha2`, `alpha3`: Reward component weights (super/meta agents)
- `exploration_bias`: Exploration bonus (super/meta agents)

### PPO Hyperparameters
- `learning_rate`: Learning rate
- `n_steps`: Steps per update
- `batch_size`: Batch size
- `n_epochs`: Epochs per update
- `ent_coef`: Exploration coefficient

### SAC Hyperparameters
- `learning_rate`: Learning rate
- `buffer_size`: Replay buffer size
- `learning_starts`: Steps before learning
- `batch_size`: Batch size
- `ent_coef`: Exploration coefficient

### Training
- `train_ratio`: Training data proportion
- `val_ratio`: Validation data proportion
- `timesteps`: Total training steps
- `algorithm`: 'ppo', 'sac', or 'both'

## Benefits

1. **Easier Experimentation**: Change parameters without editing code
2. **Reproducibility**: Share configurations as simple dictionaries
3. **Faster Iteration**: Try different setups quickly
4. **Better Organization**: All configs in one place
5. **Version Control Friendly**: Track experiments easily

## Migration from Old Code

### If you had:
```python
# Old way - edit .py file
def __init__(self, price_data, features):
    self.vol_window = 10  # Hard-coded
```

### Now you have:
```python
# New way - pass from notebook
sentiment_env_config = {
    'vol_window': 10  # Configure in notebook
}
```

## Files Reference

### Core Files (Modified)
- `agent_wrapper.py`: Updated to accept parameters
- `sentiment_enviroment.py`: Risk-adjusted rewards + parameters
- `technical_enviroment.py`: Risk-adjusted rewards + parameters
- `super_agent_envoriment.py`: Complex rewards + parameters
- `meta_agent_enviroment.py`: Complex rewards + parameters
- `harlf_system.ipynb`: Added configuration cell

### Documentation Files (New)
- `CONFIGURATION_GUIDE.md`: Complete usage guide
- `experiment_configs.py`: Pre-defined configurations
- `reward_config_reference.py`: Parameter reference
- `reward_function_refactoring_summary.md`: Technical details
- `WHATS_NEW.md`: This file

## Next Steps

1. Read `CONFIGURATION_GUIDE.md` for detailed instructions
2. Try the default configuration first
3. Experiment with presets from `experiment_configs.py`
4. Fine-tune based on validation results
5. Document your best configuration

## Backward Compatibility

All changes maintain backward compatibility:
- Default parameter values match previous behavior
- Old code will still work without modifications
- New parameters are optional

## Questions?

See:
- `CONFIGURATION_GUIDE.md` for usage
- `reward_function_refactoring_summary.md` for technical details
- `experiment_configs.py` for examples

