"""
Configuration
====================

CHANGES IN THIS VERSION:
- SOFTMAX_TEMPERATURE: 3.0 → 1.0 (more decisive actions)
- GAMMA: 0.99 → 0.90 (appropriate for weekly rebalancing)
- PATIENCE: 5 → 15 (less aggressive early stopping)
- Added TRANSACTION_COST parameter
- Made max_concentration configurable for multi-objective
"""

# Data paths
DATA_DIR = '../data_hierarchical'
MODELS_DIR = '../models'

# Training settings
TOTAL_STEPS = 300_000      # How long to train
EVAL_FREQ = 5_000          # Evaluate every N steps
PATIENCE = 15              # Early stopping patience (CHANGED from 5)
LEARNING_RATE = 3e-4       # Learning rate
GAMMA = 0.90               # Discount factor (CHANGED from 0.99 - better for weekly)
SEED = 42                  # Random seed

# Environment settings
SOFTMAX_TEMPERATURE = 1.0  # Temperature for action->weights (CHANGED from 3.0)
RANDOM_START = True        # Random starting positions in training
TRANSACTION_COST = 0.0025  

# Algorithms to test
ALGORITHMS = ['PPO', 'SAC', 'A2C']

# ============================================================================
# Reward-specific configs (you can edit these!)
# ============================================================================

# EMA Sharpe config
EMA_CONFIG = {
    'rolling_vol_window': 12  # Window for EMA
}

# Differential Sharpe config
DIFF_CONFIG = {
    'decay_factor': 0.95  # Memory length (~20 steps)
}

# Multi-Objective configs (different for technical vs sentiment)
# NOTE: Consider using same values for both initially, then tune separately
MULTI_TECHNICAL = {
    'return_scale': 8.0,              # Higher = chase returns
    'volatility_penalty': 0.050,      # Penalize volatility
    'concentration_penalty': 0.250,   # Force diversification
    'turnover_penalty': 0.005,        # Penalize trading
    'vol_window': 12,
    'max_concentration': 0.30         # NEW: Max weight threshold
}

MULTI_SENTIMENT = {
    'return_scale': 3.0,              # More conservative
    'volatility_penalty': 0.2,        # Stronger risk control
    'concentration_penalty': 1.0,     # Force diversification
    'turnover_penalty': 0.02,         # Minimize trading
    'vol_window': 12,
    'max_concentration': 0.30         # NEW: Max weight threshold
}

# ============================================================================
# Helper function to get config
# ============================================================================

def get_config(reward_type='ema_sharpe', agent_type='technical'):
    """
    Get configuration for training.
    
    Parameters
    ----------
    reward_type : str
        'ema_sharpe', 'differential_sharpe', 'multi_objective', 
        'simple_return', or 'simple_sharpe'
    agent_type : str
        'technical' or 'sentiment' (only matters for multi_objective)
    
    Returns
    -------
    dict : Configuration dictionary
    
    Examples
    --------
    >>> config = get_config('ema_sharpe')
    >>> config = get_config('multi_objective', 'technical')
    """
    
    # Base config
    config = {
        'data_dir': DATA_DIR,
        'models_dir': MODELS_DIR,
        'total_steps': TOTAL_STEPS,
        'eval_freq': EVAL_FREQ,
        'patience': PATIENCE,
        'learning_rate': LEARNING_RATE,
        'gamma': GAMMA,
        'softmax_temperature': SOFTMAX_TEMPERATURE,
        'random_start': RANDOM_START,
        'seed': SEED,
        'transaction_cost': TRANSACTION_COST,  # NEW
    }
    
    # Add reward-specific params
    if reward_type == 'ema_sharpe':
        config.update(EMA_CONFIG)
    elif reward_type == 'differential_sharpe':
        config.update(DIFF_CONFIG)
    elif reward_type == 'multi_objective':
        if agent_type == 'technical':
            config.update(MULTI_TECHNICAL)
        else:
            config.update(MULTI_SENTIMENT)
    
    return config


if __name__ == '__main__':
    print("Configuration loaded!")
    print(f"\nData directory: {DATA_DIR}")
    print(f"Total steps: {TOTAL_STEPS:,}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"\n⚠️  REFACTORED VERSION - Key changes:")
    print(f"  - Softmax temperature: 1.0 (was 3.0)")
    print(f"  - Gamma: 0.90 (was 0.99)")
    print(f"  - Patience: 15 (was 5)")
    print(f"  - Transaction cost: {TRANSACTION_COST*10000:.1f} bps")