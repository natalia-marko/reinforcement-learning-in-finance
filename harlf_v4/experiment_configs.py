"""
Example Experiment Configurations

Copy these into your notebook to try different setups.
"""

# Regime Indicator Configuration
# Use with Super and Meta agents only
regime_config = {
    'enabled': True,        # Enable/disable regime indicators
    'window': 6,            # SMA window (months) for regime detection
}

# Experiment 1: Quick Test (Fast Training)
quick_test = {
    'sentiment_env_config': {
        'vol_window': 3,
        'transaction_cost': 0.01,
    },
    'technical_env_config': {
        'vol_window': 3,
    },
    'super_agent_config': {
        'alpha1': 1.0,
        'alpha2': 0.5,
        'alpha3': 0.5,
        'exploration_bias': 0.01,
    },
    'meta_agent_config': {
        'alpha1': 1.0,
        'alpha2': 0.5,
        'alpha3': 0.5,
        'exploration_bias': 0.01,
    },
    'regime_config': {
        'enabled': True,
        'window': 6,
    },
    'ppo_config': {
        'learning_rate': 3e-4,
        'ent_coef': 0.02,  # More exploration
    },
    'sac_config': {
        'learning_rate': 3e-4,
        'ent_coef': 'auto',
    },
    'training_config': {
        'timesteps': 50000,  # Faster
        'algorithm': 'both',
    }
}

# Experiment 2: Conservative (Low Risk Focus)
conservative = {
    'sentiment_env_config': {
        'vol_window': 10,      # Longer window = smoother signals
        'transaction_cost': 0.002,  # Higher cost = less trading
        'risk_lambda': 1.0,    # More risk penalty
    },
    'technical_env_config': {
        'vol_window': 10,
    },
    'super_agent_config': {
        'alpha1': 1.0,
        'alpha2': 1.0,          # Higher drawdown penalty
        'alpha3': 1.0,          # Higher volatility penalty
        'exploration_bias': 0.005,
    },
    'meta_agent_config': {
        'alpha1': 1.0,
        'alpha2': 1.0,          # Higher drawdown penalty
        'alpha3': 1.0,          # Higher volatility penalty
        'exploration_bias': 0.0005,
    },
    'regime_config': {
        'enabled': True,
        'window': 12,           # Longer window for stable regimes
    },
    'ppo_config': {
        'learning_rate': 1e-4,  # Lower learning rate
        'ent_coef': 0.005,      # Less exploration
    },
    'sac_config': {
        'learning_rate': 1e-4,
        'ent_coef': 0.05,       # Less exploration
    },
    'training_config': {
        'timesteps': 150000,    # More training
        'algorithm': 'both',
    }
}

# Experiment 3: Aggressive (High Returns Focus)
aggressive = {
    'sentiment_env_config': {
        'vol_window': 3,        # Shorter window = faster response
        'transaction_cost': 0.0005,  # Lower cost = more trading
        'risk_lambda': 0.2,     # Less risk penalty
    },
    'technical_env_config': {
        'vol_window': 3,
    },
    'super_agent_config': {
        'alpha1': 1.5,          # Emphasize returns more
        'alpha2': 0.2,          # Lower drawdown penalty
        'alpha3': 0.2,          # Lower volatility penalty
        'exploration_bias': 0.02,
    },
    'meta_agent_config': {
        'alpha1': 1.5,          # Emphasize returns more
        'alpha2': 0.2,          # Lower drawdown penalty
        'alpha3': 0.2,          # Lower volatility penalty
        'exploration_bias': 0.002,
    },
    'regime_config': {
        'enabled': True,
        'window': 3,            # Shorter window for faster response
    },
    'ppo_config': {
        'learning_rate': 5e-4,  # Higher learning rate
        'ent_coef': 0.02,       # More exploration
    },
    'sac_config': {
        'learning_rate': 5e-4,
        'ent_coef': 'auto',
    },
    'training_config': {
        'timesteps': 100000,
        'algorithm': 'both',
    }
}

# Experiment 4: PPO Only (Faster Training)
ppo_only = {
    'sentiment_env_config': {
        'vol_window': 5,
        'transaction_cost': 0.001,
    },
    'technical_env_config': {
        'vol_window': 5,
    },
    'super_agent_config': {
        'alpha1': 1.0,
        'alpha2': 0.5,
        'alpha3': 0.5,
        'exploration_bias': 0.01,
    },
    'meta_agent_config': {
        'alpha1': 1.0,
        'alpha2': 0.5,
        'alpha3': 0.5,
        'exploration_bias': 0.001,
    },
    'regime_config': {
        'enabled': True,
        'window': 6,
    },
    'ppo_config': {
        'learning_rate': 3e-4,
        'n_steps': 4096,        # More steps per update
        'batch_size': 128,      # Larger batch
        'ent_coef': 0.01,
    },
    'sac_config': None,  # Not used
    'training_config': {
        'timesteps': 100000,
        'algorithm': 'ppo',     # Only PPO
    }
}

# Experiment 5: SAC Only (Better for Continuous Actions)
sac_only = {
    'sentiment_env_config': {
        'vol_window': 5,
        'transaction_cost': 0.001,
    },
    'technical_env_config': {
        'vol_window': 5,
    },
    'super_agent_config': {
        'alpha1': 1.0,
        'alpha2': 0.5,
        'alpha3': 0.5,
        'exploration_bias': 0.01,
    },
    'meta_agent_config': {
        'alpha1': 1.0,
        'alpha2': 0.5,
        'alpha3': 0.5,
        'exploration_bias': 0.001,
    },
    'regime_config': {
        'enabled': True,
        'window': 6,
    },
    'ppo_config': None,  # Not used
    'sac_config': {
        'learning_rate': 3e-4,
        'buffer_size': 50000,   # Larger buffer
        'batch_size': 256,      # Larger batch
        'ent_coef': 'auto',
    },
    'training_config': {
        'timesteps': 100000,
        'algorithm': 'sac',     # Only SAC
    }
}

# Experiment 6: Long Training (Production Quality)
production = {
    'sentiment_env_config': {
        'vol_window': 7,
        'transaction_cost': 0.001,
        'risk_lambda': 0.5,
    },
    'technical_env_config': {
        'vol_window': 7,
    },
    'super_agent_config': {
        'alpha1': 1.0,
        'alpha2': 0.7,          # Slightly more conservative
        'alpha3': 0.7,
        'exploration_bias': 0.005,
    },
    'meta_agent_config': {
        'alpha1': 1.0,
        'alpha2': 0.7,
        'alpha3': 0.7,
        'exploration_bias': 0.0005,
    },
    'regime_config': {
        'enabled': True,
        'window': 9,            # Longer window for stability
    },
    'ppo_config': {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'ent_coef': 0.01,
    },
    'sac_config': {
        'learning_rate': 3e-4,
        'buffer_size': 100000,  # Much larger buffer
        'batch_size': 256,
        'ent_coef': 'auto',
    },
    'training_config': {
        'timesteps': 500000,    # Much more training
        'algorithm': 'both',
    }
}


# How to use in notebook:
"""
# Option 1: Load a preset configuration
config = conservative  # or aggressive, production, etc.

sentiment_env_config = config['sentiment_env_config']
technical_env_config = config['technical_env_config']
ppo_config = config['ppo_config']
sac_config = config['sac_config']
training_config = config['training_config']

# Option 2: Mix and match
sentiment_env_config = conservative['sentiment_env_config']
technical_env_config = aggressive['technical_env_config']
ppo_config = production['ppo_config']
sac_config = sac_only['sac_config']
training_config = quick_test['training_config']

# Option 3: Custom tweaks
sentiment_env_config = {
    'vol_window': 8,  # Your custom value
    'transaction_cost': 0.0015,
}
"""

