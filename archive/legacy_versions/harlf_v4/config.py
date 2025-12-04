"""
config.py
Configuration for RL Trading System with Weekly Data
"""

# =============================================================================
# PORTFOLIO CONFIGURATION
# =============================================================================

PORTFOLIO = {
   # 'RDDT': 200,
    'NVDA': 133,
    'MU': 50,
    'APP': 16,
    'SMR': 145,
    'AMD': 30,
    'ASML': 6,
    'MSFT': 20,
    'GOOG': 16,
    'AI': 315,
    'ARBE': 3050,
}

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DATA_CONFIG = {
    'start_date': '2020-01-01',
    'end_date': None,  # None = today
    'frequency': 'weekly',  # 'weekly' or 'monthly'
    'interval': '1wk',  # Yahoo Finance interval
    
    # Technical indicator lookbacks (in weeks)
    'lookback_short': 5,   # ~1 month
    'lookback_medium': 13,  # ~3 months
    'lookback_long': 26,    # ~6 months
    
    # File paths
    'price_file': 'data/prepared_prices_weekly.csv',
    'features_file': 'data/prepared_features_weekly.csv',
    'summary_file': 'data/data_summary.json',
}

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

ENV_CONFIG = {
    'max_position': 0.60,           # 60% max position (balanced)
    'transaction_cost': 0.003,      # 30 bps per trade (realistic with slippage)
    'use_log_returns': True,
    'initial_balance': 100_000.0,
    'annualization_factor': 52,     # 52 for weekly, 12 for monthly
    
    # Reward function options: 'simple', 'differential_sharpe', 'sortino', 'composite'
    'reward_function': 'differential_sharpe',  # Composite with turnover penalty
    'reward_scaling': 1.0,
    'reward_lookback': 52,          # 1 year lookback (stable Sharpe)
    
    # Composite reward weights (BALANCED to reduce overtrading)
    'composite_return_weight': 3.0,      # Base weight for returns
    'composite_risk_weight': 0.3,        # Penalty for volatility
    'composite_drawdown_weight': 0.5,    # Penalty for drawdowns
    'composite_turnover_weight': 0.8,    # STRONG penalty for trading (8x from 0.1)
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    'algorithm': 'SAC',  # 'SAC', 'TD3', or 'PPO'
    
    # RL hyperparameters
    'gamma': 0.95,                  # Discount factor (higher for weekly)
    'learning_rate': 3e-4,
    'buffer_size': 50_000,          # REDUCED from 50k (enough for weekly data)
    'batch_size': 512,              # INCREASED for faster, more stable learning
    'tau': 0.01,                    # INCREASED from 0.005 (faster target updates)
    'network_arch': [256, 256],     # Good capacity for 24 features
    
    # Training parameters
    'total_training_steps': 50_000,  # REDUCED from 150k (sufficient for weekly)
    'steps_per_epoch': 5_000,        # Validation every 5k steps
    'early_stopping_patience': 8,    # Allow 8 epochs without improvement
    
    # Model save paths
    'model_dir': 'models',
    'checkpoint_dir': 'models/checkpoints',
    'log_dir': 'models/logs',
}

# =============================================================================
# WALK-FORWARD VALIDATION CONFIGURATION
# =============================================================================

WALKFORWARD_CONFIG = {
    # Window sizes (in weeks)
    'train_weeks': 156,     # REDUCED from 156: ~2 years (faster training)
    'val_weeks': 52,        # REDUCED from 52: ~6 months (faster validation)
    'test_weeks': 26,       # ~6 months (unchanged)
    'step_size': 26,        # INCREASED from 26: ~1 year (fewer windows, faster)
    'test_gap_weeks': 4,    # ~1 month gap to prevent leakage
    
    # Results
    'results_dir': 'results',
}

# =============================================================================
# ANNUALIZATION FACTORS
# =============================================================================

ANNUALIZATION = {
    'weekly': 52,
    'monthly': 12,
    'daily': 252,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_annualization_factor():
    """Get annualization factor based on data frequency"""
    return ANNUALIZATION[DATA_CONFIG['frequency']]

def get_model_hyperparams():
    """
    Extract only the hyperparameters needed for SAC/TD3/PPO initialization.
    Filters out training control parameters.
    """
    return {
        'learning_rate': MODEL_CONFIG['learning_rate'],
        'buffer_size': MODEL_CONFIG['buffer_size'],
        'batch_size': MODEL_CONFIG['batch_size'],
        'gamma': MODEL_CONFIG['gamma'],
        'tau': MODEL_CONFIG['tau'],
        'policy_kwargs': {'net_arch': MODEL_CONFIG['network_arch']},
    }

def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    
    print("\nPortfolio:")
    for ticker, shares in sorted(PORTFOLIO.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ticker:8s}: {shares:4d} shares")
    
    print(f"\nData Configuration:")
    for key, value in DATA_CONFIG.items():
        print(f"  {key:20s}: {value}")
    
    print(f"\nEnvironment Configuration:")
    for key, value in ENV_CONFIG.items():
        print(f"  {key:25s}: {value}")
    
    print(f"\nModel Configuration:")
    for key, value in MODEL_CONFIG.items():
        if key not in ['model_dir', 'checkpoint_dir', 'log_dir']:
            print(f"  {key:25s}: {value}")
    
    print(f"\nWalk-Forward Configuration:")
    for key, value in WALKFORWARD_CONFIG.items():
        if key != 'results_dir':
            print(f"  {key:20s}: {value}")
    
    print("=" * 80)


if __name__ == "__main__":
    print_config()