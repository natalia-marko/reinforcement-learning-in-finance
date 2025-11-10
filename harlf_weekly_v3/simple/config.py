# ======================
# Data paths
# ======================
DATA_DIR = '/Users/nataliamarko/Documents/GitHub_Projects/reinforcement_learning_in_finance/harlf_weekly_v3/data_hierarchical'
MODELS_DIR = '/Users/nataliamarko/Documents/GitHub_Projects/reinforcement_learning_in_finance/harlf_weekly_v3/models'

# ======================
# Training settings
# ======================
TOTAL_STEPS = 300_000
EVAL_FREQ = 5_000
PATIENCE = 15
LEARNING_RATE = 3e-4
GAMMA = 0.90
SEED = 42
SOFTMAX_TEMPERATURE = 1.0
RANDOM_START = True
TRANSACTION_COST = 0.0025

# ======================
# Algorithms (simplified to PPO only)
# ======================
ALGORITHMS = ['PPO']

# ======================
# EMA Sharpe config
# ======================
EMA_CONFIG = {
    'rolling_vol_window': 12
}

# ======================
# Multi-Objective configs
# ======================
MULTI_TECHNICAL = {
    'return_scale': 8.0,
    'volatility_penalty': 0.050,
    'concentration_penalty': 0.250,
    'turnover_penalty': 0.005,
    'vol_window': 12,
    'max_concentration': 0.35
}

MULTI_SENTIMENT = {
    'return_scale': 3.0,
    'volatility_penalty': 0.2,
    'concentration_penalty': 1.0,
    'turnover_penalty': 0.02,
    'vol_window': 12,
    'max_concentration': 0.35
}

def get_config(reward_type='ema_sharpe', agent_type='technical'):
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
        'transaction_cost': TRANSACTION_COST,
    }
    if reward_type == 'ema_sharpe':
        config.update(EMA_CONFIG)
    elif reward_type == 'multi_objective':
        if agent_type == 'technical':
            config.update(MULTI_TECHNICAL)
        else:
            config.update(MULTI_SENTIMENT)
    return config
