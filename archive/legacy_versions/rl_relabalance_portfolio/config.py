"""
Configuration for RL Portfolio Training
Simple parameter definitions - modify as needed
"""
import json
from pathlib import Path
import torch.nn as nn
import numpy as np

# Get project root (parent of config.py directory)
PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# DATA
# ============================================================================

# Set USE_MINIMAL_FEATURES to True to use the evidence-based minimal feature set (22 features)
# Set to False to use the original feature set (100 features)
USE_MINIMAL_FEATURES = True

# Original data paths (100 features)
DATA_PATH_ORIGINAL = PROJECT_ROOT / 'data' / 'processed' / 'train_weekly.parquet'
TEST_PATH_ORIGINAL = PROJECT_ROOT / 'data' / 'processed' / 'test_weekly.parquet'
METADATA_PATH_ORIGINAL = PROJECT_ROOT / 'data' / 'processed' / 'metadata_weekly.json'

# Minimal features paths (22 features - evidence-based, weekly data)
DATA_PATH_MINIMAL = PROJECT_ROOT / 'data' / 'processed' / 'train_weekly.parquet'
TEST_PATH_MINIMAL = PROJECT_ROOT / 'data' / 'processed' / 'test_weekly.parquet'
METADATA_PATH_MINIMAL = PROJECT_ROOT / 'data' / 'processed' / 'metadata_weekly.json'

# Select which data to use
if USE_MINIMAL_FEATURES:
    DATA_PATH = DATA_PATH_MINIMAL
    TEST_PATH = TEST_PATH_MINIMAL
    METADATA_PATH = METADATA_PATH_MINIMAL
    OUTPUT_DIR = PROJECT_ROOT / 'models'  # Unified models directory
else:
    DATA_PATH = DATA_PATH_ORIGINAL
    TEST_PATH = TEST_PATH_ORIGINAL
    METADATA_PATH = METADATA_PATH_ORIGINAL
    OUTPUT_DIR = PROJECT_ROOT / 'models'  # Unified models directory

DATE_COL = 'date'

# Data date range
START_DATE = '2015-01-01'
END_DATE = '2025-11-01'

# Tickers (7 tech stocks - weekly 2015-2024)
TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']

# Features - load from metadata
try:
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        FEATURE_COLS = metadata['feature_cols']
        print(f"✓ Loaded {len(FEATURE_COLS)} features from {METADATA_PATH.name}")
        if USE_MINIMAL_FEATURES:
            print(f"  Using MINIMAL feature set (evidence-based)")
        else:
            print(f"  Using ORIGINAL feature set")
except FileNotFoundError:
    # Fallback if metadata doesn't exist yet
    FEATURE_COLS = []
    print(f"Warning: {METADATA_PATH} not found. FEATURE_COLS will be empty.")


# ============================================================================
# ENVIRONMENT
# ============================================================================

# Reward
REWARD_TYPE = 'log_return'  # Simple log return: log(V_t+1) - log(V_t)
REWARD_LOOKBACK = 12             # Weeks (12 = 3 months for weekly data)

# Trading
REBALANCE_FREQUENCY = 4          # Weeks (4 = monthly)
TRANSACTION_COST = 0.0025        # 0.25% = 25 bps
MAX_WEIGHT_PER_ASSET = 0.40      # 40% max per asset

# Portfolio
INITIAL_CAPITAL = 100000.0       # $100k
RANDOM_SEED = 42

# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

N_FOLDS = 3                      # Number of folds
MIN_TRAIN_SIZE = None            # None = auto (20% of data)

# ============================================================================
# PPO TRAINING
# ============================================================================

# Duration - Evidence-based from literature
TOTAL_TIMESTEPS = 100000          # Standard training duration
EVAL_FREQ = 5000                  # NEW: Less frequent (give more time to learn before checking)
N_EVAL_EPISODES = 10
EARLY_STOP_PATIENCE = 3           # NEW: More aggressive stopping (was 5)

# Risk Management Parameters
MIN_ENTROPY = 0.5           # Minimum portfolio entropy (for multi_component reward)
MAX_TURNOVER = 0.20         # Maximum acceptable turnover (for multi_component reward)
ANNUALIZATION_FACTOR = 52   # 52 for weekly, 252 for daily, 12 for monthly
RISK_FREE_RATE = 0.04       # Annual risk-free rate for Sharpe calculation (4%)


# Hyperparameters - Optimized to reduce overfitting
LEARNING_RATE = 0.0001           # Reduced from 3e-4 - slower, more stable
# Cosine annealing learning rate schedule
LEARNING_RATE_SCHEDULE = lambda progress: 0.0001 * (0.5 * (1 + np.cos(np.pi * progress)))
N_STEPS = 1024                   # Reduced from 2048 - smaller batches
BATCH_SIZE = 64                  # Reduced from 64 - better generalization
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01                  # Increased from 0.01 - more exploration
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Network architecture - MLP
POLICY_KWARGS = {
    'net_arch': [256, 256],
    'activation_fn': nn.Tanh
}

# ============================================================================
# LSTM PPO CONFIGURATION
# ============================================================================

# LSTM architecture - Literature-backed (Jiang 2017, Portfolio LSTM 2024)
# Research shows successful papers use 50-75 units, not 100
# Trial 2 config: LSTM=50 performed best (Sharpe 1.097, 2/3 folds beat baseline)
# Trial 5 (PROPER): LSTM=50 + EVAL_FREQ=5000 + PATIENCE=3 (testing early stopping strategy)
LSTM_HIDDEN_SIZE = 50            # 
N_LSTM_LAYERS = 1                # Number of LSTM layers (standard for weekly data)
SHARED_LSTM = True               # Share LSTM between actor and critic
ENABLE_CRITIC_LSTM = False       # Must be False when shared_lstm=True

# LSTM-specific training parameters - Evidence-based
LSTM_LEARNING_RATE = 0.0001      # Standard for LSTM (Jiang 2017)
LSTM_N_STEPS = 128               # RecurrentPPO constraint (memory)
LSTM_BATCH_SIZE = 64             # Standard batch size
LSTM_N_EPOCHS = 5                # Fewer epochs for recurrent (prevents overfitting)
LSTM_ENT_COEF = 0.01             # Standard entropy coefficient

# Network architecture for LSTM policy
LSTM_POLICY_KWARGS = {
    'lstm_hidden_size': LSTM_HIDDEN_SIZE,
    'n_lstm_layers': N_LSTM_LAYERS,
    'shared_lstm': SHARED_LSTM,
    'enable_critic_lstm': ENABLE_CRITIC_LSTM,
    'net_arch': {
        'pi': [256],  # Actor network after LSTM
        'vf': [256]   # Critic network after LSTM
    }
}

# ============================================================================
# OUTPUT

VERBOSE = 0  # 0 = silent, 1 = info, 2 = debug
SAVE_BEST_MODEL = True
SAVE_ALL_MODELS = True
USE_TENSORBOARD = True
TENSORBOARD_LOG = './logs'
