"""
Configuration for RL Portfolio Training
Simple parameter definitions - modify as needed
"""
import json
from pathlib import Path
import torch.nn as nn

# Get project root (parent of config.py directory)
PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# DATA
# ============================================================================

DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'train.parquet'
TEST_PATH = PROJECT_ROOT / 'data' / 'processed' / 'test.parquet'
METADATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'metadata.json'
OUTPUT_DIR = PROJECT_ROOT / 'models'
DATE_COL = 'date'

# Tickers (7 tech stocks - weekly 2020-2024)
TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']

# Features (149 total - loaded from metadata)
try:
    with open(METADATA_PATH, 'r') as f:
        FEATURE_COLS = json.load(f)['feature_cols']
except FileNotFoundError:
    # Fallback if metadata doesn't exist yet
    FEATURE_COLS = []
    print(f"Warning: {METADATA_PATH} not found. FEATURE_COLS will be empty.")

# ============================================================================
# ENVIRONMENT
# ============================================================================

# Reward
REWARD_TYPE = 'multi_component'  # 'log_return', 'simple_return', 'multi_component', 'risk_aware'
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

# Duration
TOTAL_TIMESTEPS = 50000          # Steps per fold
EVAL_FREQ = 2500                 # Evaluate every N steps
N_EVAL_EPISODES = 5
# Risk Management Parameters
VOL_TARGET = 0.35           # Target volatility for risk_aware reward
MIN_ENTROPY = 0.5           # Minimum portfolio entropy (diversification)
MAX_TURNOVER = 0.20         # Maximum acceptable turnover
ANNUALIZATION_FACTOR = 52   # 52 for weekly, 252 for daily, 12 for monthly

# ============================================================================
# REWARD FUNCTION WEIGHTS
# ============================================================================
# All tunable weights and normalization factors for reward functions
# NO MAGIC NUMBERS should appear in rl_system_refactored.py - all here!

REWARD_WEIGHTS = {
    "multi_component": {
        # Component weights (must sum to reasonable total)
        "sharpe": 0.4,              # Weight for Sharpe ratio component
        "return": 0.2,              # Weight for absolute return component
        "drawdown": 0.2,            # Weight for drawdown penalty
        "volatility": 0.2,          # Weight for volatility penalty
        "turnover_weight": 1.0,     # Multiplier for turnover penalty (additive)

        # Normalization factors (divide by these before tanh)
        "sharpe_norm": 3.0,         # Sharpe normalization (typical range: -2 to 2)
        "return_norm": 10.0,        # Return normalization (10x for percentage scale)
        "drawdown_norm": 10.0,      # Drawdown normalization (10x for percentage scale)
        "volatility_norm": 5.0,     # Volatility excess normalization

        # Penalty coefficients
        "turnover_penalty_coef": 0.1,  # Turnover penalty coefficient

        # Clipping bounds
        "clip_min": -2.0,           # Minimum reward value
        "clip_max": 2.0,            # Maximum reward value
    },

    "risk_aware": {
        # Component weights (must sum to 1.0 for main components)
        "mean_return": 0.30,         # Weight for mean return component
        "volatility_penalty": 0.25,  # Weight for volatility penalty
        "drawdown_penalty": 0.25,    # Weight for drawdown penalty
        "diversification_bonus": 0.10,  # Weight for diversification bonus
        "turnover_penalty": 0.10,    # Weight for turnover penalty

        # Normalization factors
        "mean_return_norm": 100.0,   # Mean return normalization (100x for small values)
        "volatility_norm": 0.15,     # Volatility normalization divisor
        "drawdown_exp": 5.0,         # Exponent for drawdown exponential penalty
        "entropy_factor": 2.0,       # Entropy bonus scaling factor
        "turnover_norm": 5.0,        # Turnover penalty normalization

        # Clipping bounds
        "clip_min": -2.0,            # Minimum reward value
        "clip_max": 2.0,             # Maximum reward value
    }
}

# Hyperparameters
LEARNING_RATE = 0.0003           # 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Network
POLICY_KWARGS = {
    'net_arch': [256, 256],
    'activation_fn': nn.Tanh  # Use callable, not string
}

# ============================================================================
# OUTPUT
# ============================================================================

VERBOSE = 1
SAVE_BEST_MODEL = True
SAVE_ALL_MODELS = False
USE_TENSORBOARD = True
TENSORBOARD_LOG = './logs'

# ============================================================================
# HELPERS
# ============================================================================

def load_data():
    """Load training data"""
    import pandas as pd
    data = pd.read_parquet(str(DATA_PATH))
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])
    return data


def print_config():
    """Print configuration"""
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"\n📊 DATA")
    print(f"   Data: {DATA_PATH}")
    print(f"   Tickers: {len(TICKERS)} - {', '.join(TICKERS)}")
    print(f"   Features: {len(FEATURE_COLS)}")
    print(f"   Metadata: {METADATA_PATH}")
    print(f"\n🎯 ENVIRONMENT")
    print(f"   Reward: {REWARD_TYPE} (lookback={REWARD_LOOKBACK})")
    print(f"   Rebalance: {REBALANCE_FREQUENCY} weeks")
    print(f"   Transaction cost: {TRANSACTION_COST:.4f} ({TRANSACTION_COST*10000:.0f} bps)")
    print(f"   Max weight: {MAX_WEIGHT_PER_ASSET:.0%}")
    print(f"\n🔄 VALIDATION")
    print(f"   Folds: {N_FOLDS}")
    print(f"\n🤖 TRAINING")
    print(f"   Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Network: {POLICY_KWARGS['net_arch']}")
    print("=" * 80)


def print_training_params():
    """Print detailed training parameters"""
    print("=" * 80)
    print("TRAINING PARAMETERS")
    print("=" * 80)

    print(f"\n🎯 REWARD FUNCTION:")
    print(f"   Type: {REWARD_TYPE}")
    print(f"   Lookback: {REWARD_LOOKBACK} weeks ({REWARD_LOOKBACK/4:.1f} months)")

    print(f"\n📊 ENVIRONMENT:")
    print(f"   Rebalance frequency: {REBALANCE_FREQUENCY} weeks (monthly)")
    print(f"   Transaction cost: {TRANSACTION_COST:.4f} ({TRANSACTION_COST*10000:.0f} bps)")
    print(f"   Max weight per asset: {MAX_WEIGHT_PER_ASSET:.0%}")
    print(f"   Initial capital: ${INITIAL_CAPITAL:,.0f}")

    print(f"\n🤖 PPO HYPERPARAMETERS:")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   N steps: {N_STEPS}")
    print(f"   N epochs: {N_EPOCHS}")
    print(f"   Gamma (discount): {GAMMA}")
    print(f"   GAE lambda: {GAE_LAMBDA}")
    print(f"   Clip range: {CLIP_RANGE}")
    print(f"   Entropy coefficient: {ENT_COEF}")
    print(f"   Value function coefficient: {VF_COEF}")
    print(f"   Max gradient norm: {MAX_GRAD_NORM}")

    print(f"\n🧠 NETWORK ARCHITECTURE:")
    print(f"   Layers: {POLICY_KWARGS['net_arch']}")
    print(f"   Activation: {POLICY_KWARGS['activation_fn'].__name__}")

    print(f"\n⏱️  TRAINING SCHEDULE:")
    print(f"   Total timesteps per fold: {TOTAL_TIMESTEPS:,}")
    print(f"   Evaluation frequency: {EVAL_FREQ:,} steps")
    print(f"   Evaluation episodes: {N_EVAL_EPISODES}")
    print(f"   Early stopping patience: 20 evaluations")

    print(f"\n🔄 VALIDATION:")
    print(f"   Number of folds: {N_FOLDS}")
    print(f"   Min train size: {MIN_TRAIN_SIZE if MIN_TRAIN_SIZE else 'Auto (20% of data)'}")

    print("=" * 80)


if __name__ == '__main__':
    print_config()
