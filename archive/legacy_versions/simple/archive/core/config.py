import os

# --- Paths ---
# Directory definitions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Data files (legacy - for backward compatibility)
RAW_DATA_TRAIN_FILE = os.path.join(DATA_DIR, 'raw_data_train.csv')
BENCHMARK_FILE = os.path.join(DATA_DIR, 'qqq_benchmark.csv')
TEST_SET_FILE = os.path.join(DATA_DIR, 'test_set_NEVER_TOUCH.csv')

# Helper function to get mode-specific data paths
def get_data_paths(expanded_mode=False):
    """
    Get data file paths.
    
    Args:
        expanded_mode: Ignored (kept for compatibility)
        
    Returns:
        dict with keys: 'train', 'test', 'benchmark', 'data_dir'
    """
    # Flattened structure: Save directly to DATA_DIR
    return {
        'train': os.path.join(DATA_DIR, 'raw_data_train.csv'),
        'test': os.path.join(DATA_DIR, 'test_set_NEVER_TOUCH.csv'),
        'benchmark': os.path.join(DATA_DIR, 'qqq_benchmark.csv'),
        'data_dir': DATA_DIR
    }


# --- Assets ---
TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']
MACRO_SYMBOLS = {
    'vix': '^VIX', 
    'dxy': 'DX-Y.NYB', 
    'oil': 'CL=F', 
    'tnx': '^TNX', # 10 Year Treasury Yield
    'irx': '^IRX'  # 13 Week Treasury Bill (Risk Free Rate Proxy)
}

# --- Dates ---
START_DATE = '2014-01-01' 
END_DATE = '2025-11-01'

# --- Environment Settings ---
INITIAL_BALANCE = 100000
REBALANCE_PERIOD = 4  # Weeks

# UPDATED: Reduced transaction costs to encourage active trading
TRANSACTION_COST_BPS = 0.0005  # 5 basis points (was 30)
SLIPPAGE_BPS = 0.0005  # 5 basis points (was 20)
TOTAL_COST_BPS = TRANSACTION_COST_BPS + SLIPPAGE_BPS  # 10 bps total (was 50)

# --- Training Settings ---
N_FOLDS = 3  # Expanding window: each fold gets progressively more training data (1x, 2x, 3x)
# CHANGED: Only purge the label overlap (rebalance period)
PURGE_WINDOW = 4
# CHANGED: Add small buffer. Total gap = 8 weeks (Safe & Efficient)
GAP = PURGE_WINDOW + 4
TEST_RATIO = 0.2  # NEW: Reserve 20% for true test set

# Epoch-based training (standard in financial RL papers)
EPOCHS_PER_FOLD = 100  # Balanced: allows learning without excessive overfitting
EVAL_FREQ_EPOCHS = 5   # Evaluate every 5 epochs through the data

# --- PPO Hyperparameters (Updated for limited training data + anti-overfitting) ---
PPO_PARAMS = {
    "learning_rate": 0.0003,
    "n_steps": 192,  # Reduced to ~15 episodes (was 738 = 57 episodes, too many for fold 1)
    "batch_size": 64,  # 3 mini-batches (192 / 64)
    "ent_coef": 0.04,  # Increased from 0.02 to 0.05 for stronger exploration/regularization
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
}

# --- Feature Engineering ---
MIN_HISTORY = 26  # Minimum history needed for stable features
FEATURE_WINDOWS = [4, 13, 26]  # Various lookback windows