import os

# --- Paths ---
# Directory definitions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Data files (legacy - for backward compatibility)
RAW_DATA_TRAIN_FILE = os.path.join(DATA_DIR, 'raw_data_train.csv')
BENCHMARK_FILE = os.path.join(DATA_DIR, 'qqq_benchmark.csv')
TEST_SET_FILE = os.path.join(DATA_DIR, 'test_set_NEVER_TOUCH.csv')

# Helper function to get mode-specific data paths
def get_data_paths(expanded_mode=False):
    """
    Get data file paths based on feature mode.
    
    Args:
        expanded_mode: If True, use expanded feature set directory
        
    Returns:
        dict with keys: 'train', 'test', 'benchmark', 'data_dir'
    """
    mode_dir = 'expanded' if expanded_mode else 'simple'
    data_subdir = os.path.join(DATA_DIR, mode_dir)
    os.makedirs(data_subdir, exist_ok=True)
    
    return {
        'train': os.path.join(data_subdir, 'raw_data_train.csv'),
        'test': os.path.join(data_subdir, 'test_set_NEVER_TOUCH.csv'),
        'benchmark': os.path.join(data_subdir, 'qqq_benchmark.csv'),
        'data_dir': data_subdir
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
INITIAL_BALANCE = 10000
REBALANCE_PERIOD = 4  # Weeks

# CRITICAL FIX: Realistic transaction costs
TRANSACTION_COST_BPS = 0.0030  # 30 basis points (was 10)
SLIPPAGE_BPS = 0.0020  # 20 basis points (NEW)
TOTAL_COST_BPS = TRANSACTION_COST_BPS + SLIPPAGE_BPS  # 50 bps total

# --- Training Settings ---
N_FOLDS = 3
PURGE_WINDOW = 52  # NEW: Maximum lookback period in features
GAP = PURGE_WINDOW + REBALANCE_PERIOD  # CRITICAL FIX: Proper embargo (was 8)
TEST_RATIO = 0.2  # NEW: Reserve 20% for true test set
TRAIN_TIMESTEPS = 100000

# --- PPO Hyperparameters (LSTM Optimized) ---
PPO_PARAMS = {
    "learning_rate": 0.0003,
    "n_steps": 256,  # Reduced from 2048 to match dataset size (~600 samples)
    "batch_size": 64,  # Reduced from 128 for better gradient updates
    "ent_coef": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
}

# --- Feature Engineering ---
MIN_HISTORY = 52  # Minimum history needed for stable features
FEATURE_WINDOWS = [4, 13, 26, 52]  # Various lookback windows