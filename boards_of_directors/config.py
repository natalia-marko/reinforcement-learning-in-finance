# config.py

# Assets to Trade
<<<<<<< HEAD
# Tech + Defensive Mix from notebook
TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG', 'GLD', 'TLT', 'XLE', 'XLP']
=======
TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']
>>>>>>> 612598b
BENCHMARK = 'QQQ'

# Hyperparameters
WINDOW_SIZE = 60
<<<<<<< HEAD
SMA_WINDOW = 50
RSI_WINDOW = 14
RISK_AVERSION = 4.0
TX_COST_BPS = 0.0010   # 10 bps
REBALANCE_THRESHOLD = 0.05 # Default, optimized later

# Training
TRAIN_TEST_SPLIT = 0.8
ENTROPY_COEF = 0.05
TOTAL_TIMESTEPS_BULL = 30_000
TOTAL_TIMESTEPS_SNIPER = 50_000

# Optuna Optimization
OPTUNA_N_TRIALS = 100
PANIC_VOL_RANGE = (0.015, 0.035)
THRESH_RANGE = (0.02, 0.10)
REENTRY_ALLOC_RANGE = (0.3, 0.9)

# Date Range
START_DATE = '2015-01-01'
END_DATE = '2025-10-31'
=======
RISK_AVERSION = 4.0
TX_COST_BPS = 0.0010 # 0.10%
REBALANCE_THRESHOLD = 0.05

# Date Range
START_DATE = '2015-01-01'
END_DATE = '2025-12-31'
>>>>>>> 612598b

# Saved Model Names
MODEL_BULL = "agent_bull"
MODEL_BEAR = "agent_bear"
MODEL_SNIPER = "agent_sniper"

<<<<<<< HEAD
# RESTORING THE "MASTERPIECE" SETTINGS (Reference)
=======

# RESTORING THE "MASTERPIECE" SETTINGS
>>>>>>> 612598b
BEST_FAST_VOL = 0.0172  # 1.72% (Tighter leash)
BEST_SLOW_VOL = 0.0255  # 2.55%
BEST_THRESH   = 0.0976  # 9.76% (Lazy trading)