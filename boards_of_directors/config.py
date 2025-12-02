# config.py

# Assets to Trade
TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']
BENCHMARK = 'QQQ'

# Hyperparameters
WINDOW_SIZE = 60
RISK_AVERSION = 4.0
TX_COST_BPS = 0.0010 # 0.10%
REBALANCE_THRESHOLD = 0.05

# Date Range
START_DATE = '2015-01-01'
END_DATE = '2025-12-31'

# Saved Model Names
MODEL_BULL = "agent_bull"
MODEL_BEAR = "agent_bear"
MODEL_SNIPER = "agent_sniper"


# RESTORING THE "MASTERPIECE" SETTINGS
BEST_FAST_VOL = 0.0172  # 1.72% (Tighter leash)
BEST_SLOW_VOL = 0.0255  # 2.55%
BEST_THRESH   = 0.0976  # 9.76% (Lazy trading)