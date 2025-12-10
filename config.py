
# --- Strategy Hyperparameters ---
WINDOW_SIZE = 60                # Observation window for the AI
RISK_AVERSION = 4.0             # Reward penalty for volatility
TX_COST_BPS = 0.0005            # 5 bps transaction cost
REBALANCE_DAYS = 5              # Weekly rebalancing
REBALANCE_THRESHOLD = 0.05      # Only trade if allocation drifts > 5%
RISK_FREE_RATE = 0.04           # 4% annual risk-free rate

# --- Feature Engineering ---
VOLATILITY_WINDOW = 20
VOLATILITY_NORMALIZATION_WINDOW = 252
RSI_PERIOD = 14
CORRELATION_WINDOW = 60

# --- Board of Directors (Ensemble) Thresholds ---
# These are FLOORS. We will optimize them in the Validation step.
PANIC_VOL_THRESHOLD = 0.015   
CHOPPY_VOL_THRESHOLD = 0.008  
RSI_OVERSOLD = 30              
RSI_OVERBOUGHT = 70            

# --- Assets ---
# Added 'GLD' for crisis management. 
# CRITICAL: Tickers MUST be in alphabetical order.
TICKERS = sorted(['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG', 'GLD'])
BENCHMARK = 'QQQ'
