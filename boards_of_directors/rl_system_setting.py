import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import gymnasium as gym
import config
from gymnasium import spaces

# --- CONFIGURATION ---
# This list combines High-Beta Tech (for the Bull) and Uncorrelated Hedges (for the Sniper/Bear)
V3_TICKERS = [
    'NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG',  # Growth / Tech
    'GLD',  # Gold (Inflation Hedge)
    'TLT',  # Long Term Treasuries (Deflation Hedge)
    'XLE',  # Energy (Commodity Cycle Hedge)
    'XLP'   # Consumer Staples (Low Volatility Hedge)
]

def load_real_data(tickers=None, benchmark='QQQ', start='2015-01-01', end=None):
    """
    Downloads and cleans financial data for the V3.1 Alpha Hunter system.
    """
    # 1. dynamic End Date
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
        
    if tickers is None:
        tickers = V3_TICKERS

    # 2. Download Data
    # We add the benchmark to the list to download everything in one shot
    all_tickers = tickers + [benchmark]
    print(f"📥 Downloading {len(all_tickers)} assets from {start} to {end}...")
    
    # auto_adjust=True handles stock splits and dividends automatically
    raw_data = yf.download(all_tickers, start=start, end=end, progress=False, auto_adjust=True)
    
    # 3. Formatting Fixes
    # YFinance returns a MultiIndex ('Close', 'AAPL'). We just want the Close prices.
    if isinstance(raw_data.columns, pd.MultiIndex):
        try:
            df = raw_data['Close'].copy()
        except KeyError:
            # Fallback for older yfinance versions
            df = raw_data['close'].copy()
    else:
        df = raw_data.copy()

    # 4. Benchmark Renaming
    # The Feature Engineer expects a column strictly named 'SPY' to calculate correlations.
    # We rename whatever benchmark you chose (e.g., QQQ) to 'SPY' internally.
    if benchmark in df.columns:
        df = df.rename(columns={benchmark: 'SPY'})
    else:
        raise ValueError(f"Benchmark '{benchmark}' failed to download. Check ticker symbol.")

    # 5. Data Safety & Cleaning (The "No-Delete" Fix)
    # First, forward fill to bridge small gaps (holidays/weekends)
    df = df.ffill()
    
    # Critical Step: Check for "Broken" Assets
    # If an asset (like a new ETF) has >5% missing data, drop the ASSET, not the ROWS.
    # This prevents 1 bad stock from deleting 7 years of history for NVDA.
    missing_fractions = df.isnull().mean()
    valid_cols = missing_fractions[missing_fractions < 0.05].index
    
    dropped_assets = [c for c in df.columns if c not in valid_cols]
    if dropped_assets:
        print(f"⚠️ Dropping assets with insufficient history: {dropped_assets}")
        
    df = df[valid_cols]
    
    # Now it is safe to drop any remaining rows (usually just the first few days)
    df = df.dropna().round(2)
    
    # 6. Final Formatting
    # Remove timezone info to prevent errors in backtesting loop
    df.index = df.index.tz_localize(None)
    
    # Ensure column order: Assets first, Benchmark last
    # (This is required for the environment's matrix slicing)
    final_asset_cols = [t for t in tickers if t in df.columns and t != 'SPY']
    final_cols = final_asset_cols + ['SPY']
    
    print(f"✅ Data Ready: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Date Range: {df.index.min().date()} -> {df.index.max().date()}")
    
    return df[final_cols]

class PortfolioRebalanceEnv(gym.Env):
    def __init__(self, price_data, features_data, benchmark_data, 
                 initial_balance=100_000, lookback_window=60):
        super().__init__()
        self.prices = price_data
        self.features = features_data
        self.benchmark = benchmark_data
        self.n_assets = self.prices.shape[1]
        self.n_features = self.features.shape[2]
        self.lookback_window = lookback_window
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(lookback_window, self.n_assets, self.n_features), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.cash = 100_000.0
        self.shares = np.zeros(self.n_assets)
        self.portfolio_value = 100_000.0
        self.history_nav = [100_000.0]
        return self._get_obs(), {}

    def step(self, action):
        exp_action = np.exp(action)
        weights = exp_action / np.sum(exp_action)
        
        current_prices = self.prices[self.current_step]
        target_val = self.portfolio_value * weights[:-1]
        current_holdings = self.shares * current_prices
        cost = np.sum(np.abs(target_val - current_holdings)) * config.TX_COST_BPS
        
        self.cash = (self.portfolio_value * weights[-1]) - cost
        self.shares = target_val / current_prices
        
        daily_ret = []
        steps = min(20, len(self.prices) - self.current_step - 1)
        for _ in range(steps):
            self.current_step += 1
            val = self.cash + np.sum(self.shares * self.prices[self.current_step])
            daily_ret.append(np.log(val / self.history_nav[-1]))
            self.history_nav.append(val)
            
        self.portfolio_value = self.history_nav[-1]
        
        # Dynamic Reward
        bench_slice = self.benchmark[self.current_step-20:self.current_step]
        mkt_vol = np.std(bench_slice) if len(bench_slice) > 1 else 0.01
        lam = config.RISK_AVERSION + ((mkt_vol - 0.015)*100 if mkt_vol > 0.015 else 0)
        
        r = np.array(daily_ret)
        downside = np.std(r[r<0]) if len(r[r<0]) > 0 else 0.0
        reward = np.sum(r) - (lam * downside)
        
        return self._get_obs(), reward, self.current_step >= len(self.prices) - 2, False, {}

    def _get_obs(self):
        return self.features[self.current_step-self.lookback_window : self.current_step].astype(np.float32)
