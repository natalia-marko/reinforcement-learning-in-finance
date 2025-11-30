import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import gymnasium as gym
import config
from gymnasium import spaces


def load_real_data(tickers=None, benchmark='QQQ', start='2015-01-01', end=None):
    """
    Downloads and cleans financial data for the V3.1 Alpha Hunter system.
    """
    # 1. dynamic End Date
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
        
    if tickers is None:
        tickers = config.TICKERS

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
    
    print(f"   Date Range: {df.index.min().date()} -> {df.index.max().date()}")
    
    return df[final_cols]

class FinancialFeatureEngineer:
    def preprocess_data(self, df):
        data = df.copy()
        asset_cols = [c for c in data.columns if c != 'SPY']
        
        # Log Returns
        log_ret = np.log(data[asset_cols] / data[asset_cols].shift(1)).fillna(0)
        
        # Features
        # 1. Volatility (Normalized)
        roll_std = log_ret.rolling(20).std()
        norm_vol = (roll_std - roll_std.rolling(252).mean()) / (roll_std.rolling(252).std() + 1e-8)
        
        # 2. RSI (Normalized)
        delta = data[asset_cols].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        norm_rsi = (100 - (100 / (1 + rs)) - 50) / 50 
        
        # 3. Distance from SMA
        sma_50 = data[asset_cols].rolling(50).mean()
        dist_sma = (data[asset_cols] / sma_50) - 1.0

        # 4. Relative Strength vs Benchmark
        asset_cum = data[asset_cols].pct_change(60)
        bench_cum = data['SPY'].pct_change(60)
        rel_str = asset_cum.sub(bench_cum, axis=0)
        
        # 5. Beta
        market_ret = log_ret.mean(axis=1) 
        rolling_cov = log_ret.rolling(60).cov(market_ret)
        rolling_var = market_ret.rolling(60).var()
        beta = (rolling_cov.div(rolling_var, axis=0)).fillna(1.0)

        # Stack Features
        features = np.stack([
            norm_vol.fillna(0).clip(-3, 3).values, 
            norm_rsi.fillna(0).clip(-1, 1).values, 
            dist_sma.fillna(0).clip(-0.5, 0.5).values, 
            rel_str.fillna(0).clip(-0.5, 0.5).values,
            beta.fillna(1.0).clip(-2, 2).values, 
            log_ret.values
        ], axis=-1)
        
        return features.astype(np.float32), data[asset_cols].values.astype(np.float32), data['SPY'].values.astype(np.float32), asset_cols

class PortfolioRebalanceEnv(gym.Env):
    def __init__(self, price_data, features_data, benchmark_data, 
                 initial_balance=100_000, lookback_window=60,
                 panic_vol=0.02, strong_trend=0.005):
        super().__init__()
        self.prices = price_data
        self.features = features_data
        self.benchmark = benchmark_data
        self.initial_balance = initial_balance
        self.n_assets = self.prices.shape[1]
        # +1 for Regime Signal (0=Normal, 1=Crash, 2=Rally)
        self.n_features = self.features.shape[2] + 1 
        self.lookback_window = lookback_window
        
        # Strategy Params for Regime Detection
        self.panic_vol = panic_vol
        self.strong_trend = strong_trend
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        # Observation space now includes Regime
        self.observation_space = spaces.Box(low=-5, high=5, shape=(lookback_window, self.n_assets, self.n_features), dtype=np.float32)
        
        # DSR Variables
        self.ret_avg = 0.0
        self.ret_std = 0.0
        self.decay = 0.99  # For running stats
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.cash = float(self.initial_balance)
        self.shares = np.zeros(self.n_assets)
        self.portfolio_value = float(self.initial_balance)
        self.history_nav = [float(self.initial_balance)]
        
        # Reset DSR stats
        self.ret_avg = 0.0
        self.ret_std = 0.0
        
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
            step_ret = np.log(val / self.history_nav[-1])
            daily_ret.append(step_ret)
            self.history_nav.append(val)
            
            # Update DSR Running Stats incrementally
            self.ret_avg = self.decay * self.ret_avg + (1 - self.decay) * step_ret
            self.ret_std = self.decay * self.ret_std + (1 - self.decay) * (step_ret ** 2)
            
        self.portfolio_value = self.history_nav[-1]
        
        # === IMPROVEMENT #2: DIFFERENTIAL SHARPE RATIO (DSR) ===
        # DSR = (Return_t - Return_t-1) / Std_t
        # We use the running average of returns as a proxy for the gradient
        sigma = np.sqrt(self.ret_std - self.ret_avg**2) + 1e-8
        dsr = self.ret_avg / sigma
        
        # We can also add a penalty for extreme volatility (Panic)
        # But DSR naturally handles risk. We just scale it up.
        reward = dsr * 10.0 
        
        return self._get_obs(), reward, self.current_step >= len(self.prices) - 2, False, {}

    def _get_obs(self):
        # Base Features
        base_feats = self.features[self.current_step-self.lookback_window : self.current_step]
        
        # === IMPROVEMENT #1: REGIME LOGIC ===
        # Calculate Regime for the current step
        bench_slice = self.benchmark[self.current_step-20:self.current_step]
        mkt_vol = np.std(bench_slice) if len(bench_slice) > 1 else 0.01
        
        # Trend Strength (using feature index 3 which is Relative Strength or similar trend feature)
        # Note: Ensure feature index 3 corresponds to a trend metric in your Feature Engineer
        avg_trend = np.mean(base_feats[:, :, 3]) 
        
        regime = 0.0 # Normal
        if mkt_vol > self.panic_vol:
            regime = 1.0 # Crash
        elif avg_trend > self.strong_trend:
            regime = 2.0 # Rally
            
        # Broadcast Regime to match shape (Lookback, Assets, 1)
        regime_layer = np.full((self.lookback_window, self.n_assets, 1), regime, dtype=np.float32)
        
        # Concatenate: (Lookback, Assets, Features) + (Lookback, Assets, 1) -> (Lookback, Assets, Features+1)
        return np.concatenate([base_feats, regime_layer], axis=2).astype(np.float32)

class AlphaHunterStrategy:
    def __init__(self, model, tickers, panic_vol, strong_trend, rebalance_threshold, cooldown=3):
        self.model = model
        self.tickers = tickers
        self.panic_vol = panic_vol
        self.strong_trend = strong_trend
        self.rebalance_threshold = rebalance_threshold
        self.cooldown = cooldown
        self.days_since_crash = 999 
        
    def _softmax(self, x):
        e = np.exp(x - np.max(x)); return e / e.sum()

    def predict(self, obs, market_vol, trend_strength):
        action, _ = self.model.predict(obs, deterministic=True)
        if action.ndim > 1: action = action[0]
        weights = self._softmax(action) 
        regime = "NORMAL"
        
        self.days_since_crash += 1
        
        # 1. CRASH LOGIC
        if market_vol > self.panic_vol:
            regime = "CRASH"
            self.days_since_crash = 0
        elif self.days_since_crash < self.cooldown:
            regime = "CRASH" # Stay in safety
            
        # 2. RALLY LOGIC
        elif trend_strength > self.strong_trend:
            regime = "RALLY"

        # 3. APPLY REGIME
        if regime == "CRASH":
            weights = np.zeros_like(weights); weights[-1] = 1.0
        elif regime == "RALLY":
            weights[-1] = 0.0 # No Cash
            if np.sum(weights) > 0: weights /= np.sum(weights)
                
        return weights, regime

