import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
from config import TX_COST_BPS, RISK_AVERSION
from stable_baselines3 import PPO
import yfinance as yf

def load_real_data(tickers=None, benchmark='QQQ', start='2015-01-01', end='2025-10-31'):
    """
    Robust loader for Yahoo Finance data.
    Fixes: Timezones, Missing Values (FFILL), and MultiIndex formatting.
    """
    if tickers is None:
        # Tech heavy portfolio
        tickers = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']
    
    # Download everything at once
    all_tickers = tickers + [benchmark]
    print(f"Downloading {len(all_tickers)} assets...")
    
    # auto_adjust=True gives us 'Close' adjusted for splits/dividends
    raw_data = yf.download(all_tickers, start=start, end=end, progress=False, auto_adjust=True)
    
    # 1. Handle YFinance MultiIndex (Price Type, Ticker)
    # The columns usually look like ('Close', 'AAPL'). We just want the 'Close' slice.
    if isinstance(raw_data.columns, pd.MultiIndex):
        try:
            # Try grabbing the 'Close' cross-section directly
            df = raw_data['Close'].copy()
        except KeyError:
            # Fallback if yfinance changes format or case
            df = raw_data['close'].copy()
    else:
        df = raw_data.copy()

    # 2. Rename Benchmark to 'QQQ' 
    # (The Environment expects a column named 'QQQ' for the market regime logic)
    if benchmark in df.columns:
        df = df.rename(columns={benchmark: 'QQQ'})
    
    # 3. Data Cleaning (Crucial for RL)
    # Forward Fill first (carry forward yesterday's price to fill holidays)
    df = df.ffill()
    # Then drop any remaining NaNs (usually the first few days if data starts at different times)
    df = df.dropna().round(2)
    
    # 4. Strip Timezone (Make it naive)
    df.index = df.index.tz_localize(None)
    
    # 5. Reorder: Assets first, Benchmark last
    # This ensures the matrix slicing in the Env works as expected
    asset_cols = [t for t in tickers if t in df.columns]
    final_cols = asset_cols + ['QQQ']
    
    # Verify we didn't lose the benchmark
    if 'QQQ' not in df.columns:
        raise ValueError("Benchmark failed to download or rename!")
        
    return df[final_cols]


class FinancialFeatureEngineer:
    def preprocess_data(self, df):
        """
        Input: DataFrame (Time, Assets + QQQ)
        Output: 
            - features: 3D Array (Time, Assets, Num_Features)
            - prices: 2D Array (Time, Assets)
            - benchmark: 1D Array (Time)
        """
        data = df.copy()
        asset_cols = [c for c in data.columns if c != 'QQQ']
        
        # We will build a list of feature matrices (one per feature type)
        # 1. Log Returns (Momentum/Volatility proxy)
        log_ret = np.log(data[asset_cols] / data[asset_cols].shift(1)).fillna(0)
        
        # 2. Rolling Volatility (Risk) - Normalized
        # Z-Score relative to last 252 days
        roll_std = log_ret.rolling(20).std()
        roll_mean_std = roll_std.rolling(252).mean()
        roll_std_std = roll_std.rolling(252).std()
        norm_vol = (roll_std - roll_mean_std) / (roll_std_std + 1e-8)
        
        # 3. RSI (Simplified Vectorized)
        # Note: In production, use pandas_ta.rsi(data[col])
        delta = data[asset_cols].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        norm_rsi = (rsi - 50) / 100 # Scale to -0.5 to 0.5
        
        # 4. Correlation to QQQ (Systemic Risk)
        # How much does this asset move with the market?
        market_ret = data['QQQ'].pct_change()
        correlations = log_ret.rolling(60).corr(market_ret)
        
        # Clean NaNs
        norm_vol = norm_vol.fillna(0).clip(-3, 3)
        norm_rsi = norm_rsi.fillna(0)
        correlations = correlations.fillna(0)
        
        # Stack Features: Shape (Time, Assets, 3)
        # We stack them so features[t, asset_i] = [Vol, RSI, Corr]
        feature_stack = np.stack([norm_vol.values, norm_rsi.values, correlations.values], axis=-1)
        
        return feature_stack, data[asset_cols].values, data['QQQ'].values

class PortfolioRebalanceEnv(gym.Env):
    def __init__(self, price_data, features_data, benchmark_data, 
                 initial_balance=100_000, lookback_window=60):
        super(PortfolioRebalanceEnv, self).__init__()
        
        self.prices = price_data
        self.features = features_data
        self.benchmark = benchmark_data
        
        self.n_assets = self.prices.shape[1]
        self.n_features = self.features.shape[2]
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        
        # Action: Weight for each asset + Cash (Softmaxed later)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        
        # Observation: (Window, Assets, Features)
        self.observation_space = spaces.Box(
            low=-5, high=5, 
            shape=(lookback_window, self.n_assets, self.n_features), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start after window, end before last month
        self.current_step = self.lookback_window
        self.cash = self.initial_balance
        self.shares = np.zeros(self.n_assets)
        self.portfolio_value = self.initial_balance
        self.history_nav = [self.initial_balance]
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Action -> Weights (Softmax)
        exp_action = np.exp(action)
        weights = exp_action / np.sum(exp_action)
        asset_weights = weights[:-1]
        cash_weight = weights[-1]
        
        # 2. Execute Rebalance at T
        current_prices = self.prices[self.current_step]
        target_val = self.portfolio_value * asset_weights
        
        # Cost Logic
        current_holdings = self.shares * current_prices
        turnover = np.sum(np.abs(target_val - current_holdings))
        cost = turnover * TX_COST_BPS
        
        self.cash = (self.portfolio_value * cash_weight) - cost
        self.shares = target_val / current_prices
        
        # 3. Simulate Month (20 Days)
        daily_log_returns = []
        steps_to_sim = min(20, len(self.prices) - self.current_step - 1)
        
        for i in range(steps_to_sim):
            self.current_step += 1
            daily_price = self.prices[self.current_step]
            
            # NAV update
            new_nav = self.cash + np.sum(self.shares * daily_price)
            daily_ret = np.log(new_nav / self.history_nav[-1])
            
            self.history_nav.append(new_nav)
            daily_log_returns.append(daily_ret)
            
        self.portfolio_value = self.history_nav[-1]
        
        # 4. Intelligent Reward (Dynamic Sortino)
        reward = self._calculate_reward(daily_log_returns)
        
        # 5. Check Done
        terminated = self.current_step >= len(self.prices) - 2
        
        return self._get_obs(), reward, terminated, False, {'nav': self.portfolio_value}

    def _get_obs(self):
        obs = self.features[self.current_step - self.lookback_window : self.current_step]
        return obs.astype(np.float32)

    def _calculate_reward(self, returns):
        # Dynamic Risk Aversion
        # Check Benchmark Volatility over last month
        bench_slice = self.benchmark[self.current_step-20 : self.current_step]
        mkt_vol = np.std(bench_slice) if len(bench_slice) > 1 else 0.01
        
        # If Market Vol > 1.5%, lambda goes from 2.0 to 5.0+
        local_lambda = RISK_AVERSION
        if mkt_vol > 0.015:
            local_lambda += (mkt_vol - 0.015) * 100
            
        # Sortino
        r = np.array(returns)
        downside = r[r < 0]
        downside_dev = np.std(downside) if len(downside) > 0 else 0.001
        
        return np.sum(r) - (local_lambda * downside_dev)


def create_regime_datasets(prices, features, benchmark, vol_window=20):
    """
    Splits the dataset into 'Bull' and 'Bear' subsets based on realized volatility.
    """
    # 1. Calculate Rolling Volatility of the Benchmark (Market Regime)
    # Ensure benchmark is a Series
    if isinstance(benchmark, np.ndarray):
        bench_series = pd.Series(benchmark)
    else:
        bench_series = benchmark
        
    rolling_vol = bench_series.pct_change().rolling(vol_window).std()
    
    # 2. Define Thresholds (Data Science approach: Percentiles)
    # Bear Regime: Top 25% most volatile days
    # Bull Regime: Bottom 50% volatility (Calm days)
    threshold_bear = rolling_vol.quantile(0.75)
    threshold_bull = rolling_vol.quantile(0.50)
    
    # 3. Create Masks
    # We need contiguous chunks for LSTM/Windowed setups, but for PPO with 
    # random sampling, masking frames is okay-ish. 
    # BETTER: We just train on the whole timeline but use a custom 
    # 'RegimeWeighted' sampling, OR simply slice the DataFrame if rows are independent.
    
    # Since our Env needs contiguous windows (Lookback=60), 
    # we cannot just drop random rows.
    # SOLUTION: We identify 'Bear Years' and 'Bull Years' to keep contiguous blocks.
    
    # Identify Indices
    is_bear = rolling_vol > threshold_bear
    is_bull = rolling_vol < threshold_bull
    
    # We simply return the MASKS. The training loop will use these to 
    # "Reset" the environment to valid start points or we filter data before env creation.
    
    # For simplicity/robustness in Gym:
    # We will create copies of data where the 'off-regime' periods are zeroed out 
    # or we simply pass the indices to the Env to restrict reset().
    
    # KISS Approach: Just return the thresholds so we can log them.
    # We will slice the data MANUALLY based on index chunks if using real dates.
    
    return threshold_bull, threshold_bear

class SmartBacktesterWithWeights:
    def __init__(self, model, env, threshold=0.05):
        self.model = model
        self.env = env
        self.threshold = threshold
        
    def run(self):
        cash = 100_000
        shares = np.zeros(self.env.n_assets)
        history = []
        
        curr_step = self.env.lookback_window
        max_step = len(self.env.prices) - 22
        
        while curr_step < max_step:
            # ... [Same Prediction Logic as before] ...
            obs = self.env.features[curr_step - self.env.lookback_window : curr_step]
            action, _ = self.model.predict(obs, deterministic=True)
            exp_act = np.exp(action)
            target_weights = exp_act / np.sum(exp_act)
            
            # ... [Same Wrapper Logic as before] ...
            current_prices = self.env.prices[curr_step]
            current_val = cash + np.sum(shares * current_prices)
            
            if current_val > 0:
                # Calculate percentages for logging
                asset_vals = shares * current_prices
                curr_w = np.append(asset_vals / current_val, cash / current_val)
            else:
                curr_w = np.zeros(self.env.n_assets + 1)
                curr_w[-1] = 1.0 # 100% Cash

            turnover = np.sum(np.abs(target_weights - curr_w))
            
            action_type = "HOLD"
            if turnover > self.threshold:
                action_type = "TRADE"
                target_vals = current_val * target_weights
                cost = np.sum(np.abs(target_vals[:-1] - (shares*current_prices))) * TX_COST_BPS
                cash = target_vals[-1] - cost
                shares = target_vals[:-1] / current_prices
                # Update weights after trade for logging
                curr_w = target_weights 

            # Simulate Month
            for t in range(20):
                day_idx = curr_step + t + 1
                if day_idx >= len(self.env.prices): break
                
                day_price = self.env.prices[day_idx]
                nav = cash + np.sum(shares * day_price)
                
                # --- NEW: Save Weights ---
                history.append({
                    'step': day_idx, 
                    'nav': nav, 
                    'action': action_type if t==0 else None,
                    'weights': curr_w # <--- Storing the vector
                })
            
            curr_step += 20
            
        return pd.DataFrame(history)

        
class BoardOfDirectors:
    def __init__(self, bull_path, bear_path, sniper_path, threshold=0.05):
        self.bull = PPO.load(bull_path)
        self.bear = PPO.load(bear_path)
        self.sniper = PPO.load(sniper_path)
        self.threshold = threshold
        
    def predict(self, obs, market_vol_percent):
        """
        Input: 
            obs: Observation for the agent
            market_vol_percent: StdDev of Benchmark (e.g., 0.015)
        Output:
            final_weights: The agreed portfolio allocation
            regime_name: String for logging
        """
        # 1. Get Votes (Logits)
        # We assume single observation (non-vectorized) for backtesting loop
        a_bull, _ = self.bull.predict(obs, deterministic=True)
        a_bear, _ = self.bear.predict(obs, deterministic=True)
        a_sniper, _ = self.sniper.predict(obs, deterministic=True)
        
        # 2. Convert to Weights (Softmax)
        w_bull = self._softmax(a_bull)
        w_bear = self._softmax(a_bear)
        w_sniper = self._softmax(a_sniper)
        
        # 3. The Gavel (Regime Decision)
        # Using the thresholds we calculated earlier or standard heuristics
        
        # Panic Threshold (e.g., Vol > 2%)
        PANIC_LEVEL = 0.020 
        # Caution Threshold (e.g., Vol > 1.2%)
        CAUTION_LEVEL = 0.012
        
        if market_vol_percent > PANIC_LEVEL:
            # CRASH REGIME: The Bear takes command
            # 80% Bear, 20% Sniper (to keep some liquidity), 0% Bull
            final_weights = (0.8 * w_bear) + (0.2 * w_sniper)
            regime = "CRASH (Bear Mode)"
            
        elif market_vol_percent > CAUTION_LEVEL:
            # CHOPPY REGIME: The Sniper leads, Bull/Bear advise
            # 50% Sniper, 25% Bear, 25% Bull
            final_weights = (0.5 * w_sniper) + (0.25 * w_bear) + (0.25 * w_bull)
            regime = "CHOPPY (Mixed Mode)"
            
        else:
            # GROWTH REGIME: The Bull runs free
            # 70% Bull, 30% Sniper
            final_weights = (0.7 * w_bull) + (0.3 * w_sniper)
            regime = "GROWTH (Bull Mode)"
            
        return final_weights, regime

    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

class EnsembleBacktester:
    def __init__(self, board, env, threshold=0.05):
        self.board = board
        self.env = env
        self.threshold = threshold
        
    def run(self):
        cash = 100_000
        shares = np.zeros(self.env.n_assets)
        history = []
        
        curr_step = self.env.lookback_window
        max_step = len(self.env.prices) - 22
        
        while curr_step < max_step:
            # 1. Get Data
            obs = self.env.features[curr_step - self.env.lookback_window : curr_step]
            
            # 2. Get Market Regime (Vol of last 20 days)
            # We use the 'benchmark' array from the env
            # --- NEW (CORRECTED) ---
            # 1. Get the price slice
            bench_slice = self.env.benchmark[curr_step-20 : curr_step]

            # 2. Convert to Pandas Series to use pct_change() easily
            if len(bench_slice) > 1:
                # Calculate Standard Deviation of Returns (not Prices!)
                mkt_vol = pd.Series(bench_slice).pct_change().std()
                # Fill NaN (first item) with 0 to be safe
                if np.isnan(mkt_vol): mkt_vol = 0.01
            else:
                mkt_vol = 0.01
            
            # 3. GET ENSEMBLE VOTE
            target_weights, regime = self.board.predict(obs, mkt_vol)
            
            # 4. Wrapper Logic
            current_prices = self.env.prices[curr_step]
            current_val = cash + np.sum(shares * current_prices)
            
            if current_val > 0:
                curr_w = np.append((shares * current_prices) / current_val, cash / current_val)
            else:
                curr_w = np.zeros_like(target_weights)
            
            turnover = np.sum(np.abs(target_weights - curr_w))
            
            # Dynamic Threshold (Ultrathinking: Be faster to sell in crashes)
            active_threshold = self.threshold
            if "CRASH" in regime:
                active_threshold = 0.01 # Lower inertia to exit fast
            
            action_type = f"HOLD"
            if turnover > active_threshold:
                action_type = f"TRADE"
                target_vals = current_val * target_weights
                cost = np.sum(np.abs(target_vals[:-1] - (shares*current_prices))) * 0.0010
                cash = target_vals[-1] - cost
                shares = target_vals[:-1] / current_prices
                
            # 5. Simulate
            for t in range(20):
                day_idx = curr_step + t + 1
                if day_idx >= len(self.env.prices): break
                nav = cash + np.sum(shares * self.env.prices[day_idx])
                # Log the regime too so we can color code plots!
                history.append({'step': day_idx, 'nav': nav, 'action': action_type if t==0 else np.nan, 'regime': regime})
            
            curr_step += 20
            
        return pd.DataFrame(history)

class BoardOfDirectorsWithCircuitBreaker:
    def __init__(self, bull_path, bear_path, sniper_path):
        # Load models
        self.bull = PPO.load(bull_path)
        self.bear = PPO.load(bear_path)
        self.sniper = PPO.load(sniper_path)
        
    def predict(self, obs, market_vol_percent):
        """
        Input: 
            obs: The technical features
            market_vol_percent: StdDev of Benchmark (Returns)
        """
        # --- LOGIC GATES ---
        
        # Panic > 2.0% Daily Volatility (approx VIX 32)
        # We INCREASED this slightly to ensure we only trigger on real crashes
        PANIC_LEVEL = 0.020 
        
        # Caution > 1.2% Daily Volatility (approx VIX 19)
        CAUTION_LEVEL = 0.012
        
        # 1. THE CIRCUIT BREAKER (Hard Override)
        if market_vol_percent > PANIC_LEVEL:
            # FORCE CASH. No voting allowed.
            # We construct a weight vector where the last element (Cash) is 1.0
            # We need to know the shape. Since we don't have n_assets here, 
            # we can infer it from a dummy prediction or pass it in init.
            # Safer way: Predict with Bear to get shape, then overwrite.
            dummy_action, _ = self.bear.predict(obs, deterministic=True)
            n_actions = len(dummy_action)
            
            final_weights = np.zeros(n_actions)
            final_weights[-1] = 1.0 # 100% Cash
            
            regime = "CRASH (Circuit Breaker)"
            return final_weights, regime

        # 2. Normal Voting (If no panic)
        p_bull, _ = self.bull.predict(obs, deterministic=True)
        p_bear, _ = self.bear.predict(obs, deterministic=True)
        p_sniper, _ = self.sniper.predict(obs, deterministic=True)
        
        w_bull = self._softmax(p_bull)
        w_bear = self._softmax(p_bear)
        w_sniper = self._softmax(p_sniper)
        
        if market_vol_percent > CAUTION_LEVEL:
            # CHOPPY: 50% Sniper, 25% Bear, 25% Bull
            final_weights = (0.5 * w_sniper) + (0.25 * w_bear) + (0.25 * w_bull)
            regime = "CHOPPY (Mixed)"
            
        else:
            # GROWTH: 70% Bull, 30% Sniper
            final_weights = (0.7 * w_bull) + (0.3 * w_sniper)
            regime = "GROWTH (Bull)"
            
        return final_weights, regime

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

class BoardOfDirectorsV2:
    def __init__(self, bull_path, bear_path, sniper_path):
        self.bull = PPO.load(bull_path)
        self.bear = PPO.load(bear_path)
        self.sniper = PPO.load(sniper_path)
        
    def predict(self, obs, market_vol, asset_rsi_mean, price_vs_sma):
        # --- THRESHOLDS ---
        PANIC_VOL = 0.020      
        CAUTION_VOL = 0.012    
        RSI_OVERSOLD = 30      
        
        # Get raw agent votes
        p_bull, _ = self.bull.predict(obs, deterministic=True)
        p_bear, _ = self.bear.predict(obs, deterministic=True)
        p_sniper, _ = self.sniper.predict(obs, deterministic=True)
        
        w_bull = self._softmax(p_bull)
        w_bear = self._softmax(p_bear)
        w_sniper = self._softmax(p_sniper)
        
        # --- LOGIC TREE ---
        
        # 1. CRASH SCENARIO
        if market_vol > PANIC_VOL:
            # A. MOMENTUM RE-ENTRY CHECK
            if asset_rsi_mean < RSI_OVERSOLD or price_vs_sma:
                # ---------------------------------------------------------
                # FIX: Normalize Asset Weights to ensure sum = 1.0
                # ---------------------------------------------------------
                
                # 1. Isolate Sniper's Asset weights (exclude cash)
                asset_weights = w_sniper[:-1] 
                
                # 2. Normalize them to sum to 1.0 (Relative Conviction)
                # (Add epsilon to avoid divide by zero if sniper was 100% cash)
                asset_weights = asset_weights / (np.sum(asset_weights) + 1e-9)
                
                # 3. Scale Assets to 50% allocation
                asset_weights = asset_weights * 0.5
                
                # 4. Reconstruct vector with 50% Cash
                final_weights = np.append(asset_weights, 0.5)
                
                return final_weights, "CRASH (Re-Entry)"
            
            # B. TRUE PANIC
            else:
                cash_only = np.zeros_like(w_bull)
                cash_only[-1] = 1.0
                return cash_only, "CRASH (Cash)"

        # 2. CHOPPY SCENARIO
        elif market_vol > CAUTION_VOL:
            final_weights = (0.5 * w_sniper) + (0.25 * w_bear) + (0.25 * w_bull)
            return final_weights, "CHOPPY"
            
        # 3. GROWTH SCENARIO
        else:
            final_weights = (0.7 * w_bull) + (0.3 * w_sniper)
            return final_weights, "GROWTH"
            
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


class V2EnsembleBacktester:
    def __init__(self, board, env, threshold=0.05):
        self.board = board
        self.env = env
        self.threshold = threshold
        self.tx_cost_bps = 0.0010
        
    def run(self):
        cash = 100_000
        shares = np.zeros(self.env.n_assets)
        history = []
        
        # Start step (Lookback window)
        curr_step = self.env.lookback_window
        # End step (Leave buffer for final month sim)
        max_step = len(self.env.prices) - 22
        
        print(f"Starting V2 Backtest from step {curr_step} to {max_step}...")
        
        while curr_step < max_step:
            # --- A. Prepare Data ---
            # 1. Get Observation for Agents (Lookback, Assets, Features)
            obs = self.env.features[curr_step - self.env.lookback_window : curr_step]
            
            # --- B. Calculate Signals for the Board ---
            
            # 1. Market Volatility (5-day, annualized approx)
            bench_slice_short = self.env.benchmark[curr_step-5 : curr_step]
            if len(bench_slice_short) > 1:
                # Standard deviation of returns
                vol_short = pd.Series(bench_slice_short).pct_change().std()
            else:
                vol_short = 0.01
            if np.isnan(vol_short): vol_short = 0.01

            # 2. Asset RSI (Momentum)
            # In Feature Engineering, we stored RSI at index 1.
            # Feature shape: (Time, Assets, 3) -> [Vol, RSI, Corr]
            # We get the most recent RSI values for all assets:
            latest_feats = obs[-1] # Shape: (n_assets, 3)
            # Extract Normalized RSI (index 1) and convert back to 0-100 scale
            # (Recall: norm_rsi = (rsi - 50) / 100)
            norm_rsi_values = latest_feats[:, 1] 
            raw_rsi_values = (norm_rsi_values * 100) + 50
            # Average RSI across the portfolio
            avg_rsi = np.mean(raw_rsi_values)

            # 3. Price vs SMA (Trend)
            # We verify if the Market (Benchmark) is above its 50-day SMA
            bench_history_50 = self.env.benchmark[curr_step-50 : curr_step]
            if len(bench_history_50) > 0:
                sma_50 = np.mean(bench_history_50)
                current_bench_price = self.env.benchmark[curr_step]
                price_above_sma = current_bench_price > sma_50
            else:
                price_above_sma = False # Fallback

            # --- C. Get Board Decision ---
            # Pass all 4 signals to the V2 Board
            target_weights, regime = self.board.predict(
                obs, 
                market_vol=vol_short, 
                asset_rsi_mean=avg_rsi, 
                price_vs_sma=price_above_sma
            )
            
            # --- D. Portfolio Wrapper Logic (Inertia & Execution) ---
            current_prices = self.env.prices[curr_step]
            current_val = cash + np.sum(shares * current_prices)
            
            # Calculate Current Weights
            if current_val > 0:
                curr_w = np.append((shares * current_prices) / current_val, cash / current_val)
            else:
                curr_w = np.zeros_like(target_weights)
                curr_w[-1] = 1.0

            turnover = np.sum(np.abs(target_weights - curr_w))
            
            # Dynamic Threshold: 
            # If CRASH logic dictates a move, we lower threshold to execute fast.
            active_threshold = self.threshold
            if "CRASH" in regime:
                active_threshold = 0.01 
            
            action_type = "HOLD"
            if turnover > active_threshold:
                action_type = "TRADE"
                target_vals = current_val * target_weights
                
                # Transaction Costs
                current_assets_val = shares * current_prices
                diffs = np.abs(target_vals[:-1] - current_assets_val)
                cost = np.sum(diffs) * self.tx_cost_bps
                
                cash = target_vals[-1] - cost
                shares = target_vals[:-1] / current_prices
                
            # --- E. Simulation (Time Travel) ---
            for t in range(20):
                day_idx = curr_step + t + 1
                if day_idx >= len(self.env.prices): break
                
                nav = cash + np.sum(shares * self.env.prices[day_idx])
                
                history.append({
                    'step': day_idx, 
                    'nav': nav, 
                    'regime': regime,
                    'action': action_type if t==0 else None,
                    'vol': vol_short,
                    'rsi': avg_rsi # Log this to verify re-entry
                })
            
            curr_step += 20
            
        return pd.DataFrame(history)