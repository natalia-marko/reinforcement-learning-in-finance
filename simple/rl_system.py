import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
from config import TX_COST_BPS, RISK_AVERSION

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ==========================================
# 1. THE BOARD OF DIRECTORS (Dual-Trigger Logic)
# ==========================================
class FastBoardOfDirectors:
    def __init__(self, bull_path, bear_path, sniper_path):
        print("Loading Agents...")
        self.bull = PPO.load(bull_path)
        self.bear = PPO.load(bear_path)
        self.sniper = PPO.load(sniper_path)
        print("Agents Loaded.")
        
    def predict(self, obs, short_term_vol, long_term_vol):
        """
        Input: 
            obs: The technical features for the agents
            short_term_vol: 5-day Volatility (Fast reaction)
            long_term_vol: 20-day Volatility (Regime stability)
        """
        
        # --- 1. THE CIRCUIT BREAKER (Updated Logic) ---
        # Trigger if SHORT term spikes (Immediate danger)
        # OR if LONG term is elevated (Sustained bear market)
        
        # Thresholds (Daily Standard Deviation of Returns)
        # 0.015 = 1.5% daily moves (approx VIX 24) -> Fast Panic
        # 0.020 = 2.0% daily moves (approx VIX 32) -> Sustained Panic
        PANIC_FAST = 0.015  
        PANIC_SLOW = 0.020  
        
        if short_term_vol > PANIC_FAST or long_term_vol > PANIC_SLOW:
            # FORCE CASH (Circuit Breaker)
            # We predict with one agent just to get the shape of the action vector
            dummy_action, _ = self.bear.predict(obs, deterministic=True)
            
            # Create a weight vector where everything is 0.0 except the last index (Cash)
            final_weights = np.zeros(len(dummy_action))
            final_weights[-1] = 1.0 
            
            return final_weights, "CRASH (Circuit Breaker)"

        # --- 2. Normal Voting (If Market is Safe) ---
        # Get raw logits
        p_bull, _ = self.bull.predict(obs, deterministic=True)
        p_bear, _ = self.bear.predict(obs, deterministic=True)
        p_sniper, _ = self.sniper.predict(obs, deterministic=True)
        
        # Convert to probabilities
        w_bull = self._softmax(p_bull)
        w_bear = self._softmax(p_bear)
        w_sniper = self._softmax(p_sniper)
        
        # Use Long Term vol for the "Choppy vs Growth" decision (smoother transition)
        if long_term_vol > 0.012:
            # CHOPPY: Mixed Allocation
            final_weights = (0.5 * w_sniper) + (0.25 * w_bear) + (0.25 * w_bull)
            regime = "CHOPPY (Mixed)"
        else:
            # GROWTH: Aggressive Allocation
            final_weights = (0.7 * w_bull) + (0.3 * w_sniper)
            regime = "GROWTH (Bull)"
            
        return final_weights, regime

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()



# ==========================================
# 2. THE BACKTESTER (Calculates Fast/Slow Vol)
# ==========================================
class FastEnsembleBacktester:
    def __init__(self, board, env, threshold=0.05):
        self.board = board
        self.env = env
        self.threshold = threshold
        self.tx_cost_bps = 0.0010 # 0.10%
        
    def run(self):
        cash = 100_000
        shares = np.zeros(self.env.n_assets)
        history = []
        
        curr_step = self.env.lookback_window
        max_step = len(self.env.prices) - 22
        
        print(f"Starting Fast Backtest from step {curr_step} to {max_step}...")
        
        while curr_step < max_step:
            # --- A. Prepare Inputs ---
            # 1. Get Observation
            obs = self.env.features[curr_step - self.env.lookback_window : curr_step]
            
            # 2. Volatility Calculations (Crucial Fix: Use Returns, not Prices)
            
            # Long Term (20 days) - The "Regime" Detector
            bench_slice_long = self.env.benchmark[curr_step-20 : curr_step]
            if len(bench_slice_long) > 1:
                vol_long = pd.Series(bench_slice_long).pct_change().std()
            else:
                vol_long = 0.01
            
            # Short Term (5 days) - The "Fast Trigger"
            bench_slice_short = self.env.benchmark[curr_step-5 : curr_step]
            if len(bench_slice_short) > 1:
                vol_short = pd.Series(bench_slice_short).pct_change().std()
            else:
                vol_short = 0.01
            
            # Safety fill NaNs
            if np.isnan(vol_long): vol_long = 0.01
            if np.isnan(vol_short): vol_short = 0.01
            
            # --- B. Get Decision ---
            target_weights, regime = self.board.predict(obs, vol_short, vol_long)
            
            # --- C. Wrapper Logic (Inertia) ---
            current_prices = self.env.prices[curr_step]
            current_val = cash + np.sum(shares * current_prices)
            
            # Calculate Current Weights
            if current_val > 0:
                curr_w = np.append((shares * current_prices) / current_val, cash / current_val)
            else:
                # Fallback if 100% cash or broke
                curr_w = np.zeros_like(target_weights)
                curr_w[-1] = 1.0 # Assume all cash

            turnover = np.sum(np.abs(target_weights - curr_w))
            
            # Dynamic Threshold: If in CRASH mode, remove the barrier to sell.
            active_threshold = self.threshold
            if "CRASH" in regime:
                active_threshold = 0.01 # Exit immediately
            
            action_type = "HOLD"
            if turnover > active_threshold:
                action_type = "TRADE"
                # Execute Trade
                target_vals = current_val * target_weights
                
                # Calculate Cost (simplification: cost on delta)
                current_assets_val = shares * current_prices
                diffs = np.abs(target_vals[:-1] - current_assets_val)
                cost = np.sum(diffs) * self.tx_cost_bps
                
                cash = target_vals[-1] - cost
                shares = target_vals[:-1] / current_prices
                
            # --- D. Simulation (Time Travel 20 days) ---
            for t in range(20):
                day_idx = curr_step + t + 1
                if day_idx >= len(self.env.prices): break
                
                nav = cash + np.sum(shares * self.env.prices[day_idx])
                
                history.append({
                    'step': day_idx, 
                    'nav': nav, 
                    'regime': regime,
                    'action': action_type if t==0 else None,
                    'vol_short': vol_short, # Log for debugging
                    'vol_long': vol_long
                })
            
            curr_step += 20
            
        return pd.DataFrame(history)
