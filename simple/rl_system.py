import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    """
    Advanced Portfolio Environment (Gymnasium Version).
    """
    def __init__(self, df, tickers, lookback_window=1, rebalance_period=4, initial_balance=10000):
        super(PortfolioEnv, self).__init__()
        
        self.df = df
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.rebalance_period = rebalance_period 
        self.initial_balance = initial_balance
        
        # Data parsing
        self.ret_cols = [f'{t}_log_ret' for t in tickers]
        self.feat_cols = [c for c in df.columns if c not in self.ret_cols]
        
        self.data_matrix = df[self.feat_cols].values
        self.return_matrix = df[self.ret_cols].values
        
        # Gymnasium Action Space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        
        # Gymnasium Observation Space
        n_features = len(self.feat_cols)
        self.obs_shape = (n_features) + self.n_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        # Gymnasium requires seed in reset
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.balance]
        
        # Gymnasium returns: observation, info
        return self._get_observation(), {}

    def _get_observation(self):
        market_obs = self.data_matrix[self.current_step]
        obs = np.concatenate([market_obs, self.current_weights])
        return obs.astype(np.float32)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def step(self, action):
        target_weights = self._softmax(action)
        
        # Transaction Cost (10bps)
        turnover = np.sum(np.abs(target_weights - self.current_weights))
        cost = self.balance * turnover * 0.0010 
        
        period_log_returns = []
        end_step = min(self.current_step + self.rebalance_period, len(self.df) - 1)
        
        # Check for termination BEFORE simulating
        if self.current_step >= len(self.df) - 1:
            # Gymnasium expects: obs, reward, terminated, truncated, info
            return self._get_observation(), 0.0, True, False, {}

        current_period_balance = self.balance - cost
        
        for t in range(self.current_step, end_step):
            r_t_log = self.return_matrix[t]
            r_t_simple = np.exp(r_t_log) - 1
            port_ret = np.dot(target_weights, r_t_simple)
            current_period_balance *= (1 + port_ret)
            period_log_returns.append(np.log(1 + port_ret))
            
        self.balance = current_period_balance
        self.portfolio_history.append(self.balance)
        
        # --- REWARD FIX ---
        returns_arr = np.array(period_log_returns)
        avg_ret = np.mean(returns_arr)
        negative_returns = returns_arr[returns_arr < 0]
        
        # If no negative returns (perfect month), assume a small baseline risk (1%) 
        # to prevent division by 1e-6 which causes reward explosion (1.8 Million)
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns) + 1e-5
        else:
            downside_std = 0.01 
            
        # Reward is now stable (~ -2 to +20 range)
        reward = (avg_ret / downside_std)
        reward -= (turnover * 0.1) # Reduced penalty

        self.current_step = end_step
        self.current_weights = target_weights
        
        terminated = self.current_step >= len(self.df) - 1
        truncated = False # Gymnasium requirement
        
        info = {
            'balance': self.balance,
            'weights': target_weights
        }
        
        return self._get_observation(), reward, terminated, truncated, info