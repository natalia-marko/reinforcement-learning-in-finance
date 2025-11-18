"""
Portfolio Environment for Reinforcement Learning
Implements Gymnasium-based environment for portfolio rebalancing.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple


class PortfolioEnv(gym.Env):
    """
    Portfolio rebalancing environment for RL.
    
    State: Asset features + current weights + portfolio stats + time embeddings
    Action: Weight allocation [0,1] for each asset (softmax normalized to sum=1)
    Reward: Sharpe ratio calculated over 4-week rolling window
    Constraints: Max 40% per asset, transaction costs (2.5 bps per trade)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        tickers: List[str],
        rebalance_frequency: int = 4,
        transaction_cost: float = 0.00025,
        max_weight_per_asset: float = 0.4,
        reward_lookback: int = 4,
        initial_capital: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize portfolio environment.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Preprocessed data with features, dates, tickers, and prices
        feature_cols : List[str]
            List of feature column names
        tickers : List[str]
            List of asset tickers (must match data['ticker'].unique())
        rebalance_frequency : int
            Rebalance every N weeks (default: 4 for monthly)
        transaction_cost : float
            Transaction cost per trade (default: 0.00025 = 2.5 bps)
        max_weight_per_asset : float
            Maximum weight per asset (default: 0.4 = 40%)
        reward_lookback : int
            Weeks to look back for Sharpe ratio calculation (default: 4)
        initial_capital : float
            Initial portfolio value (default: 1.0)
        seed : Optional[int]
            Random seed for reproducibility
        """
        super().__init__()
        
        # Ensure date column is datetime
        data = data.copy()
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        
        self.data = data.sort_values(['date', 'ticker']).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.tickers = sorted(tickers)
        self.n_assets = len(self.tickers)
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.max_weight_per_asset = max_weight_per_asset
        self.reward_lookback = reward_lookback
        self.initial_capital = initial_capital
        
        # Get unique dates (sorted)
        self.dates = sorted(self.data['date'].unique())
        self.n_periods = len(self.dates)
        
        # State space: features per asset + current weights + portfolio stats + time embeddings
        n_features_per_asset = len(feature_cols)
        n_portfolio_stats = 3  # portfolio_return, portfolio_vol, portfolio_sharpe
        n_time_features = 2  # sin/cos month embeddings
        
        self.state_dim = (
            n_features_per_asset * self.n_assets +  # Asset features
            self.n_assets +  # Current weights
            n_portfolio_stats +  # Portfolio statistics
            n_time_features  # Time embeddings
        )
        
        # Action space: weight allocation for each asset (will be softmax normalized)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset(seed=seed)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets  # Equal weights initially
        self.portfolio_returns = []
        self.rebalance_count = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
            'step': self.current_step
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Parameters:
        -----------
        action : np.ndarray
            Raw action (will be softmax normalized and constrained)
        
        Returns:
        --------
        observation : np.ndarray
            Next state observation
        reward : float
            Reward for this step
        terminated : bool
            Whether episode is terminated
        truncated : bool
            Whether episode is truncated
        info : dict
            Additional information
        """
        # Normalize action to valid portfolio weights
        new_weights = self._normalize_action(action)
        
        # Check if we should rebalance (every N weeks)
        should_rebalance = (self.current_step % self.rebalance_frequency == 0)
        
        if should_rebalance:
            # Calculate transaction costs
            weight_change = np.abs(new_weights - self.weights)
            transaction_cost_amount = np.sum(weight_change) * self.transaction_cost
            self.portfolio_value *= (1 - transaction_cost_amount)
            
            # Update weights
            self.weights = new_weights.copy()
            self.rebalance_count += 1
        
        # Calculate portfolio return for this period
        current_date = self.dates[self.current_step]
        period_return = self._calculate_portfolio_return(current_date)

        # Update portfolio value (use exp for log returns)
        self.portfolio_value *= np.exp(period_return)
        self.portfolio_returns.append(period_return)
        
        # Calculate reward (Sharpe ratio over lookback window)
        reward = self._calculate_reward()
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.n_periods - 1
        truncated = False
        
        # Get next observation
        observation = self._get_observation()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
            'period_return': period_return,
            'cumulative_return': (self.portfolio_value / self.initial_capital) - 1.0,
            'step': self.current_step,
            'rebalance_count': self.rebalance_count
        }
        
        return observation, reward, terminated, truncated, info
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action to valid portfolio weights.
        
        Constraints:
        - Sum to 1.0
        - Each weight <= max_weight_per_asset
        """
        # Softmax normalization
        action = np.clip(action, 0, 1)
        weights = action / (np.sum(action) + 1e-10)
        
        # Apply max weight constraint
        weights = np.clip(weights, 0, self.max_weight_per_asset)
        
        # Renormalize to sum to 1
        weights = weights / (np.sum(weights) + 1e-10)
        
        return weights.astype(np.float32)
    
    def _calculate_portfolio_return(self, date: pd.Timestamp) -> float:
        """Calculate portfolio return for current period."""
        # Get returns for each asset for this date
        asset_returns = []
        for ticker in self.tickers:
            ticker_data = self.data[
                (self.data['date'] == date) & 
                (self.data['ticker'] == ticker)
            ]
            if len(ticker_data) > 0:
                # Calculate log return from previous period
                prev_date_idx = self.current_step - 1
                if prev_date_idx >= 0:
                    prev_date = self.dates[prev_date_idx]
                    prev_data = self.data[
                        (self.data['date'] == prev_date) & 
                        (self.data['ticker'] == ticker)
                    ]
                    if len(prev_data) > 0:
                        prev_price = prev_data['close'].iloc[0]
                        curr_price = ticker_data['close'].iloc[0]
                        # Calculate log return (consistent with data preparation)
                        asset_return = np.log(curr_price / prev_price)
                        asset_returns.append(asset_return)
                    else:
                        asset_returns.append(0.0)
                else:
                    asset_returns.append(0.0)
            else:
                asset_returns.append(0.0)
        
        # Portfolio return = weighted sum of asset returns
        if len(asset_returns) == self.n_assets:
            portfolio_return = np.dot(self.weights, asset_returns)
        else:
            portfolio_return = 0.0
        
        return float(portfolio_return)
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward as Sharpe ratio over lookback window.
        
        Reward = mean(returns) / (std(returns) + eps) * sqrt(52) for annualization
        """
        if len(self.portfolio_returns) < 2:
            return 0.0
        
        # Use last N returns for Sharpe calculation
        recent_returns = self.portfolio_returns[-self.reward_lookback:]
        
        if len(recent_returns) < 2:
            return 0.0
        
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if std_return < 1e-8:
            return 0.0
        
        # Annualized Sharpe ratio (52 weeks per year)
        sharpe = (mean_return / std_return) * np.sqrt(52)
        
        return float(sharpe)
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        if self.current_step >= len(self.dates):
            # Return zero observation if out of bounds
            return np.zeros(self.state_dim, dtype=np.float32)
        
        current_date = self.dates[self.current_step]
        
        # 1. Asset features (flattened)
        asset_features = []
        for ticker in self.tickers:
            ticker_data = self.data[
                (self.data['date'] == current_date) & 
                (self.data['ticker'] == ticker)
            ]
            if len(ticker_data) > 0:
                features = ticker_data[self.feature_cols].iloc[0].values
                asset_features.extend(features)
            else:
                # Fill with zeros if data missing
                asset_features.extend([0.0] * len(self.feature_cols))
        
        # 2. Current portfolio weights
        weights_features = self.weights.tolist()
        
        # 3. Portfolio statistics
        if len(self.portfolio_returns) >= 2:
            portfolio_return = np.mean(self.portfolio_returns[-self.reward_lookback:])
            portfolio_vol = np.std(self.portfolio_returns[-self.reward_lookback:])
            portfolio_sharpe = self._calculate_reward()
        else:
            portfolio_return = 0.0
            portfolio_vol = 0.0
            portfolio_sharpe = 0.0
        
        portfolio_stats = [portfolio_return, portfolio_vol, portfolio_sharpe]
        
        # 4. Time embeddings (month sin/cos)
        month = current_date.month
        time_features = [
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12)
        ]
        
        # Combine all features
        observation = np.array(
            asset_features + weights_features + portfolio_stats + time_features,
            dtype=np.float32
        )
        
        return observation
    
    def get_portfolio_stats(self) -> Dict:
        """Get current portfolio statistics."""
        if len(self.portfolio_returns) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'portfolio_value': float(self.portfolio_value)
            }

        returns = np.array(self.portfolio_returns)
        # For log returns, cumulative is sum of log returns, then exponentiate
        cumulative = np.exp(np.cumsum(returns))

        total_return = cumulative[-1] - 1.0
        sharpe = self._calculate_reward()
        volatility = np.std(returns) * np.sqrt(52)

        # Max drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'volatility': float(volatility),
            'max_drawdown': float(max_drawdown),
            'portfolio_value': float(self.portfolio_value)
        }

