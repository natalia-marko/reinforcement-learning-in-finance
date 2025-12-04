"""
Sentiment Environment - Optimized for Monthly Data with Log Returns

Key Features:
- Uses log returns as rewards (more stable for RL training)
- Proper monthly data handling (Sharpe ratio annualization)
- Robust weight normalization
- NaN handling
"""

import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd


class SentimentEnv(gymnasium.Env):
    """
    Sentiment-based trading environment for portfolio optimization.
    
    Designed for MONTHLY data with log returns.
    """
    
    def __init__(self, price_data, sentiment_features, initial_capital=100000, 
                 transaction_cost=0.001, verbose=0,
                 alpha_returns=1.0, alpha_mdd=0.3, alpha_vol=0.3, exploration_bias=0.01,
                 vol_window=None, **kwargs):
        super().__init__()
        
        # Store data
        self.price_data = price_data
        self.sentiment_features = sentiment_features
        self.n_assets = len(price_data.columns)
        self.initial_capital = initial_capital
        
        # Risk and cost parameters
        self.transaction_cost = transaction_cost
        self.verbose = verbose
        
        # Aligned reward function parameters (matching super/meta hierarchy)
        # Base agents use slightly lower penalties to encourage exploration
        self.alpha_returns = alpha_returns
        self.alpha_mdd = alpha_mdd
        self.alpha_vol = alpha_vol
        self.exploration_bias = exploration_bias
        
        # Backward compatibility: vol_window is no longer used with aligned reward
        if vol_window is not None and verbose > 0:
            print(f"Warning: vol_window parameter is deprecated and ignored with aligned reward function.")
        
        # For risk-adjusted returns calculation
        self.returns_history = []
        
        # Action space: portfolio weights [0, 1] for each asset
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_assets,), 
            dtype=np.float32
        )
        
        # Observation space: sentiment features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(sentiment_features.shape[1],), 
            dtype=np.float32
        )
        
        # Find common dates between sentiment and prices
        self.common_dates = sentiment_features.index.intersection(price_data.index)
        
        if len(self.common_dates) < 2:
            raise ValueError(f"Not enough overlapping dates! Only {len(self.common_dates)} found. "
                           f"Price data: {len(price_data)} dates, "
                           f"Sentiment data: {len(sentiment_features)} dates")
        
        if self.verbose > 0:
            print(f"Sentiment Environment initialized:")
            print(f"  - Assets: {self.n_assets}")
            print(f"  - Sentiment features: {sentiment_features.shape[1]}")
            print(f"  - Time steps: {len(self.common_dates)}")
            print(f"  - Date range: {self.common_dates[0]} to {self.common_dates[-1]}")
        
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.initial_capital]
        self.log_returns_history = []
        self.returns_history = []
        
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        """Get current observation (sentiment features)"""
        if self.current_step >= len(self.common_dates):
            self.current_step = len(self.common_dates) - 1
            
        current_date = self.common_dates[self.current_step]
        obs = self.sentiment_features.loc[current_date].values
        
        # Handle NaN values
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs.astype(np.float32)

    def step(self, action):
        """
        Execute one step in the environment.
        
        Uses LOG RETURNS as reward for more stable training.
        """
        # Normalize action to valid portfolio weights
        action = np.clip(action, 0, 1)
        total = action.sum()
        
        if total < 1e-6:
            # If all weights are zero, use equal weighting
            weights = np.ones(self.n_assets) / self.n_assets
        else:
            weights = action / total
        
        # Get current and next dates
        current_date = self.common_dates[self.current_step]
        
        if self.current_step + 1 < len(self.common_dates):
            next_date = self.common_dates[self.current_step + 1]
        else:
            # Last step, use current date (no return)
            next_date = current_date
        
        # Calculate log returns for each asset
        log_returns = np.log(
            self.price_data.loc[next_date] / self.price_data.loc[current_date]
        )
        
        # Handle NaN, inf values in returns
        log_returns = log_returns.fillna(0)
        log_returns = np.nan_to_num(log_returns.values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate portfolio return correctly
        # Convert log returns to arithmetic returns
        arithmetic_returns = np.exp(log_returns) - 1
        
        # Portfolio arithmetic return (weighted sum)
        portfolio_return = np.sum(weights * arithmetic_returns)
        
        # Apply transaction costs
        prev_weights = self.weights.copy()
        transaction_cost = self.transaction_cost * np.sum(np.abs(weights - prev_weights))
        portfolio_return -= transaction_cost
        
        # Convert portfolio return to log return
        portfolio_log_return = np.log(1 + portfolio_return)
        
        # Store for metrics calculation
        self.log_returns_history.append(portfolio_log_return)
        self.returns_history.append(portfolio_return)
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # Calculate aligned composite reward (matching super/meta agents)
        # This ensures coherent objectives throughout the hierarchy
        
        # Maximum drawdown
        if len(self.portfolio_history) > 1:
            portfolio_series = pd.Series(self.portfolio_history)
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak
            current_mdd = abs(drawdown.iloc[-1])
        else:
            current_mdd = 0.0
        
        # Volatility (recent)
        if len(self.returns_history) > 1:
            volatility = np.std(self.returns_history)
        else:
            volatility = 0.0
        
        # Composite reward aligned with hierarchy
        # Base agents use lower penalties than super/meta to encourage exploration
        reward = (self.alpha_returns * portfolio_log_return - 
                 self.alpha_mdd * current_mdd - 
                 self.alpha_vol * volatility + 
                 self.exploration_bias)

        # Store weights and increment step
        self.weights = weights
        self.current_step += 1
        done = self.current_step >= len(self.common_dates) - 1
        
        # Info dictionary
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "portfolio_log_return": portfolio_log_return,
            "weights": weights.copy(),
            "step": self.current_step
        }
        
        # Get next observation
        obs = self._get_observation()
        
        # Return (obs, reward, done, truncated, info)
        return obs, reward, done, False, info

    def get_portfolio_metrics(self):
        """
        Calculate portfolio performance metrics.
        
        IMPORTANT: Adjusted for MONTHLY data frequency.
        - Annualization factor: 12 (not 252)
        - Sharpe ratio: * sqrt(12)
        - Volatility: * sqrt(12)
        """
        portfolio_series = pd.Series(self.portfolio_history)
        
        # Calculate returns from portfolio values
        returns = portfolio_series.pct_change().dropna()
        
        # Total return
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (annualized for MONTHLY data)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(12)  # 12 months, not 252 days
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        if len(portfolio_series) > 0:
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0.0
        
        # Volatility (annualized for MONTHLY data)
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(12)  # 12 months, not 252 days
        else:
            volatility = 0.0
        
        # Average log return (optional, useful for RL analysis)
        if len(self.log_returns_history) > 0:
            avg_log_return = np.mean(self.log_returns_history)
        else:
            avg_log_return = 0.0
        
        # Win rate
        if len(returns) > 0:
            win_rate = (returns > 0).sum() / len(returns)
        else:
            win_rate = 0.0
        
        # Calmar ratio (return / max drawdown)
        if max_drawdown > 0:
            calmar_ratio = total_return / max_drawdown
        else:
            calmar_ratio = 0.0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "final_value": self.portfolio_value,
            "avg_log_return": avg_log_return,
            "win_rate": win_rate,
            "calmar_ratio": calmar_ratio,
            "num_periods": len(returns)
        }
    
    def get_detailed_metrics(self):
        """Get more detailed performance metrics"""
        metrics = self.get_portfolio_metrics()
        
        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()
        
        if len(returns) > 0:
            # Sortino ratio (downside deviation, annualized for monthly)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(12)
            else:
                sortino_ratio = metrics['sharpe_ratio']  # Fallback to Sharpe
            
            # Max consecutive wins/losses
            win_streak = 0
            loss_streak = 0
            current_win_streak = 0
            current_loss_streak = 0
            
            for ret in returns:
                if ret > 0:
                    current_win_streak += 1
                    current_loss_streak = 0
                    win_streak = max(win_streak, current_win_streak)
                elif ret < 0:
                    current_loss_streak += 1
                    current_win_streak = 0
                    loss_streak = max(loss_streak, current_loss_streak)
            
            # Best and worst month
            best_month = returns.max()
            worst_month = returns.min()
            
            metrics.update({
                "sortino_ratio": sortino_ratio,
                "best_month": best_month,
                "worst_month": worst_month,
                "max_win_streak": win_streak,
                "max_loss_streak": loss_streak
            })
        
        return metrics
