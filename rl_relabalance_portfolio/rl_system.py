"""
RL Portfolio Rebalancing System - REFACTORED
============================================
Refactored to use config parameters instead of hardcoded values.
Fixed reward function consistency issues.

Components:
- Monitoring and callbacks (PortfolioMonitor, TrainingLogger)
- Neural network models (SimpleActor)
- Portfolio environment (PortfolioEnv)
- Walk-forward validation utilities
"""

__all__ = [
    # Callbacks & Monitoring
    'PortfolioMonitor',
    'TrainingLogger',
    # Models
    'SimpleActor',
    # Environment
    'PortfolioEnv',
    # Validation Utilities
    'create_walk_forward_folds',
    'print_fold_summary',
    # Baseline Strategies
    'BaselineStrategy',
    'EqualWeight',
    'BuyHold',
    'RiskParity',
    'Momentum',
    'MinVariance',
    # Backtest Functions
    'backtest_rl_agent',
    'backtest_baseline'
]

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.callbacks import BaseCallback


# ============================================================================
# CALLBACKS & MONITORING
# ============================================================================

class PortfolioMonitor(gym.Wrapper):
    """
    Custom Monitor that stores portfolio stats before environment resets.

    This fixes the issue where get_portfolio_stats() returns zeros after reset.
    Wraps a PortfolioEnv and tracks episode statistics properly across resets.
    """

    def __init__(self, env):
        """
        Initialize PortfolioMonitor.

        Parameters:
        -----------
        env : gym.Env
            The environment to wrap (typically PortfolioEnv)
        """
        super().__init__(env)
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_portfolio_stats = []
        self.current_episode_rewards = []

    def reset(self, **kwargs):
        """Reset environment and store final stats from previous episode."""
        # Store final stats before reset
        if hasattr(self.env, 'get_portfolio_stats') and len(self.current_episode_rewards) > 0:
            stats = self.env.get_portfolio_stats()
            self.episode_portfolio_stats.append(stats)

        # Reset tracking
        self.current_episode_rewards = []

        return self.env.reset(**kwargs)

    def step(self, action):
        """Execute step and track rewards/stats."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_episode_rewards.append(reward)

        # If episode done, store stats in info
        if terminated or truncated:
            stats = self.env.get_portfolio_stats()
            info['portfolio_stats'] = stats
            self.episode_returns.append(sum(self.current_episode_rewards))
            self.episode_lengths.append(len(self.current_episode_rewards))

        return obs, reward, terminated, truncated, info


class TrainingLogger(BaseCallback):
    """
    Comprehensive logging callback for PPO training.

    Tracks training and validation metrics, implements early stopping,
    and saves logs to disk. Works with PortfolioMonitor to properly
    extract portfolio statistics.

    Features:
    - Episode-level training metrics
    - Periodic validation evaluation
    - Early stopping based on validation Sharpe ratio
    - CSV logging of all metrics
    - Best model checkpointing
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        fold_dir: Path,
        patience: int = 20,  # Using config default
        verbose: int = 1
    ):
        """
        Initialize TrainingLogger.

        Parameters:
        -----------
        eval_env : VecEnv
            Vectorized validation environment
        eval_freq : int
            Evaluate every N steps
        fold_dir : Path
            Directory to save logs and models
        patience : int
            Early stopping patience (default: 20 evaluations)
        verbose : int
            Verbosity level (0=silent, 1=info)
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.fold_dir = Path(fold_dir)
        self.patience = patience

        # Tracking
        self.best_val_sharpe = -np.inf
        self.no_improvement_count = 0
        self.early_stopped = False

        # Logging lists
        self.train_episodes = []
        self.val_evaluations = []

        # Create log directory
        (self.fold_dir / 'logs').mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Called at every step."""
        # Track episode completion
        infos = self.locals.get('infos', [])
        for info in infos:
            # Check if episode just completed (uses portfolio_stats from PortfolioMonitor)
            if 'portfolio_stats' in info:
                stats = info['portfolio_stats']

                self.train_episodes.append({
                    'step': self.num_timesteps,
                    'episode': len(self.train_episodes),
                    'total_return': stats.get('total_return', 0.0),
                    'sharpe_ratio': stats.get('sharpe_ratio', 0.0),
                    'volatility': stats.get('volatility', 0.0),
                    'max_drawdown': stats.get('max_drawdown', 0.0),
                    'portfolio_value': stats.get('portfolio_value', 0.0)
                })

                # Print progress
                if self.verbose > 0 and len(self.train_episodes) % 20 == 0:
                    print(f"  Train Episode {len(self.train_episodes)}: "
                          f"Sharpe={stats['sharpe_ratio']:.3f}, "
                          f"Return={stats['total_return']*100:.2f}%")

        # Validation evaluation
        should_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if should_eval:
            self._run_validation()

            # Check early stopping
            if self.no_improvement_count >= self.patience:
                print(f"\n  Early stopping at step {self.num_timesteps}")
                self.early_stopped = True
                return False

        return True

    def _run_validation(self):
        """Run full validation episode and log results."""
        # Run complete validation episode
        obs = self.eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            episode_reward += reward[0]
            episode_length += 1

            if done[0]:
                done = True
                # Get stats from info (PortfolioMonitor stores it)
                if 'portfolio_stats' in info[0]:
                    stats = info[0]['portfolio_stats']
                else:
                    # Fallback: try to get from environment
                    val_env_unwrapped = self.eval_env.envs[0]
                    while hasattr(val_env_unwrapped, 'env'):
                        val_env_unwrapped = val_env_unwrapped.env
                    stats = val_env_unwrapped.get_portfolio_stats()

        # Log validation results
        val_sharpe = stats.get('sharpe_ratio', 0.0)
        self.val_evaluations.append({
            'step': self.num_timesteps,
            'evaluation': len(self.val_evaluations),
            'reward': episode_reward,
            'length': episode_length,
            'total_return': stats.get('total_return', 0.0),
            'sharpe_ratio': val_sharpe,
            'volatility': stats.get('volatility', 0.0),
            'max_drawdown': stats.get('max_drawdown', 0.0),
            'portfolio_value': stats.get('portfolio_value', 0.0)
        })

        # Print validation update
        if self.verbose > 0:
            print(f"\n  Validation at step {self.num_timesteps}: "
                  f"Sharpe={val_sharpe:.3f}, "
                  f"Return={stats['total_return']*100:.2f}%, "
                  f"Drawdown={stats['max_drawdown']*100:.2f}%")

        # Check if this is the best model
        if val_sharpe > self.best_val_sharpe:
            self.best_val_sharpe = val_sharpe
            self.no_improvement_count = 0

            # Save best model
            model_path = self.fold_dir / 'best_model.zip'
            self.model.save(str(model_path))
            print(f"  → New best model saved (Sharpe: {val_sharpe:.3f})")
        else:
            self.no_improvement_count += 1
            if self.verbose > 0:
                print(f"  → No improvement for {self.no_improvement_count}/{self.patience} evaluations")

    def _on_training_end(self) -> None:
        """Called at end of training."""
        # Save all logs to CSV
        if self.train_episodes:
            train_df = pd.DataFrame(self.train_episodes)
            train_df.to_csv(self.fold_dir / 'logs' / 'train_episodes.csv', index=False)

        if self.val_evaluations:
            val_df = pd.DataFrame(self.val_evaluations)
            val_df.to_csv(self.fold_dir / 'logs' / 'val_evaluations.csv', index=False)

        # Summary
        print(f"\n  Training Summary:")
        print(f"    Total episodes: {len(self.train_episodes)}")
        print(f"    Total validations: {len(self.val_evaluations)}")
        print(f"    Best validation Sharpe: {self.best_val_sharpe:.3f}")
        print(f"    Early stopped: {self.early_stopped}")


# ============================================================================
# MODEL
# ============================================================================

class SimpleActor(nn.Module):
    """
    Simplified actor network for RL agent.
    
    FIXED: Removed complex hierarchical structure that caused instability.
    Now uses simple feed-forward network with residual connections.
    """
    
    def __init__(self, state_dim: int, n_assets: int, hidden_dim: int = 256):
        """
        Initialize SimpleActor.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state observation
        n_assets : int
            Number of assets in portfolio
        hidden_dim : int
            Hidden layer size (default: 256)
        """
        super().__init__()
        
        # Simplified architecture with layer normalization
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, n_assets)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        state : torch.Tensor
            State observation
            
        Returns:
        --------
        torch.Tensor
            Portfolio weights (after softmax)
        """
        # Layer 1 with residual
        x = F.relu(self.ln1(self.fc1(state)))
        
        # Layer 2 with residual connection
        x_res = x
        x = F.relu(self.ln2(self.fc2(x)))
        x = x + x_res  # Residual connection
        
        # Output layer
        weights = F.softmax(self.fc3(x), dim=-1)
        
        return weights


# ============================================================================
# ENVIRONMENT
# ============================================================================

class PortfolioEnv(gym.Env):
    """
    Portfolio rebalancing environment for RL training.
    
    REFACTORED: Now uses config parameters instead of hardcoded values.
    FIXED: Reward function consistency issues.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        tickers: List[str],
        rebalance_frequency: int,
        transaction_cost: float,
        max_weight_per_asset: float,
        reward_lookback: int,
        initial_capital: float,
        reward_type: str = 'risk_aware',
        date_col: str = 'date',
        seed: int = 42,
        # New config parameters for reward functions
        vol_target: float = 0.35,  # Target volatility for penalties
        min_entropy: float = 0.5,  # Minimum portfolio entropy
        max_turnover: float = 0.20,  # Maximum acceptable turnover
        annualization_factor: int = 52,  # For weekly data
        reward_weights: Optional[Dict] = None  # Reward weights from config
    ):
        """
        Initialize PortfolioEnv.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with columns: date, ticker, close, and features
        feature_cols : List[str]
            List of feature column names
        tickers : List[str]
            List of ticker symbols
        rebalance_frequency : int
            Rebalance every N periods
        transaction_cost : float
            Transaction cost per unit of turnover
        max_weight_per_asset : float
            Maximum weight per asset (e.g., 0.4 for 40%)
        reward_lookback : int
            Lookback window for reward calculation
        initial_capital : float
            Starting capital
        reward_type : str
            Type of reward function ('log_return', 'multi_component', 'risk_aware')
        date_col : str
            Name of date column
        seed : int
            Random seed for reproducibility
        vol_target : float
            Target volatility for risk_aware reward
        min_entropy : float
            Minimum acceptable portfolio entropy
        max_turnover : float
            Maximum acceptable turnover rate
        annualization_factor : int
            Factor for annualizing metrics (52 for weekly, 252 for daily)
        reward_weights : Dict, optional
            Dictionary of reward component weights and normalization factors.
            If None, uses default values from config.REWARD_WEIGHTS
        """
        super().__init__()

        # Store config parameters
        self.data = data
        self.feature_cols = feature_cols
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.max_weight_per_asset = max_weight_per_asset
        self.reward_lookback = reward_lookback
        self.initial_capital = initial_capital
        self.reward_type = reward_type
        self.date_col = date_col

        # New config parameters
        self.vol_target = vol_target
        self.min_entropy = min_entropy
        self.max_turnover = max_turnover
        self.annualization_factor = annualization_factor

        # Load reward weights from config if not provided
        if reward_weights is None:
            # Import here to avoid circular imports
            try:
                from config import REWARD_WEIGHTS
                self.reward_weights = REWARD_WEIGHTS
            except ImportError:
                # Fallback to empty dict if config not found
                self.reward_weights = {}
                print("Warning: Could not load REWARD_WEIGHTS from config. Using hardcoded values.")
        else:
            self.reward_weights = reward_weights
        
        # Ensure data is sorted by date
        self.data = self.data.sort_values(date_col)
        self.dates = sorted(self.data[date_col].unique())
        self.n_periods = len(self.dates)
        
        # State dimension calculation
        # Features per asset * n_assets + portfolio weights + portfolio stats + time features
        self.n_features = len(self.feature_cols)
        # 7 portfolio stats: return, vol, sharpe, drawdown, entropy, turnover, vol_trend
        self.state_dim = (self.n_features * self.n_assets) + self.n_assets + 7 + 2
        
        # Action space: portfolio weights
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
        
        # Track additional state for risk metrics
        self.prev_weights = self.weights.copy()
        self.portfolio_values = [self.initial_capital]
        self.weight_history = [self.weights.copy()]
        
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
        # ALWAYS calculate returns, not just on rebalance
        current_date = self.dates[self.current_step]
        next_date = self.dates[self.current_step + 1]
        
        # Get price changes for ALL assets
        price_changes = []
        for ticker in self.tickers:
            curr_price = self.data[(self.data[self.date_col] == current_date) & 
                                   (self.data['ticker'] == ticker)]['close'].iloc[0]
            next_price = self.data[(self.data[self.date_col] == next_date) & 
                                   (self.data['ticker'] == ticker)]['close'].iloc[0]
            price_changes.append(np.log(next_price / curr_price))
        
        # Calculate portfolio return EVERY STEP
        portfolio_return = np.sum(self.weights * np.array(price_changes))
        
        # Only update weights on rebalance
        should_rebalance = (self.current_step % self.rebalance_frequency == 0)
        if should_rebalance:
            # Save previous weights before updating
            self.prev_weights = self.weights.copy()
            
            new_weights = self._normalize_action(action)
            transaction_cost = self.transaction_cost * np.sum(np.abs(new_weights - self.weights))
            portfolio_return -= transaction_cost
            self.weights = new_weights
            self.rebalance_count += 1
            
            # Track weight history
            self.weight_history.append(self.weights.copy())
        
        # ALWAYS append returns
        self.portfolio_returns.append(portfolio_return)
        self.portfolio_value *= np.exp(portfolio_return)
        
        # Track portfolio value history
        self.portfolio_values.append(self.portfolio_value)
        
        # Calculate reward
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
            'period_return': portfolio_return,
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
        # Clip to valid range
        action = np.clip(action, 0, 1)
        
        # Normalize to sum to 1
        weights = action / (np.sum(action) + 1e-10)
        
        # Apply max weight constraint
        weights = np.clip(weights, 0, self.max_weight_per_asset)
        
        # Renormalize to sum to 1
        weights = weights / (np.sum(weights) + 1e-10)
        
        return weights.astype(np.float32)
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on reward_type.
        
        FIXED: Consistent reward calculations with proper normalization.
        Uses config parameters instead of hardcoded values.
        """
        if len(self.portfolio_returns) < 1:
            return 0.0
        
        # Log return reward (simplest, often most stable)
        if self.reward_type in ['log_return', 'simple_return']:
            return float(self.portfolio_returns[-1])
        
        # Multi-component reward
        elif self.reward_type == 'multi_component':
            if len(self.portfolio_returns) < 2:
                return 0.0

            # Get weights from config
            rw = self.reward_weights.get('multi_component', {})

            # Use reward_lookback parameter
            lookback = min(self.reward_lookback, len(self.portfolio_returns))
            recent_returns = np.array(self.portfolio_returns[-lookback:])

            if len(recent_returns) < 2:
                return 0.0

            # 1. Risk-adjusted return (Sharpe)
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns, ddof=1) + 1e-8
            sharpe = mean_return / std_return * np.sqrt(self.annualization_factor)
            sharpe_reward = np.tanh(sharpe / rw.get('sharpe_norm', 3.0))

            # 2. Absolute return
            total_return = np.exp(np.sum(recent_returns)) - 1
            return_reward = np.tanh(total_return * rw.get('return_norm', 10.0))

            # 3. Drawdown penalty
            cum_returns = np.exp(np.cumsum(recent_returns))
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            drawdown_penalty = np.tanh(max_drawdown * rw.get('drawdown_norm', 10.0))

            # 4. Volatility penalty
            annualized_vol = std_return * np.sqrt(self.annualization_factor)
            vol_excess = max(0, annualized_vol - self.vol_target)
            vol_penalty = -np.tanh(vol_excess * rw.get('volatility_norm', 5.0))

            # 5. Turnover penalty
            turnover = np.sum(np.abs(self.weights - self.prev_weights))
            turnover_penalty = -rw.get('turnover_penalty_coef', 0.1) * turnover

            # Combined reward with weights from config
            reward = (
                rw.get('sharpe', 0.4) * sharpe_reward +
                rw.get('return', 0.2) * return_reward +
                rw.get('drawdown', 0.2) * drawdown_penalty +
                rw.get('volatility', 0.2) * vol_penalty +
                turnover_penalty
            )

            # Clip using config bounds
            return float(np.clip(reward, rw.get('clip_min', -2.0), rw.get('clip_max', 2.0)))
        
        # Risk-aware reward (most sophisticated)
        elif self.reward_type == 'risk_aware':
            """
            Custom 5-component reward with explicit risk penalties.
            Uses config parameters for all weights and normalization.
            """
            if len(self.portfolio_returns) < 2:
                return 0.0

            # Get weights from config
            rw = self.reward_weights.get('risk_aware', {})

            # Use configured lookback
            lookback = min(self.reward_lookback, len(self.portfolio_returns))
            recent_returns = np.array(self.portfolio_returns[-lookback:])

            if len(recent_returns) < 2:
                return 0.0

            # Component 1: Mean Return
            mean_return = np.mean(recent_returns)
            return_component = np.tanh(mean_return * rw.get('mean_return_norm', 100.0))

            # Component 2: Volatility Penalty
            volatility = np.std(recent_returns, ddof=1) * np.sqrt(self.annualization_factor)
            vol_penalty = -np.tanh((volatility - self.vol_target) / rw.get('volatility_norm', 0.15))

            # Component 3: Drawdown Penalty - EXPONENTIAL
            cum_returns = np.exp(np.cumsum(recent_returns))
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            dd_penalty = -np.tanh(np.exp(abs(max_drawdown) * rw.get('drawdown_exp', 5.0)) - 1)

            # Component 4: Diversification Bonus
            weights_safe = self.weights + 1e-8
            entropy = -np.sum(weights_safe * np.log(weights_safe))
            max_entropy = np.log(self.n_assets)
            entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.0

            # Bonus if entropy above minimum threshold
            diversification_bonus = np.tanh((entropy_normalized - self.min_entropy) * rw.get('entropy_factor', 2.0))

            # Component 5: Turnover Penalty
            turnover = np.sum(np.abs(self.weights - self.prev_weights))
            turnover_penalty = -np.tanh((turnover - self.max_turnover) * rw.get('turnover_norm', 5.0))

            # Combined reward with weights from config
            reward = (
                rw.get('mean_return', 0.30) * return_component +
                rw.get('volatility_penalty', 0.25) * vol_penalty +
                rw.get('drawdown_penalty', 0.25) * dd_penalty +
                rw.get('diversification_bonus', 0.10) * diversification_bonus +
                rw.get('turnover_penalty', 0.10) * turnover_penalty
            )

            # Clip using config bounds
            return float(np.clip(reward, rw.get('clip_min', -2.0), rw.get('clip_max', 2.0)))
        
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}. "
                           f"Use 'log_return', 'multi_component', or 'risk_aware'")
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        
        FIXED: Consistent portfolio statistics calculation.
        """
        if self.current_step >= len(self.dates):
            return np.zeros(self.state_dim, dtype=np.float32)
        
        current_date = self.dates[self.current_step]
        
        # 1. Asset features (flattened)
        asset_features = []
        for ticker in self.tickers:
            ticker_data = self.data[
                (self.data[self.date_col] == current_date) &
                (self.data['ticker'] == ticker)
            ]
            if len(ticker_data) > 0:
                features = ticker_data[self.feature_cols].iloc[0].values
                asset_features.extend(features)
            else:
                asset_features.extend([0.0] * len(self.feature_cols))
        
        # 2. Current portfolio weights
        weights_features = self.weights.tolist()
        
        # 3. Portfolio statistics (7 metrics)
        if len(self.portfolio_returns) >= 2:
            lookback = min(self.reward_lookback, len(self.portfolio_returns))
            
            # Basic metrics
            portfolio_return = np.mean(self.portfolio_returns[-lookback:])
            portfolio_vol = np.std(self.portfolio_returns[-lookback:], ddof=1)
            
            # Calculate actual Sharpe (not reward)
            sharpe = (portfolio_return / (portfolio_vol + 1e-8)) * np.sqrt(self.annualization_factor)
            
            # Current drawdown
            if len(self.portfolio_values) > 1:
                running_max = max(self.portfolio_values)
                current_drawdown = (self.portfolio_value - running_max) / running_max
            else:
                current_drawdown = 0.0
            
            # Weight entropy (diversification)
            weights_safe = self.weights + 1e-8
            weight_entropy = -np.sum(weights_safe * np.log(weights_safe))
            max_entropy = np.log(self.n_assets)
            weight_entropy = weight_entropy / max_entropy if max_entropy > 0 else 0.0
            
            # Recent turnover
            recent_turnover = np.sum(np.abs(self.weights - self.prev_weights))
            
            # Volatility trend
            if len(self.portfolio_returns) >= 24:
                vol_recent = np.std(self.portfolio_returns[-12:], ddof=1)
                vol_older = np.std(self.portfolio_returns[-24:-12], ddof=1)
                vol_trend = (vol_recent - vol_older) / (vol_older + 1e-8)
            else:
                vol_trend = 0.0
            
        else:
            # Not enough data yet
            portfolio_return = 0.0
            portfolio_vol = 0.0
            sharpe = 0.0
            current_drawdown = 0.0
            weight_entropy = 1.0
            recent_turnover = 0.0
            vol_trend = 0.0
        
        portfolio_stats = [
            portfolio_return, portfolio_vol, sharpe,
            current_drawdown, weight_entropy, recent_turnover, vol_trend
        ]
        
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
        """
        Get current portfolio statistics.
        
        FIXED: Consistent calculation with config parameters.
        """
        if len(self.portfolio_returns) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'portfolio_value': float(self.portfolio_value)
            }
        
        returns = np.array(self.portfolio_returns)
        cumulative = np.exp(np.cumsum(returns))
        
        total_return = cumulative[-1] - 1.0
        
        # Calculate ACTUAL Sharpe ratio
        if len(returns) >= 2:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1) + 1e-8
            sharpe = (mean_return / std_return) * np.sqrt(self.annualization_factor)
        else:
            sharpe = 0.0
        
        volatility = np.std(returns) * np.sqrt(self.annualization_factor)
        
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


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def create_walk_forward_folds(
    train_data: pd.DataFrame,
    n_folds: int,
    min_train_size: Optional[int] = None,
    date_col: str = 'date'
) -> List[Dict[str, pd.DataFrame]]:
    """
    Create chronological folds for walking forward validation.
    
    REFACTORED: Uses config parameters.
    """
    # Get unique dates sorted
    dates = sorted(train_data[date_col].unique())
    n_periods = len(dates)
    
    # Calculate minimum train size
    if min_train_size is None:
        min_train_size = max(52, int(n_periods * 0.2))  # At least 1 year or 20%
    
    # Calculate validation size per fold
    remaining_periods = n_periods - min_train_size
    val_size_per_fold = remaining_periods // n_folds
    
    # Ensure at least 4 weeks per validation fold
    if val_size_per_fold < 4:
        val_size_per_fold = 4
    
    # Calculate fold boundaries
    folds = []
    
    for fold_idx in range(n_folds):
        # Training: all data up to current split point
        train_end_idx = min_train_size + (fold_idx * val_size_per_fold)
        
        # Validation: next chunk of data
        val_start_idx = train_end_idx
        val_end_idx = min(val_start_idx + val_size_per_fold, n_periods)
        
        # Ensure we have enough validation data
        if val_end_idx - val_start_idx < 4:
            break
        
        # Create fold
        train_dates = dates[:train_end_idx]
        val_dates = dates[val_start_idx:val_end_idx]
        
        fold = {
            'train': train_data[train_data[date_col].isin(train_dates)].copy(),
            'val': train_data[train_data[date_col].isin(val_dates)].copy()
        }
        
        folds.append(fold)
    
    return folds


def print_fold_summary(folds: List[Dict[str, pd.DataFrame]], date_col: str = 'date'):
    """Print summary of walk-forward folds."""
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION FOLDS")
    print("="*80)
    
    for i, fold in enumerate(folds):
        train_dates = sorted(fold['train'][date_col].unique())
        val_dates = sorted(fold['val'][date_col].unique())
        
        print(f"\nFold {i}:")
        print(f"  Train: {train_dates[0].date()} to {train_dates[-1].date()} "
              f"({len(train_dates)} periods)")
        print(f"  Val:   {val_dates[0].date()} to {val_dates[-1].date()} "
              f"({len(val_dates)} periods)")
        print(f"  Train samples: {len(fold['train'])}, Val samples: {len(fold['val'])}")
    
    print("="*80)


# ============================================================================
# BASELINE STRATEGIES
# ============================================================================

class BaselineStrategy:
    """Base class for baseline strategies."""
    
    def __init__(self, name: str, tickers: List[str], max_weight_per_asset: float):
        self.name = name
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.max_weight_per_asset = max_weight_per_asset
    
    def calculate_weights(self, data, step):
        """Calculate portfolio weights. Override in subclasses."""
        raise NotImplementedError
    
    def normalize_weights(self, weights):
        """Normalize and constrain weights."""
        weights = np.abs(weights)
        weights = np.clip(weights, 0, self.max_weight_per_asset)
        weights = weights / (np.sum(weights) + 1e-10)
        return weights


class EqualWeight(BaselineStrategy):
    """Equal weight portfolio."""
    
    def __init__(self, tickers: List[str], max_weight_per_asset: float):
        super().__init__("Equal Weight", tickers, max_weight_per_asset)
    
    def calculate_weights(self, data, step):
        return np.ones(self.n_assets) / self.n_assets


class BuyHold(BaselineStrategy):
    """Buy and hold equal weight portfolio."""
    
    def __init__(self, tickers: List[str], max_weight_per_asset: float):
        super().__init__("Buy & Hold", tickers, max_weight_per_asset)
    
    def calculate_weights(self, data, step):
        # Initial equal weights, then let drift
        return np.ones(self.n_assets) / self.n_assets


class RiskParity(BaselineStrategy):
    """Risk parity portfolio (equal risk contribution)."""
    
    def __init__(self, tickers: List[str], max_weight_per_asset: float,
                 date_col: str, lookback: int = 12):
        super().__init__("Risk Parity", tickers, max_weight_per_asset)
        self.date_col = date_col
        self.lookback = lookback
    
    def calculate_weights(self, data, step):
        if step < self.lookback:
            return np.ones(self.n_assets) / self.n_assets
        
        dates = sorted(data[self.date_col].unique())
        lookback_dates = dates[max(0, step-self.lookback):step]
        
        volatilities = []
        for ticker in self.tickers:
            ticker_data = data[(data['ticker'] == ticker) & 
                              data[self.date_col].isin(lookback_dates)]
            if len(ticker_data) > 1:
                returns = ticker_data['close'].pct_change().dropna()
                vol = returns.std()
                volatilities.append(vol if vol > 0 else 0.01)
            else:
                volatilities.append(0.01)
        
        inv_vols = 1.0 / np.array(volatilities)
        weights = inv_vols / inv_vols.sum()
        
        return self.normalize_weights(weights)


class Momentum(BaselineStrategy):
    """Momentum portfolio (overweight recent winners)."""
    
    def __init__(self, tickers: List[str], max_weight_per_asset: float,
                 date_col: str, lookback: int = 12):
        super().__init__("Momentum", tickers, max_weight_per_asset)
        self.date_col = date_col
        self.lookback = lookback
    
    def calculate_weights(self, data, step):
        if step < self.lookback:
            return np.ones(self.n_assets) / self.n_assets
        
        dates = sorted(data[self.date_col].unique())
        lookback_dates = dates[max(0, step-self.lookback):step]
        
        returns = []
        for ticker in self.tickers:
            ticker_data = data[(data['ticker'] == ticker) & 
                              data[self.date_col].isin(lookback_dates)]
            if len(ticker_data) >= 2:
                period_return = (ticker_data['close'].iloc[-1] / 
                               ticker_data['close'].iloc[0]) - 1
                returns.append(period_return)
            else:
                returns.append(0.0)
        
        returns = np.array(returns)
        returns = returns - returns.min() + 0.01
        
        return self.normalize_weights(returns)


class MinVariance(BaselineStrategy):
    """Minimum variance portfolio optimization."""
    
    def __init__(self, tickers: List[str], max_weight_per_asset: float,
                 date_col: str, lookback: int = 12):
        super().__init__("Min Variance", tickers, max_weight_per_asset)
        self.date_col = date_col
        self.lookback = lookback
    
    def calculate_weights(self, data, step):
        if step < self.lookback:
            return np.ones(self.n_assets) / self.n_assets
        
        dates = sorted(data[self.date_col].unique())
        lookback_dates = dates[max(0, step-self.lookback):step]
        
        returns_matrix = []
        for ticker in self.tickers:
            ticker_data = data[(data['ticker'] == ticker) & 
                              data[self.date_col].isin(lookback_dates)]
            if len(ticker_data) > 1:
                rets = ticker_data['close'].pct_change().dropna().values
                returns_matrix.append(rets)
            else:
                returns_matrix.append(np.zeros(len(lookback_dates)-1))
        
        returns_matrix = np.array(returns_matrix).T
        cov_matrix = np.cov(returns_matrix.T) + np.eye(self.n_assets) * 1e-5
        
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(self.n_assets)
            weights = inv_cov @ ones / (ones @ inv_cov @ ones)
            weights = np.abs(weights)
        except:
            weights = np.ones(self.n_assets) / self.n_assets
        
        return self.normalize_weights(weights)


# ============================================================================
# BACKTEST FUNCTIONS
# ============================================================================

def backtest_rl_agent(model, data: pd.DataFrame,
                      feature_cols: List[str],
                      tickers: List[str],
                      rebalance_frequency: int,
                      transaction_cost: float,
                      max_weight_per_asset: float,
                      reward_lookback: int,
                      initial_capital: float,
                      reward_type: str,
                      seed: int = 42,
                      **env_kwargs) -> Dict:
    """
    Backtest RL agent on test data.
    
    REFACTORED: Accepts additional environment kwargs for config parameters.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        env = PortfolioEnv(
            data=data.copy(),
            feature_cols=feature_cols,
            tickers=tickers,
            rebalance_frequency=rebalance_frequency,
            transaction_cost=transaction_cost,
            max_weight_per_asset=max_weight_per_asset,
            reward_lookback=reward_lookback,
            initial_capital=initial_capital,
            reward_type=reward_type,
            seed=seed,
            **env_kwargs  # Pass additional config parameters
        )
        return PortfolioMonitor(env)
    
    env = DummyVecEnv([make_env])
    
    # Run backtest
    obs = env.reset()
    done = False
    
    trajectory = {'weights': [], 'values': [], 'returns': [], 'dates': []}
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Get env state
        env_unwrapped = env.envs[0]
        while hasattr(env_unwrapped, 'env'):
            env_unwrapped = env_unwrapped.env
        
        trajectory['weights'].append(env_unwrapped.weights.copy())
        trajectory['values'].append(env_unwrapped.portfolio_value)
        
        if len(env_unwrapped.portfolio_returns) > 0:
            trajectory['returns'].append(env_unwrapped.portfolio_returns[-1])
        
        if env_unwrapped.current_step < len(env_unwrapped.dates):
            trajectory['dates'].append(env_unwrapped.dates[env_unwrapped.current_step])
        
        if done[0]:
            stats = info[0].get('portfolio_stats', env_unwrapped.get_portfolio_stats())
    
    return {'trajectory': trajectory, 'stats': stats, 'name': 'RL Agent'}


def backtest_baseline(strategy, data: pd.DataFrame,
                     date_col: str,
                     tickers: List[str],
                     rebalance_frequency: int,
                     transaction_cost: float,
                     initial_capital: float,
                     annualization_factor: int = 52) -> Dict:
    """
    Backtest baseline strategy on test data.
    
    REFACTORED: Uses config parameter for annualization.
    """
    dates = sorted(data[date_col].unique())
    n_periods = len(dates)
    
    portfolio_value = initial_capital
    weights = strategy.calculate_weights(data, 0)
    
    trajectory = {
        'weights': [weights.copy()],
        'values': [portfolio_value],
        'returns': [],
        'dates': [dates[0]]
    }
    
    for step in range(n_periods - 1):
        current_date = dates[step]
        next_date = dates[step + 1]
        
        # Calculate asset returns (log returns)
        asset_returns = []
        for ticker in tickers:
            curr = data[(data[date_col] == current_date) & 
                       (data['ticker'] == ticker)]['close'].iloc[0]
            next = data[(data[date_col] == next_date) & 
                       (data['ticker'] == ticker)]['close'].iloc[0]
            asset_returns.append(np.log(next / curr))
        
        asset_returns = np.array(asset_returns)
        portfolio_return = np.sum(weights * asset_returns)
        
        # Rebalance?
        should_rebalance = (step % rebalance_frequency == 0) and (strategy.name != "Buy & Hold")
        
        if should_rebalance:
            new_weights = strategy.calculate_weights(data, step + 1)
            transaction_cost_amount = transaction_cost * np.sum(np.abs(new_weights - weights))
            portfolio_return -= transaction_cost_amount
            weights = new_weights
        elif strategy.name == "Buy & Hold" and step > 0:
            # Update weights by returns (drift)
            weights = weights * np.exp(asset_returns)
            weights = weights / weights.sum()
        
        portfolio_value *= np.exp(portfolio_return)
        
        trajectory['returns'].append(portfolio_return)
        trajectory['weights'].append(weights.copy())
        trajectory['values'].append(portfolio_value)
        trajectory['dates'].append(next_date)
    
    # Calculate stats with config parameter
    returns = np.array(trajectory['returns'])
    total_return = (portfolio_value / initial_capital) - 1.0
    
    if len(returns) >= 2:
        sharpe = (np.mean(returns) / (np.std(returns, ddof=1) + 1e-8)) * np.sqrt(annualization_factor)
    else:
        sharpe = 0.0
    
    volatility = np.std(returns) * np.sqrt(annualization_factor)
    
    cum_returns = np.exp(np.cumsum(returns))
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    stats = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'portfolio_value': portfolio_value
    }
    
    return {'trajectory': trajectory, 'stats': stats, 'name': strategy.name}
