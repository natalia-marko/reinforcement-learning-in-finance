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
    # Data Loading
    'load_data',
    # Callbacks & Monitoring
    'PortfolioMonitor',
    'TrainingLogger',
    'CosineAnnealingLRCallback',
    # Environment
    'PortfolioEnv',
    'PortfolioEnv2',  # Holdings-based implementation
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

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load training data from parquet file.

    Parameters:
    -----------
    data_path : Path, optional
        Path to data file. If None, uses DATA_PATH from config.

    Returns:
    --------
    pd.DataFrame
        Training data with date column converted to datetime.

    Raises:
    -------
    FileNotFoundError
        If data file doesn't exist.
    """
    # Import config here to avoid circular imports
    from config import DATA_PATH, DATE_COL

    if data_path is None:
        data_path = DATA_PATH

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_parquet(data_path)

    # Ensure date column is datetime
    if DATE_COL in data.columns:
        data[DATE_COL] = pd.to_datetime(data[DATE_COL])

    return data


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
        # Store final stats before reset (if available and meaningful)
        if hasattr(self.env, 'get_portfolio_stats'):
            stats = self.env.get_portfolio_stats()
            # Only store if stats are meaningful (not all zeros from empty episode)
            if stats.get('portfolio_value', 0) > 0 or len(self.current_episode_rewards) > 0:
                # Avoid duplicate if already stored in step() when episode ended
                if len(self.episode_portfolio_stats) == len(self.episode_returns):
                    self.episode_portfolio_stats.append(stats)

        # Reset tracking
        self.current_episode_rewards = []

        return self.env.reset(**kwargs)

    def step(self, action):
        """Execute step and track rewards/stats."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_episode_rewards.append(reward)

        # If episode done, store stats in info and track episode metrics
        if terminated or truncated:
            stats = self.env.get_portfolio_stats()
            info['portfolio_stats'] = stats
            
            # Store episode metrics
            episode_return = sum(self.current_episode_rewards)
            episode_length = len(self.current_episode_rewards)
            
            self.episode_returns.append(episode_return)
            self.episode_lengths.append(episode_length)
            
            # Store stats for this episode (synchronized with returns/lengths)
            self.episode_portfolio_stats.append(stats)

        return obs, reward, terminated, truncated, info

    def get_episode_stats(self, episode_idx: int = -1) -> Dict:
        """
        Get portfolio stats for a specific episode.
        
        Parameters:
        -----------
        episode_idx : int
            Episode index (default: -1 for most recent)
            
        Returns:
        --------
        Dict
            Portfolio statistics dictionary
        """
        if len(self.episode_portfolio_stats) == 0:
            return {}
        return self.episode_portfolio_stats[episode_idx]
    
    def get_all_episode_stats(self) -> List[Dict]:
        """
        Get all stored episode portfolio statistics.
        
        Returns:
        --------
        List[Dict]
            List of portfolio statistics dictionaries
        """
        return self.episode_portfolio_stats.copy()
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics across all episodes.
        
        Returns:
        --------
        Dict
            Summary statistics including mean/std of returns, Sharpe ratios, etc.
        """
        if len(self.episode_portfolio_stats) == 0:
            return {}
        
        sharpe_ratios = [s.get('sharpe_ratio', 0.0) for s in self.episode_portfolio_stats]
        total_returns = [s.get('total_return', 0.0) for s in self.episode_portfolio_stats]
        volatilities = [s.get('volatility', 0.0) for s in self.episode_portfolio_stats]
        max_drawdowns = [s.get('max_drawdown', 0.0) for s in self.episode_portfolio_stats]
        
        return {
            'num_episodes': len(self.episode_portfolio_stats),
            'mean_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0.0,
            'std_sharpe': np.std(sharpe_ratios) if sharpe_ratios else 0.0,
            'mean_return': np.mean(total_returns) if total_returns else 0.0,
            'std_return': np.std(total_returns) if total_returns else 0.0,
            'mean_volatility': np.mean(volatilities) if volatilities else 0.0,
            'mean_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0.0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'mean_episode_return': np.mean(self.episode_returns) if self.episode_returns else 0.0
        }


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
                          f"Sharpe={stats.get('sharpe_ratio', 0.0):.3f}, "
                          f"Return={stats.get('total_return', 0.0)*100:.2f}%")

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
        stats = None

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)

            # SB3 VecEnv returns 4 values (dones = terminated | truncated)
            obs, reward, dones, info = self.eval_env.step(action)

            # VecEnv returns arrays, we access the first element [0]
            episode_reward += reward[0]
            episode_length += 1

            # VecEnv 'dones' combines terminated and truncated
            if dones[0]:
                done = True
                # Get stats from info (PortfolioMonitor stores it)
                if len(info) > 0 and 'portfolio_stats' in info[0]:
                    stats = info[0]['portfolio_stats']
                else:
                    # Fallback: try to get from environment
                    val_env_unwrapped = self.eval_env.envs[0]
                    while hasattr(val_env_unwrapped, 'env'):
                        val_env_unwrapped = val_env_unwrapped.env
                    stats = val_env_unwrapped.get_portfolio_stats()

        # Ensure stats is defined (fallback if loop exited unexpectedly)
        if stats is None:
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
                  f"Return={stats.get('total_return', 0.0)*100:.2f}%, "
                  f"Drawdown={stats.get('max_drawdown', 0.0)*100:.2f}%")

        # Check if this is the best model
        if val_sharpe > self.best_val_sharpe:
            self.best_val_sharpe = val_sharpe
            self.no_improvement_count = 0

            # Save best model
            model_path = self.fold_dir / 'best_model.zip'
            self.model.save(str(model_path))
            print(f"  -> New best model saved (Sharpe: {val_sharpe:.3f})")
        else:
            self.no_improvement_count += 1
            if self.verbose > 0:
                print(f"  -> No improvement for {self.no_improvement_count}/{self.patience} evaluations")

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

    def get_best_sharpe(self) -> float:
        """Return the best validation Sharpe ratio achieved."""
        return self.best_val_sharpe


class CosineAnnealingLRCallback(BaseCallback):
    """
    Cosine annealing learning rate schedule callback.

    LR decreases from initial value following cosine curve:
    lr(t) = lr_init * 0.5 * (1 + cos(π * progress))

    This provides smooth decay with warm restarts potential.
    """

    def __init__(self, lr_schedule_fn, verbose: int = 0):
        """
        Initialize callback.

        Parameters:
        -----------
        lr_schedule_fn : callable
            Function that takes progress (0 to 1) and returns learning rate.
            Example: lambda p: 0.0003 * 0.5 * (1 + np.cos(np.pi * p))
        verbose : int
            Verbosity level
        """
        super().__init__(verbose)
        self.lr_schedule_fn = lr_schedule_fn
        self.total_timesteps = None

    def _on_training_start(self) -> None:
        """Get total timesteps at training start."""
        self.total_timesteps = self.locals.get('total_timesteps', None)

    def _on_step(self) -> bool:
        """Update learning rate based on training progress."""
        if self.total_timesteps is None:
            self.total_timesteps = self.locals.get('total_timesteps', None)

        if self.total_timesteps is None or self.total_timesteps == 0:
            return True

        # Calculate progress (0 to 1)
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)

        # Get new learning rate from schedule
        new_lr = self.lr_schedule_fn(progress)

        # Update optimizer learning rate
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = new_lr

        return True


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
        reward_type: str = 'log_return',
        date_col: str = 'date',
        seed: int = 42,
        # Config parameters for reward functions
        min_entropy: float = 0.5,  # Minimum portfolio entropy
        max_turnover: float = 0.20,  # Maximum acceptable turnover
        annualization_factor: int = 52,  # For weekly data
        risk_free_rate: float = 0.04  # Annual risk-free rate (4%)
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
            Type of reward function ('log_return', 'multi_component')
        date_col : str
            Name of date column
        seed : int
            Random seed for reproducibility
        min_entropy : float
            Minimum acceptable portfolio entropy (for multi_component reward)
        max_turnover : float
            Maximum acceptable turnover rate (for multi_component reward)
        annualization_factor : int
            Factor for annualizing metrics (52 for weekly, 252 for daily)
        risk_free_rate : float
            Annual risk-free rate for Sharpe calculation (default 0.04)
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

        # Config parameters for rewards and metrics
        self.min_entropy = min_entropy
        self.max_turnover = max_turnover
        self.annualization_factor = annualization_factor
        self.risk_free_rate = risk_free_rate
        # Convert annual risk-free rate to per-period rate
        self.rf_per_period = risk_free_rate / annualization_factor

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
        # Check if we're already at terminal state (prevents index out of bounds)
        if self.current_step >= self.n_periods - 1:
            observation = self._get_observation()
            info = {
                'portfolio_value': self.portfolio_value,
                'weights': self.weights.copy(),
                'period_return': 0.0,
                'cumulative_return': (self.portfolio_value / self.initial_capital) - 1.0,
                'step': self.current_step,
                'rebalance_count': self.rebalance_count,
                'terminal': True
            }
            return observation, 0.0, True, False, info

        # ALWAYS calculate returns, not just on rebalance
        current_date = self.dates[self.current_step]
        next_date = self.dates[self.current_step + 1]
        
        # Get price changes for ALL assets (log returns)
        price_changes = []
        for ticker in self.tickers:
            curr_price = self.data[(self.data[self.date_col] == current_date) &
                                   (self.data['ticker'] == ticker)]['close'].iloc[0]
            next_price = self.data[(self.data[self.date_col] == next_date) &
                                   (self.data['ticker'] == ticker)]['close'].iloc[0]
            price_changes.append(np.log(next_price / curr_price))

        price_changes = np.array(price_changes)

        # Calculate asset multipliers (price ratios)
        asset_multipliers = np.exp(price_changes)

        # Calculate portfolio return with current weights
        # Correct formula: log(sum(w_i * exp(log_r_i))) not sum(w_i * log_r_i)
        portfolio_return = np.log(np.sum(self.weights * asset_multipliers))

        # Calculate drifted weights after price change (realistic weight evolution)
        # In reality, if Asset A goes up 10% and B stays flat, A's weight increases
        drifted_weights = self.weights * asset_multipliers
        sum_weights = np.sum(drifted_weights)
        if sum_weights > 1e-8:
            drifted_weights = drifted_weights / sum_weights
        else:
            # Fallback to equal weights if numerical issues
            drifted_weights = np.ones(self.n_assets) / self.n_assets

        # Check if we should rebalance this step
        should_rebalance = (self.current_step % self.rebalance_frequency == 0)

        if should_rebalance:
            # Save previous weights for turnover calculation
            self.prev_weights = self.weights.copy()

            # Get target weights from action
            target_weights = self._normalize_action(action)

            # Transaction cost: from DRIFTED weights to target (not old to target)
            transaction_cost = self.transaction_cost * np.sum(np.abs(target_weights - drifted_weights))
            portfolio_return -= transaction_cost

            # Set new weights to target
            self.weights = target_weights
            self.rebalance_count += 1

            # Track weight history
            self.weight_history.append(self.weights.copy())
        else:
            # No rebalance: weights drift naturally with price changes
            self.weights = drifted_weights
        
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
        - Each weight >= min_weight (prevents degenerate zero-weight policy)
        """
        MIN_WEIGHT = 1e-3  # Minimum 0.1% per asset (prevents collapse)

        # Clip to valid range
        action = np.clip(action, 0, 1)

        # Normalize to sum to 1
        weights = action / (np.sum(action) + 1e-10)

        # Apply max weight constraint
        weights = np.clip(weights, MIN_WEIGHT, self.max_weight_per_asset)

        # Renormalize to sum to 1
        weights = weights / (np.sum(weights) + 1e-10)

        return weights.astype(np.float32)
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on reward_type.

        Reward types:
        - 'log_return': Simple log return (default)
        - 'multi_component': Return + entropy bonus - turnover penalty

        All rewards are clipped to [-1, 1] for PPO stability.
        """
        if len(self.portfolio_returns) < 1:
            return 0.0

        base_return = self.portfolio_returns[-1]

        if self.reward_type == 'log_return':
            reward = base_return

        elif self.reward_type == 'multi_component':
            # Calculate weight entropy (diversification measure)
            weights_safe = self.weights + 1e-8
            weight_entropy = -np.sum(weights_safe * np.log(weights_safe))
            max_entropy = np.log(self.n_assets)
            normalized_entropy = weight_entropy / max_entropy if max_entropy > 0 else 0.0

            # Entropy bonus: reward diversification above minimum
            entropy_bonus = max(0, normalized_entropy - self.min_entropy) * 0.1

            # Turnover penalty: penalize excessive trading
            turnover = np.sum(np.abs(self.weights - self.prev_weights))
            turnover_penalty = max(0, turnover - self.max_turnover) * 0.2

            reward = base_return + entropy_bonus - turnover_penalty

        else:
            # Fallback to log_return
            reward = base_return

        # Clip reward to [-1, 1] for PPO stability
        reward = np.clip(reward, -1.0, 1.0)

        # Ensure reward is valid (no NaN/Inf)
        if not np.isfinite(reward):
            reward = 0.0

        return float(reward)
    
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

            # Calculate actual Sharpe with risk-free rate
            excess_return = portfolio_return - self.rf_per_period
            sharpe = (excess_return / (portfolio_vol + 1e-8)) * np.sqrt(self.annualization_factor)
            
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
        
        # Scale portfolio stats to consistent ranges for stable training
        # (features are pre-normalized, but these dynamic stats need scaling)
        portfolio_stats = [
            np.clip(portfolio_return * 10, -1, 1),      # Scale returns, clip to [-1, 1]
            np.clip(portfolio_vol * 10, 0, 1),          # Scale vol, clip to [0, 1]
            np.clip(sharpe / 3, -1, 1),                 # Normalize Sharpe, clip to [-1, 1]
            np.clip(current_drawdown * 2, -1, 0),       # Scale drawdown, clip to [-1, 0]
            weight_entropy,                              # Already in [0, 1]
            np.clip(recent_turnover, 0, 1),             # Clip to [0, 1]
            np.clip(vol_trend, -1, 1)                   # Clip to [-1, 1]
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

        # Ensure no NaN/Inf values (safety check for data issues)
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

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
        
        # Calculate ACTUAL Sharpe ratio with risk-free rate
        if len(returns) >= 2:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1) + 1e-8
            excess_return = mean_return - self.rf_per_period
            sharpe = (excess_return / std_return) * np.sqrt(self.annualization_factor)
        else:
            sharpe = 0.0
        
        volatility = np.std(returns, ddof=1) * np.sqrt(self.annualization_factor)
        
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


class PortfolioEnv2(gym.Env):
    """
    Portfolio rebalancing environment with HOLDINGS-BASED tracking.

    This is the theoretically correct implementation that:
    - Tracks actual share holdings (not just weights)
    - Calculates returns as log(value_after / value_before)
    - Applies transaction costs to portfolio value directly
    - Weights emerge naturally from holdings * prices

    Use this instead of PortfolioEnv for more accurate portfolio simulation.
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
        reward_type: str = 'log_return',
        date_col: str = 'date',
        seed: int = 42,
        min_entropy: float = 0.5,
        max_turnover: float = 0.20,
        annualization_factor: int = 52,
        risk_free_rate: float = 0.04
    ):
        """
        Initialize PortfolioEnv2 (holdings-based).

        Parameters are identical to PortfolioEnv for drop-in replacement.
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

        # Config parameters for rewards and metrics
        self.min_entropy = min_entropy
        self.max_turnover = max_turnover
        self.annualization_factor = annualization_factor
        self.risk_free_rate = risk_free_rate
        self.rf_per_period = risk_free_rate / annualization_factor

        # Ensure data is sorted by date
        self.data = self.data.sort_values(date_col)
        self.dates = sorted(self.data[date_col].unique())
        self.n_periods = len(self.dates)

        # State dimension calculation
        self.n_features = len(self.feature_cols)
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

    def _get_prices_at_step(self, step: int) -> np.ndarray:
        """Get asset prices at a specific step."""
        if step >= len(self.dates):
            step = len(self.dates) - 1

        date = self.dates[step]
        prices = []
        for ticker in self.tickers:
            price = self.data[(self.data[self.date_col] == date) &
                             (self.data['ticker'] == ticker)]['close'].iloc[0]
            prices.append(price)
        return np.array(prices)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state with holdings tracking."""
        super().reset(seed=seed)

        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.portfolio_returns = []
        self.rebalance_count = 0

        # Initialize equal-weight holdings
        init_prices = self._get_prices_at_step(0)
        init_weights = np.ones(self.n_assets) / self.n_assets
        holdings_value = self.portfolio_value * init_weights
        self.holdings = holdings_value / init_prices  # Number of shares

        self.weights = init_weights
        self.prev_weights = self.weights.copy()
        self.portfolio_values = [self.initial_capital]
        self.weight_history = [self.weights.copy()]

        observation = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
            'step': self.current_step
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step with holdings-based tracking.

        This is the theoretically correct implementation:
        - Portfolio return = log(value_after / value_before)
        - Transaction costs deducted from portfolio value
        - Weights emerge from holdings * prices
        """
        # Check terminal state
        if self.current_step >= self.n_periods - 1:
            observation = self._get_observation()
            info = {
                'portfolio_value': self.portfolio_value,
                'weights': self.weights.copy(),
                'period_return': 0.0,
                'cumulative_return': (self.portfolio_value / self.initial_capital) - 1.0,
                'step': self.current_step,
                'rebalance_count': self.rebalance_count,
                'terminal': True
            }
            return observation, 0.0, True, False, info

        # Get prices
        current_prices = self._get_prices_at_step(self.current_step)
        next_prices = self._get_prices_at_step(self.current_step + 1)

        # Portfolio value before and after price change
        value_before = np.sum(self.holdings * current_prices)
        value_after = np.sum(self.holdings * next_prices)

        # Ensure no division by zero
        if value_before < 1e-8:
            value_before = 1e-8
        if value_after < 1e-8:
            value_after = 1e-8

        # Calculate actual portfolio return (log return)
        portfolio_return = np.log(value_after / value_before)

        # Calculate drifted weights after price change
        drifted_weights = (self.holdings * next_prices) / value_after

        # Check if we should rebalance
        should_rebalance = (self.current_step % self.rebalance_frequency == 0)

        if should_rebalance:
            # Save previous weights
            self.prev_weights = self.weights.copy()

            # Get target weights from action
            target_weights = self._normalize_action(action)

            # Transaction cost based on weight change from drifted to target
            turnover = np.sum(np.abs(target_weights - drifted_weights))
            cost_fraction = self.transaction_cost * turnover

            # Deduct cost from portfolio value
            value_after_costs = value_after * (1 - cost_fraction)

            # Update holdings to achieve target weights
            self.holdings = (value_after_costs * target_weights) / next_prices
            self.weights = target_weights
            self.portfolio_value = value_after_costs

            # Recalculate return including costs
            portfolio_return = np.log(value_after_costs / value_before)

            self.rebalance_count += 1
            self.weight_history.append(self.weights.copy())
        else:
            # No rebalance: weights drift naturally
            self.weights = drifted_weights
            self.portfolio_value = value_after

        # Track returns and values
        self.portfolio_returns.append(portfolio_return)
        self.portfolio_values.append(self.portfolio_value)

        # Calculate reward
        reward = self._calculate_reward()

        # Move to next step
        self.current_step += 1

        # Check if done
        terminated = self.current_step >= self.n_periods - 1
        truncated = False

        # Get observation
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
        """Normalize action to valid portfolio weights (prevents zero-weight collapse)."""
        MIN_WEIGHT = 1e-3  # Minimum 0.1% per asset
        action = np.clip(action, 0, 1)
        weights = action / (np.sum(action) + 1e-10)
        weights = np.clip(weights, MIN_WEIGHT, self.max_weight_per_asset)
        weights = weights / (np.sum(weights) + 1e-10)
        return weights.astype(np.float32)

    def _calculate_reward(self) -> float:
        """Calculate reward based on reward_type."""
        if len(self.portfolio_returns) < 1:
            return 0.0

        base_return = self.portfolio_returns[-1]

        if self.reward_type == 'log_return':
            reward = base_return

        elif self.reward_type == 'multi_component':
            weights_safe = self.weights + 1e-8
            weight_entropy = -np.sum(weights_safe * np.log(weights_safe))
            max_entropy = np.log(self.n_assets)
            normalized_entropy = weight_entropy / max_entropy if max_entropy > 0 else 0.0

            entropy_bonus = max(0, normalized_entropy - self.min_entropy) * 0.1
            turnover = np.sum(np.abs(self.weights - self.prev_weights))
            turnover_penalty = max(0, turnover - self.max_turnover) * 0.2

            reward = base_return + entropy_bonus - turnover_penalty
        else:
            reward = base_return

        reward = np.clip(reward, -1.0, 1.0)

        if not np.isfinite(reward):
            reward = 0.0

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        if self.current_step >= len(self.dates):
            return np.zeros(self.state_dim, dtype=np.float32)

        current_date = self.dates[self.current_step]

        # 1. Asset features
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

        # 2. Current weights
        weights_features = self.weights.tolist()

        # 3. Portfolio statistics
        if len(self.portfolio_returns) >= 2:
            lookback = min(self.reward_lookback, len(self.portfolio_returns))

            portfolio_return = np.mean(self.portfolio_returns[-lookback:])
            portfolio_vol = np.std(self.portfolio_returns[-lookback:], ddof=1)

            excess_return = portfolio_return - self.rf_per_period
            sharpe = (excess_return / (portfolio_vol + 1e-8)) * np.sqrt(self.annualization_factor)

            if len(self.portfolio_values) > 1:
                running_max = max(self.portfolio_values)
                current_drawdown = (self.portfolio_value - running_max) / running_max
            else:
                current_drawdown = 0.0

            weights_safe = self.weights + 1e-8
            weight_entropy = -np.sum(weights_safe * np.log(weights_safe))
            max_entropy = np.log(self.n_assets)
            weight_entropy = weight_entropy / max_entropy if max_entropy > 0 else 0.0

            recent_turnover = np.sum(np.abs(self.weights - self.prev_weights))

            if len(self.portfolio_returns) >= 24:
                vol_recent = np.std(self.portfolio_returns[-12:], ddof=1)
                vol_older = np.std(self.portfolio_returns[-24:-12], ddof=1)
                vol_trend = (vol_recent - vol_older) / (vol_older + 1e-8)
            else:
                vol_trend = 0.0
        else:
            portfolio_return = 0.0
            portfolio_vol = 0.0
            sharpe = 0.0
            current_drawdown = 0.0
            weight_entropy = 1.0
            recent_turnover = 0.0
            vol_trend = 0.0

        portfolio_stats = [
            np.clip(portfolio_return * 10, -1, 1),
            np.clip(portfolio_vol * 10, 0, 1),
            np.clip(sharpe / 3, -1, 1),
            np.clip(current_drawdown * 2, -1, 0),
            weight_entropy,
            np.clip(recent_turnover, 0, 1),
            np.clip(vol_trend, -1, 1)
        ]

        # 4. Time features
        month = current_date.month
        time_features = [
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12)
        ]

        observation = np.array(
            asset_features + weights_features + portfolio_stats + time_features,
            dtype=np.float32
        )

        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

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
        cumulative = np.exp(np.cumsum(returns))

        total_return = cumulative[-1] - 1.0

        if len(returns) >= 2:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1) + 1e-8
            excess_return = mean_return - self.rf_per_period
            sharpe = (excess_return / std_return) * np.sqrt(self.annualization_factor)
        else:
            sharpe = 0.0

        volatility = np.std(returns, ddof=1) * np.sqrt(self.annualization_factor)

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
    embargo_size: int = 26,
    date_col: str = 'date'
) -> List[Dict[str, pd.DataFrame]]:
    """
    Create chronological folds for walk-forward validation with embargo gap.

    CRITICAL: Embargo gap prevents data leakage from:
    - Feature lookbacks (up to 26 weeks)
    - Reward lookbacks (12 weeks)
    - Market microstructure effects

    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data with date column
    n_folds : int
        Number of validation folds
    min_train_size : int, optional
        Minimum training periods. Default: max(52, 20% of data)
    embargo_size : int
        Gap between train end and val start (default 26 weeks)
    date_col : str
        Name of date column

    Returns:
    --------
    List of dicts with keys:
        - 'train': Training DataFrame
        - 'val': Validation DataFrame
        - 'fold_idx': Fold index
        - 'train_periods': Number of training periods
        - 'val_periods': Number of validation periods
        - 'embargo_periods': Embargo gap size
    """
    # Input validation
    if date_col not in train_data.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame")
    if n_folds <= 0:
        raise ValueError(f"n_folds must be > 0, got {n_folds}")
    if embargo_size < 0:
        raise ValueError(f"embargo_size must be >= 0, got {embargo_size}")

    # Get unique dates sorted
    dates = sorted(train_data[date_col].unique())
    n_periods = len(dates)

    # Calculate minimum train size
    if min_train_size is None:
        min_train_size = max(52, int(n_periods * 0.2))  # At least 1 year or 20%

    # Validate min_train_size
    if min_train_size >= n_periods:
        raise ValueError(f"min_train_size ({min_train_size}) >= n_periods ({n_periods})")

    # Calculate validation size per fold (accounting for embargo)
    remaining_periods = n_periods - min_train_size - (n_folds * embargo_size)
    val_size_per_fold = remaining_periods // n_folds

    # Ensure at least 4 weeks per validation fold
    if val_size_per_fold < 4:
        val_size_per_fold = 4
        print(f"Warning: Small validation size ({val_size_per_fold} periods). "
              f"Consider reducing n_folds or embargo_size.")

    # Calculate fold boundaries
    folds = []

    for fold_idx in range(n_folds):
        # Training: all data up to current split point
        train_end_idx = min_train_size + (fold_idx * val_size_per_fold) + (fold_idx * embargo_size)

        # Embargo gap: skip embargo_size periods after training
        embargo_end_idx = train_end_idx + embargo_size

        # Validation: starts after embargo
        val_start_idx = embargo_end_idx
        val_end_idx = min(val_start_idx + val_size_per_fold, n_periods)

        # Ensure we have enough validation data
        if val_start_idx >= n_periods or val_end_idx - val_start_idx < 4:
            break

        # Create fold
        train_dates = dates[:train_end_idx]
        val_dates = dates[val_start_idx:val_end_idx]

        fold = {
            'train': train_data[train_data[date_col].isin(train_dates)].copy(),
            'val': train_data[train_data[date_col].isin(val_dates)].copy(),
            'fold_idx': fold_idx,
            'train_periods': len(train_dates),
            'val_periods': len(val_dates),
            'embargo_periods': embargo_size,
            'train_end': train_dates[-1] if train_dates else None,
            'val_start': val_dates[0] if val_dates else None,
            'val_end': val_dates[-1] if val_dates else None
        }

        folds.append(fold)

    if len(folds) == 0:
        raise ValueError(f"Could not create any folds. Check parameters: "
                        f"n_periods={n_periods}, min_train_size={min_train_size}, "
                        f"embargo_size={embargo_size}, n_folds={n_folds}")

    return folds


def print_fold_summary(folds: List[Dict[str, pd.DataFrame]], date_col: str = 'date'):
    """
    Print summary of walk-forward validation folds.

    Shows for each fold:
    - Training period dates and size
    - Embargo gap size
    - Validation period dates and size
    - Sample counts
    """
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION FOLDS")
    print("="*80)

    for i, fold in enumerate(folds):
        # Use metadata if available, otherwise use enumerate index
        fold_idx = fold.get('fold_idx', i)
        embargo = fold.get('embargo_periods', 0)

        train_dates = sorted(fold['train'][date_col].unique())
        val_dates = sorted(fold['val'][date_col].unique())

        print(f"\nFold {fold_idx}:")
        print(f"  Train: {train_dates[0].date()} to {train_dates[-1].date()} "
              f"({len(train_dates)} periods)")
        print(f"  Embargo gap: {embargo} periods")
        print(f"  Val:   {val_dates[0].date()} to {val_dates[-1].date()} "
              f"({len(val_dates)} periods)")
        print(f"  Samples: Train={len(fold['train'])}, Val={len(fold['val'])}")

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
    stats = None

    trajectory = {'weights': [], 'values': [], 'returns': [], 'dates': []}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # SB3 VecEnv returns 4 values (dones = terminated | truncated)
        obs, reward, dones, info = env.step(action)

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

        if dones[0]:
            done = True
            stats = info[0].get('portfolio_stats', env_unwrapped.get_portfolio_stats())

    # Ensure stats is defined
    if stats is None:
        env_unwrapped = env.envs[0]
        while hasattr(env_unwrapped, 'env'):
            env_unwrapped = env_unwrapped.env
        stats = env_unwrapped.get_portfolio_stats()

    return {'trajectory': trajectory, 'stats': stats, 'name': 'RL Agent'}


def backtest_baseline(strategy, data: pd.DataFrame,
                     date_col: str,
                     tickers: List[str],
                     rebalance_frequency: int,
                     transaction_cost: float,
                     initial_capital: float,
                     annualization_factor: int = 52,
                     risk_free_rate: float = 0.04,
                     start_date=None,
                     end_date=None) -> Dict:
    """
    Backtest baseline strategy on test data.

    Parameters:
    -----------
    start_date : datetime-like, optional
        Start date for backtest period (inclusive). If None, uses first date.
    end_date : datetime-like, optional
        End date for backtest period (inclusive). If None, uses last date.

    Note: Full data is still used for lookback calculations in strategies,
    but returns are only calculated for the specified date range.
    """
    all_dates = sorted(data[date_col].unique())

    # Filter to backtest period if specified
    if start_date is not None:
        dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
    else:
        dates = all_dates

    if end_date is not None:
        dates = [d for d in dates if d <= pd.Timestamp(end_date)]

    n_periods = len(dates)

    if n_periods < 2:
        raise ValueError(f"Not enough periods for backtest: {n_periods}")

    # Create mapping from filtered dates to all_dates indices for proper lookback
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # Get initial weights using actual index in all_dates
    start_idx = date_to_idx[dates[0]]
    portfolio_value = initial_capital
    weights = strategy.calculate_weights(data, start_idx)

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
            # Use actual index in all_dates for proper lookback in strategy
            actual_idx = date_to_idx[next_date]
            new_weights = strategy.calculate_weights(data, actual_idx)
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
    
    # Calculate stats with config parameters
    returns = np.array(trajectory['returns'])
    total_return = (portfolio_value / initial_capital) - 1.0

    # Sharpe ratio with risk-free rate
    if len(returns) >= 2:
        rf_per_period = risk_free_rate / annualization_factor
        excess_return = np.mean(returns) - rf_per_period
        sharpe = (excess_return / (np.std(returns, ddof=1) + 1e-8)) * np.sqrt(annualization_factor)
    else:
        sharpe = 0.0
    
    volatility = np.std(returns, ddof=1) * np.sqrt(annualization_factor)
    
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


# ============================================================================
# CUSTOM CALLBACKS (for debugging - not used in production)
# ============================================================================

class StopTrainingOnNoModelImprovementFixed(BaseCallback):
    """
    Fixed version of StopTrainingOnNoModelImprovement

    NOTE: Caused recursion issues in practice. Use SB3 version with patience-1 workaround.
    This class is kept for reference only - DO NOT USE IN PRODUCTION.

    FIXES:
    1. Uses '>=' instead of '>' for exact patience control
    2. Better logging with verbose mode
    3. Starts checking from eval #1 (min_evals for warmup only)

    Original SB3 bug: Uses '>' instead of '>=', causing 1 extra evaluation

    :param max_no_improvement_evals: Maximum number of consecutive evaluations
        without improvement. Will stop at EXACTLY this number (not +1).
    :param min_evals: Minimum number of evaluations before starting to check
        (warmup period). Set to 0 to check from eval #1, or 3 for literature standard.
    :param verbose: Verbosity level: 0=silent, 1=stopping only, 2=every eval
    """

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0

    def _on_step(self) -> bool:
        assert self.parent is not None, \
            "``StopTrainingOnNoModelImprovementFixed`` callback must be used with an ``EvalCallback``"

        continue_training = True

        # Check if we should start monitoring (after warmup period)
        if self.n_calls > self.min_evals:
            current_best = self.parent.best_mean_reward

            # Check for improvement
            if current_best > self.last_best_mean_reward:
                self.no_improvement_evals = 0
                if self.verbose >= 2:
                    print(
                        f"  Eval {self.n_calls}: New best reward {current_best:.4f} "
                        f"(prev {self.last_best_mean_reward:.4f}), reset counter"
                    )
            else:
                self.no_improvement_evals += 1
                if self.verbose >= 2:
                    print(
                        f"  Eval {self.n_calls}: No improvement "
                        f"(best {current_best:.4f}), counter = {self.no_improvement_evals}"
                    )

                # FIXED: Use '>=' instead of '>' for exact patience
                if self.no_improvement_evals >= self.max_no_improvement_evals:
                    continue_training = False

            self.last_best_mean_reward = current_best

        else:
            # Still in warmup period
            if self.verbose >= 2:
                print(f"  Eval {self.n_calls}: Warmup period (min_evals={self.min_evals})")
            self.last_best_mean_reward = self.parent.best_mean_reward

        # Print stopping message
        if self.verbose >= 1 and not continue_training:
            print(
                f"\n{'='*80}\n"
                f"Early stopping triggered at eval {self.n_calls}\n"
                f"No improvement for {self.no_improvement_evals} consecutive evaluations\n"
                f"Best mean reward: {self.parent.best_mean_reward:.4f}\n"
                f"{'='*80}"
            )

        return continue_training


class EvalCallbackWithLogging(EvalCallback):
    """
    Extended EvalCallback with detailed logging

    Logs evaluation metrics to help diagnose training issues
    """

    def __init__(self, *args, log_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_history = []

    def _on_step(self) -> bool:
        """Called after each evaluation"""
        result = super()._on_step()

        # Log evaluation results
        if len(self.evaluations_results) > 0:
            mean_reward = np.mean(self.evaluations_results[-1])
            std_reward = np.std(self.evaluations_results[-1])

            self.eval_history.append({
                'timestep': self.num_timesteps,
                'eval_num': len(self.evaluations_results),
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'best_mean_reward': self.best_mean_reward
            })

        return result
