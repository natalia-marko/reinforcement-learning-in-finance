"""
RL Portfolio Rebalancing System
================================
Combined module containing models, environment, callbacks, and validation utilities.

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
        patience: int = 10,
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
            Early stopping patience (default: 10 evaluations)
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

        # Print validation results
        print(f"\n  Validation at step {self.num_timesteps}:")
        print(f"    Sharpe: {val_sharpe:.3f}")
        print(f"    Return: {stats.get('total_return', 0.0)*100:.2f}%")
        print(f"    Portfolio value: ${stats.get('portfolio_value', 0.0):,.2f}")

        # Check for improvement
        if val_sharpe > self.best_val_sharpe:
            self.best_val_sharpe = val_sharpe
            self.no_improvement_count = 0
            print(f"  ✓ New best! (Sharpe: {val_sharpe:.3f})")

            # Save best model
            best_path = self.fold_dir / "best_model"
            self.model.save(str(best_path))
        else:
            self.no_improvement_count += 1
            print(f"  No improvement ({self.no_improvement_count}/{self.patience})")

    def _on_training_end(self) -> None:
        """Save logs at end of training."""
        # Save train episode log
        if self.train_episodes:
            train_df = pd.DataFrame(self.train_episodes)
            train_df.to_csv(self.fold_dir / 'logs' / 'train_episodes.csv', index=False)
            print(f"\nTrain log saved: {len(self.train_episodes)} episodes")

        # Save validation log
        if self.val_evaluations:
            val_df = pd.DataFrame(self.val_evaluations)
            val_df.to_csv(self.fold_dir / 'logs' / 'val_evaluations.csv', index=False)
            print(f"Val log saved: {len(self.val_evaluations)} evaluations")


# ============================================================================
# MODELS
# ============================================================================

class SimpleActor(nn.Module):
    """
    Baseline simple actor network for portfolio allocation.

    Architecture:
    - Linear(state_dim, 256) -> ReLU -> Dropout(0.2)
    - Linear(256, 128) -> ReLU
    - Linear(128, n_assets) -> Softmax
    """

    def __init__(self, state_dim: int, n_assets: int, dropout: float = 0.2):
        """
        Initialize SimpleActor.

        Parameters:
        -----------
        state_dim : int
            Dimension of state space
        n_assets : int
            Number of assets in portfolio
        dropout : float
            Dropout probability (default: 0.2)
        """
        super(SimpleActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_assets)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor
            State tensor of shape (batch_size, state_dim)

        Returns:
        --------
        torch.Tensor
            Portfolio weights of shape (batch_size, n_assets)
        """
        logits = self.net(x)
        weights = F.softmax(logits, dim=-1)
        return weights


# ============================================================================
# PORTFOLIO ENVIRONMENT
# ============================================================================

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
        reward_lookback: int = 8,
        initial_capital: float = 100000.0,
        reward_type: str = 'log_return',
        seed: Optional[int] = None
    ):
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
        self.reward_type = reward_type

        # Get unique dates (sorted)
        self.dates = sorted(self.data['date'].unique())
        self.n_periods = len(self.dates)

        # State space: features per asset + current weights + portfolio stats + time embeddings
        n_features_per_asset = len(feature_cols)

        # ENHANCED: Added 4 new risk metrics (always use 7 stats)
        n_portfolio_stats = 7  # return, vol, sharpe, drawdown, entropy, turnover, vol_trend

        n_time_features = 2  # sin/cos month embeddings

        self.state_dim = (
            n_features_per_asset * self.n_assets +  # Asset features
            self.n_assets +  # Current weights
            n_portfolio_stats +  # Portfolio statistics (ENHANCED with risk metrics)
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

        # ENHANCED: Track additional state for risk metrics
        self.prev_weights = self.weights.copy()  # For turnover calculation
        self.portfolio_values = [self.initial_capital]  # For drawdown calculation
        self.weight_history = [self.weights.copy()]  # For turnover trend

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
            curr_price = self.data[(self.data['date'] == current_date) & 
                                   (self.data['ticker'] == ticker)]['close'].iloc[0]
            next_price = self.data[(self.data['date'] == next_date) & 
                                   (self.data['ticker'] == ticker)]['close'].iloc[0]
            price_changes.append(np.log(next_price / curr_price))
        
        # Calculate portfolio return EVERY STEP
        portfolio_return = np.sum(self.weights * np.array(price_changes))
        
        # Only update weights on rebalance
        should_rebalance = (self.current_step % self.rebalance_frequency == 0)
        if should_rebalance:
            # ENHANCED: Save previous weights before updating (for turnover calculation)
            self.prev_weights = self.weights.copy()

            new_weights = self._normalize_action(action)
            transaction_cost = self.transaction_cost * np.sum(np.abs(new_weights - self.weights))
            portfolio_return -= transaction_cost
            self.weights = new_weights
            self.rebalance_count += 1

            # ENHANCED: Track weight history for risk metrics
            self.weight_history.append(self.weights.copy())

        # ALWAYS append returns
        self.portfolio_returns.append(portfolio_return)
        self.portfolio_value *= np.exp(portfolio_return)

        # ENHANCED: Track portfolio value history for drawdown calculation
        # This allows agent to see current drawdown in observation
        self.portfolio_values.append(self.portfolio_value)

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
        # Softmax normalization
        action = np.clip(action, 0, 1)
        weights = action / (np.sum(action) + 1e-10)

        # Apply max weight constraint
        weights = np.clip(weights, 0, self.max_weight_per_asset)

        # Renormalize to sum to 1
        weights = weights / (np.sum(weights) + 1e-10)

        return weights.astype(np.float32)

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on reward_type.

        Reward Types:
        - 'log_return': Returns raw log portfolio return (simplest, best generalization)
        - 'multi_component': Multi-component reward with Sharpe, return, and drawdown

        Multi-component uses reward_lookback parameter to control lookback window.
        All components are normalized with tanh to prevent extreme values.

        Note: 'simple_return' is accepted as alias for 'log_return' for backward compatibility.
        """
        if len(self.portfolio_returns) < 1:
            return 0.0

        # Log return reward (formerly called 'simple_return')
        if self.reward_type in ['log_return', 'simple_return']:  # Accept both names
            # Just return the log return from last step
            return float(self.portfolio_returns[-1])

        # Multi-component reward (original approach with improvements)
        elif self.reward_type == 'multi_component':
            if len(self.portfolio_returns) < 2:
                return 0.0

            # Use reward_lookback parameter to control stability
            lookback = min(self.reward_lookback, len(self.portfolio_returns))
            recent_returns = np.array(self.portfolio_returns[-lookback:])

            if len(recent_returns) < 2:
                return 0.0

            # 1. Risk-adjusted return (40% weight)
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns, ddof=1) + 1e-8
            sharpe = mean_return / std_return * np.sqrt(52)
            sharpe_reward = np.tanh(sharpe / 3)  # Squash to [-1, 1]

            # 2. Absolute return (20% weight)
            total_return = np.exp(np.sum(recent_returns)) - 1
            return_reward = np.tanh(total_return * 10)  # Scale appropriately

            # 3. Drawdown penalty (20% weight)
            cum_returns = np.exp(np.cumsum(recent_returns))
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)  # Most negative = worst drawdown
            drawdown_penalty = np.tanh(max_drawdown * 10)  # Normalize

            # 4. Volatility penalty (20% weight) - for risk-averse investors
            # Annualized volatility
            annualized_vol = std_return * np.sqrt(52)
            # Target vol: 35% (between aggressive 48% and conservative 25%)
            # Penalty increases as vol exceeds target
            vol_excess = max(0, annualized_vol - 0.35)
            vol_penalty = -np.tanh(vol_excess * 5)  # Negative penalty for high vol

            # 5. Turnover penalty - reduce excessive trading
            # Compare current weights to previous weights
            turnover = np.sum(np.abs(self.weights - self.prev_weights))
            turnover_penalty = -0.1 * turnover  # Direct penalty (turnover is 0 to 2)

            # Combined reward (rebalanced for risk-averse profile)
            reward = (0.4 * sharpe_reward +      # Sharpe already accounts for risk
                     0.2 * return_reward +       # Still want growth
                     0.2 * drawdown_penalty +    # Penalize losses
                     0.2 * vol_penalty +         # Penalize high volatility
                     turnover_penalty)           # Penalize excessive trading (separate)

            return float(np.clip(reward, -2.0, 2.0))  # Reasonable bounds

        # ENHANCED: Custom risk-aware reward (addresses all identified problems)
        # This is the BEST reward for your use case based on diagnostic analysis
        elif self.reward_type == 'risk_aware':
            """
            Custom 5-component reward that explicitly penalizes ALL risk factors:

            PROBLEM ADDRESSED:
            - Your diagnostic showed: Training Sharpe 2-3.5 (unrealistic) vs Val Sharpe 0.5-1.0
            - Root cause: simple_return doesn't penalize risk, agent takes huge risks
            - Solution: Explicitly penalize vol, drawdown, concentration, turnover

            COMPONENTS (with detailed reasoning):
            1. Mean return (30%) - Still want growth
            2. Volatility penalty (25%) - Penalize variance directly
            3. Drawdown penalty (25%) - EXPONENTIAL penalty for large drawdowns
            4. Diversification bonus (10%) - Reward balanced portfolios
            5. Turnover penalty (10%) - Discourage excessive trading
            """
            if len(self.portfolio_returns) < 2:
                return 0.0

            # Use longer lookback for stability (as recommended)
            lookback = min(12, len(self.portfolio_returns))  # 12 weeks = ~3 months
            recent_returns = np.array(self.portfolio_returns[-lookback:])

            if len(recent_returns) < 2:
                return 0.0

            # === COMPONENT 1: Mean Return (30% weight) ===
            # Still want positive returns, just not at ANY cost
            mean_return = np.mean(recent_returns)
            return_component = np.tanh(mean_return * 100)  # Scale to [-1, 1]

            # === COMPONENT 2: Volatility Penalty (25% weight) ===
            # Explicitly penalize high volatility
            # Target: < 30% annualized vol
            volatility = np.std(recent_returns, ddof=1) * np.sqrt(52)
            # Penalty increases as vol > 0.30 (30%)
            vol_penalty = -np.tanh((volatility - 0.30) / 0.15)  # Negative for high vol
            # Example: 20% vol → +0.5 reward, 40% vol → -0.5 penalty, 60% vol → -0.9 penalty

            # === COMPONENT 3: Drawdown Penalty (25% weight) - EXPONENTIAL ===
            # Your diagnostic showed -34% drawdowns on average - this fixes it
            cum_returns = np.exp(np.cumsum(recent_returns))
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)  # Most negative value

            # EXPONENTIAL penalty - much stronger than multi_component
            # -10% DD → -0.4, -20% DD → -0.8, -30% DD → -0.95 (very bad!)
            dd_penalty = -np.tanh(np.exp(abs(max_drawdown) * 5) - 1)

            # === COMPONENT 4: Diversification Bonus (10% weight) ===
            # Reward balanced portfolios, penalize concentration
            # Uses Shannon entropy of weights
            weights_safe = self.weights + 1e-8  # Avoid log(0)
            entropy = -np.sum(weights_safe * np.log(weights_safe))
            max_entropy = np.log(self.n_assets)  # Perfectly diversified
            diversification_bonus = entropy / max_entropy  # [0, 1]
            # Example: Equal weights (7 assets) → 1.0, All in one asset → 0.0

            # === COMPONENT 5: Turnover Penalty (10% weight) ===
            # Your diagnostic showed 18.7% turnover - reduce it
            # Penalize excessive rebalancing (wastes money on transaction costs)
            turnover = np.sum(np.abs(self.weights - self.prev_weights))
            # Penalty if turnover > 0.20 (20%)
            turnover_penalty = -np.tanh((turnover - 0.20) * 5)
            # Example: 10% turnover → +0.4, 30% turnover → -0.4, 50% turnover → -0.9

            # === COMBINE ALL COMPONENTS ===
            reward = (
                0.30 * return_component +       # Want growth
                0.25 * vol_penalty +             # Penalize volatility
                0.25 * dd_penalty +              # Penalize drawdowns (exponential!)
                0.10 * diversification_bonus +   # Reward diversification
                0.10 * turnover_penalty          # Penalize excessive trading
            )

            # Clip to reasonable bounds
            return float(np.clip(reward, -2.0, 2.0))

        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}. Use 'log_return', 'simple_return', 'multi_component', or 'risk_aware'")

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

        # 3. Portfolio statistics (ENHANCED with 4 new risk metrics)
        # Old: return, vol, sharpe (3 metrics)
        # New: return, vol, sharpe, drawdown, entropy, turnover, vol_trend (7 metrics)
        if len(self.portfolio_returns) >= 2:
            lookback = min(self.reward_lookback, len(self.portfolio_returns))

            # Basic metrics (original 3)
            portfolio_return = np.mean(self.portfolio_returns[-lookback:])
            portfolio_vol = np.std(self.portfolio_returns[-lookback:], ddof=1)
            portfolio_sharpe = self._calculate_reward()

            # NEW METRIC 1: Current drawdown
            # Shows agent how far portfolio is from peak value
            # Helps agent learn: "if drawdown is large, reduce risk"
            if len(self.portfolio_values) > 1:
                running_max = max(self.portfolio_values)
                current_drawdown = (self.portfolio_value - running_max) / running_max
            else:
                current_drawdown = 0.0

            # NEW METRIC 2: Weight entropy (diversification measure)
            # High entropy = diversified (good), Low entropy = concentrated (risky)
            # Helps agent learn: "maintain diversification for stability"
            weights_safe = self.weights + 1e-8  # Avoid log(0)
            weight_entropy = -np.sum(weights_safe * np.log(weights_safe))
            # Normalize to [0,1] where 1 = perfectly diversified
            max_entropy = np.log(self.n_assets)
            weight_entropy = weight_entropy / max_entropy if max_entropy > 0 else 0.0

            # NEW METRIC 3: Recent turnover
            # Shows how much weights changed in last rebalance
            # Helps agent learn: "excessive turnover wastes money on fees"
            recent_turnover = np.sum(np.abs(self.weights - self.prev_weights))

            # NEW METRIC 4: Volatility trend
            # Is volatility increasing or decreasing?
            # Helps agent learn: "when vol rising, shift to safer assets"
            if len(self.portfolio_returns) >= 24:
                vol_recent = np.std(self.portfolio_returns[-12:], ddof=1)
                vol_older = np.std(self.portfolio_returns[-24:-12], ddof=1)
                vol_trend = (vol_recent - vol_older) / (vol_older + 1e-8)
            else:
                vol_trend = 0.0

        else:
            # Not enough data yet - use zeros
            portfolio_return = 0.0
            portfolio_vol = 0.0
            portfolio_sharpe = 0.0
            current_drawdown = 0.0
            weight_entropy = 1.0  # Start with max entropy (equal weights)
            recent_turnover = 0.0
            vol_trend = 0.0

        # Package all 7 portfolio statistics
        portfolio_stats = [
            portfolio_return, portfolio_vol, portfolio_sharpe,
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
        
        # Calculate ACTUAL Sharpe ratio (not the normalized reward)
        if len(returns) >= 2:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1) + 1e-8
            sharpe = (mean_return / std_return) * np.sqrt(52)  # Annualized Sharpe
        else:
            sharpe = 0.0
        
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


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def create_walk_forward_folds(
    train_data: pd.DataFrame,
    n_folds: int = 5,
    min_train_size: Optional[int] = None,
    date_col: str = 'date'
) -> List[Dict[str, pd.DataFrame]]:
    """
    Create chronological folds for walking forward validation.

    Proper walk-forward validation ensures no data leakage:
    - Fold 0: train=[0:T0], val=[T0:T1]
    - Fold 1: train=[0:T1], val=[T1:T2]  (expands training, non-overlapping val)
    - Fold 2: train=[0:T2], val=[T2:T3]  (expands training, non-overlapping val)

    Each fold's validation set becomes part of the next fold's training set.

    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data (will be split into folds)
    n_folds : int
        Number of folds to create (default: 5)
    min_train_size : Optional[int]
        Minimum number of periods in training fold (default: None = auto)
    date_col : str
        Date column name (default: 'date')

    Returns:
    --------
    List[Dict[str, pd.DataFrame]]
        List of fold dictionaries, each with 'train' and 'val' keys
    """
    # Get unique dates sorted
    dates = sorted(train_data[date_col].unique())
    n_periods = len(dates)

    # Calculate minimum train size (use 20% of data if not specified)
    if min_train_size is None:
        min_train_size = max(52, int(n_periods * 0.2))  # At least 1 year or 20%

    # Calculate validation size per fold
    # Reserve enough data for all validation folds
    remaining_periods = n_periods - min_train_size
    val_size_per_fold = remaining_periods // n_folds

    # Ensure at least 4 weeks per validation fold
    if val_size_per_fold < 4:
        val_size_per_fold = 4

    # Calculate fold boundaries
    # Each fold uses expanding training window and non-overlapping validation
    folds = []

    for fold_idx in range(n_folds):
        # Calculate split points for this fold
        val_start_idx = min_train_size + (fold_idx * val_size_per_fold)
        val_end_idx = val_start_idx + val_size_per_fold

        # Last fold gets all remaining data
        if fold_idx == n_folds - 1:
            val_end_idx = n_periods

        # Ensure we don't go beyond available data
        if val_start_idx >= n_periods:
            break
        if val_end_idx > n_periods:
            val_end_idx = n_periods

        # Training: from start to validation start (expanding window)
        train_dates = dates[:val_start_idx]
        # Validation: non-overlapping window
        val_dates = dates[val_start_idx:val_end_idx]

        # Skip if validation set is empty
        if len(val_dates) == 0:
            continue

        # Split data
        train_fold = train_data[train_data[date_col].isin(train_dates)].copy()
        val_fold = train_data[train_data[date_col].isin(val_dates)].copy()

        if len(train_fold) > 0 and len(val_fold) > 0:
            folds.append({
                'train': train_fold,
                'val': val_fold,
                'fold_idx': fold_idx,
                'train_start': train_fold[date_col].min(),
                'train_end': train_fold[date_col].max(),
                'val_start': val_fold[date_col].min(),
                'val_end': val_fold[date_col].max(),
                'n_train_periods': len(train_dates),
                'n_val_periods': len(val_dates)
            })

    return folds


def print_fold_summary(folds: List[Dict[str, pd.DataFrame]]):
    """Print summary of created folds."""
    print("=" * 80)
    print("WALKING FORWARD VALIDATION FOLDS")
    print("=" * 80)

    for fold in folds:
        print(f"\nFold {fold['fold_idx']}:")
        print(f"  Train: {fold['train_start'].date()} to {fold['train_end'].date()} ({fold['n_train_periods']} periods)")
        print(f"  Val:   {fold['val_start'].date()} to {fold['val_end'].date()} ({fold['n_val_periods']} periods)")
        print(f"  Train samples: {len(fold['train'])}")
        print(f"  Val samples:   {len(fold['val'])}")

    print("\n" + "=" * 80)


# ============================================================================
# BASELINE STRATEGIES
# ============================================================================

class BaselineStrategy:
    """Base class for baseline strategies"""

    def __init__(self, name: str, tickers: List[str], max_weight_per_asset: float):
        self.name = name
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.max_weight_per_asset = max_weight_per_asset

    def calculate_weights(self, data: pd.DataFrame, step: int) -> np.ndarray:
        raise NotImplementedError

    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize to sum=1 and respect max constraint"""
        weights = np.clip(weights, 0, self.max_weight_per_asset)
        return weights / (weights.sum() + 1e-10)


class EqualWeight(BaselineStrategy):
    """Equal weight (1/N) strategy"""
    
    def __init__(self, tickers: List[str], max_weight_per_asset: float):
        super().__init__("Equal Weight", tickers, max_weight_per_asset)

    def calculate_weights(self, data, step):
        return np.ones(self.n_assets) / self.n_assets


class BuyHold(BaselineStrategy):
    """Buy and hold - no rebalancing"""
    
    def __init__(self, tickers: List[str], max_weight_per_asset: float):
        super().__init__("Buy & Hold", tickers, max_weight_per_asset)
        self.initial_weights = None

    def calculate_weights(self, data, step):
        if self.initial_weights is None:
            self.initial_weights = np.ones(self.n_assets) / self.n_assets
        return self.initial_weights


class RiskParity(BaselineStrategy):
    """Risk parity - weight inversely to volatility"""
    
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

        vols = []
        for ticker in self.tickers:
            ticker_data = data[(data['ticker'] == ticker) & data[self.date_col].isin(lookback_dates)]
            if len(ticker_data) > 1:
                rets = ticker_data['close'].pct_change().dropna()
                vol = rets.std() if rets.std() > 0 else 1.0
            else:
                vol = 1.0
            vols.append(vol)

        inv_vol = 1.0 / np.array(vols)
        return self.normalize_weights(inv_vol)


class Momentum(BaselineStrategy):
    """Momentum - weight by past returns"""
    
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
            ticker_data = data[(data['ticker'] == ticker) & data[self.date_col].isin(lookback_dates)]
            if len(ticker_data) > 1:
                total_return = (ticker_data['close'].iloc[-1] / ticker_data['close'].iloc[0]) - 1
                returns.append(total_return)
            else:
                returns.append(0.0)

        returns = np.array(returns)
        if returns.min() < 0:
            returns = returns - returns.min() + 0.01

        return self.normalize_weights(returns)


class MinVariance(BaselineStrategy):
    """Minimum variance portfolio optimization"""
    
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
            ticker_data = data[(data['ticker'] == ticker) & data[self.date_col].isin(lookback_dates)]
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
                      seed: int = 42) -> Dict:
    """
    Backtest RL agent on test data.
    
    Parameters:
    -----------
    model : PPO
        Trained PPO model
    data : pd.DataFrame
        Test data with date, ticker, close columns and features
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
        Reward lookback period
    initial_capital : float
        Starting capital
    reward_type : str
        Reward function type
    seed : int
        Random seed
        
    Returns:
    --------
    Dict with 'trajectory', 'stats', and 'name' keys
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
            seed=seed
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
                     initial_capital: float) -> Dict:
    """
    Backtest baseline strategy on test data.
    
    Parameters:
    -----------
    strategy : BaselineStrategy
        Strategy object with calculate_weights() method
    data : pd.DataFrame
        Test data with date, ticker, close columns
    date_col : str
        Name of date column
    tickers : List[str]
        List of ticker symbols
    rebalance_frequency : int
        Rebalance every N periods
    transaction_cost : float
        Transaction cost per unit of turnover
    initial_capital : float
        Starting capital
        
    Returns:
    --------
    Dict with 'trajectory', 'stats', and 'name' keys
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
            curr = data[(data[date_col] == current_date) & (data['ticker'] == ticker)]['close'].iloc[0]
            next = data[(data[date_col] == next_date) & (data['ticker'] == ticker)]['close'].iloc[0]
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

    # Calculate stats
    returns = np.array(trajectory['returns'])
    total_return = (portfolio_value / initial_capital) - 1.0

    if len(returns) >= 2:
        sharpe = (np.mean(returns) / (np.std(returns, ddof=1) + 1e-8)) * np.sqrt(52)
    else:
        sharpe = 0.0

    volatility = np.std(returns) * np.sqrt(52)

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
