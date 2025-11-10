"""
Portfolio Optimization Environment for Reinforcement Learning

Gymnasium-compatible environment for training RL agents on portfolio allocation.
Supports walk-forward validation with proper temporal separation.

Features:
- Long format data handling (stacked by ticker)
- Transaction cost modeling
- Multiple reward functions (EMA Sharpe, Multi-Objective)
- Portfolio state tracking (weights, value, drawdown)
- Episode termination on max drawdown

Created: 2025-01-09
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, Literal
from pathlib import Path

from harlf.config import TICKERS, get_fold_files
from harlf.envs.rewards import EMASharpeReward, MultiObjectiveReward, SimpleReturnReward


class PortfolioEnv(gym.Env):
    """
    Portfolio Optimization Environment.

    State Space (154 dimensions):
        - Asset features: 22 features × 7 tickers = 154 dimensions
        - All features are already normalized via StandardScaler

    Action Space (7 dimensions):
        - Portfolio weights for 7 tickers [0, 1], must sum to 1.0
        - Enforced by Dirichlet policy

    Observation:
        Flattened feature vector [ticker1_feat1, ..., ticker1_feat23, ticker2_feat1, ...]

    Args:
        fold_id: Fold number for walk-forward validation (0-49)
        split: Data split to use ('train', 'val', 'test')
        reward_type: Reward function type ('ema_sharpe', 'multi_objective', 'simple')
        reward_kwargs: Additional kwargs for reward function
        transaction_cost_bps: Transaction cost in basis points (default: 15)
        max_drawdown_threshold: Episode termination threshold (default: 0.20 = -20%)
        enable_stop_loss: Whether to enable stop-loss termination (default: False)
        initial_balance: Starting portfolio value (default: 100000)
        include_benchmark: Whether to include benchmark returns in info (default: True)
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        fold_id: int,
        split: Literal['train', 'val', 'test'] = 'train',
        reward_type: Literal['ema_sharpe', 'multi_objective', 'simple'] = 'ema_sharpe',
        reward_kwargs: Optional[Dict[str, Any]] = None,
        transaction_cost_bps: float = 15.0,
        max_drawdown_threshold: float = 0.20,
        enable_stop_loss: bool = False,
        initial_balance: float = 100_000.0,
        include_benchmark: bool = True,
    ):
        super().__init__()

        self.fold_id = fold_id
        self.split = split
        self.transaction_cost_rate = transaction_cost_bps / 10000.0  # Convert bps to rate
        self.max_drawdown_threshold = max_drawdown_threshold
        self.enable_stop_loss = enable_stop_loss
        self.initial_balance = initial_balance
        self.include_benchmark = include_benchmark

        # Load data
        self._load_data()

        # Validate data integrity
        self._validate_data()

        # Define spaces
        self.n_assets = len(TICKERS)
        self.n_features = 22  # Number of features per ticker

        # Observation space: flattened features (22 × 7 = 154)
        obs_dim = self.n_features * self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space: portfolio weights [0, 1] summing to 1.0
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # Initialize reward function
        reward_kwargs = reward_kwargs or {}
        if reward_type == 'ema_sharpe':
            self.reward_fn = EMASharpeReward(**reward_kwargs)
        elif reward_type == 'multi_objective':
            self.reward_fn = MultiObjectiveReward(**reward_kwargs)
        elif reward_type == 'simple':
            self.reward_fn = SimpleReturnReward(**reward_kwargs)
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

        self.reward_type = reward_type

        # Episode state
        self.current_step = 0
        self.current_weights = None
        self.portfolio_value = None
        self.peak_value = None
        self.episode_returns = []
        self.episode_weights = []
        self.episode_turnovers = []

    def _load_data(self):
        """Load data for the specified fold and split."""
        fold_files = get_fold_files(self.fold_id)
        data_file = fold_files[self.split]

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Load data (long format: stacked by ticker)
        self.data = pd.read_csv(data_file, index_col=0)
        self.data.index = pd.to_datetime(self.data.index)

        # Get unique dates and tickers
        self.dates = sorted(self.data.index.unique())
        self.tickers = sorted(self.data['ticker'].unique())

        # Verify data integrity
        assert self.tickers == sorted(TICKERS), "Ticker mismatch in data"

        # Extract features (exclude 'ticker' column)
        self.feature_cols = [col for col in self.data.columns if col != 'ticker']

        # Create pivot tables for efficient access
        # Each pivot: rows=dates, cols=tickers, values=feature
        self.feature_pivots = {}
        for feature in self.feature_cols:
            self.feature_pivots[feature] = self.data.pivot_table(
                index=self.data.index,
                columns='ticker',
                values=feature,
                aggfunc='first'
            )

        # Compute daily returns for each ticker (for reward calculation)
        # Note: return_5d is a feature, but we need actual daily returns
        # We'll use return_5d / 5 as a proxy, or compute from price if available
        # For now, we'll extract the next-day return from the data shift

        # Since features are lagged by 1, the "future" return is in the next row
        # But we need to be careful: we want the return we GET from holding today's weights
        # This will be approximated using the 5-day return scaled down

        self.episode_length = len(self.dates)

    def _validate_data(self):
        """
        Validate that data contains required columns and passes sanity checks.
        """
        # Check that forward_log_return column exists
        if 'forward_log_return' not in self.feature_cols:
            raise ValueError(
                "❌ CRITICAL: 'forward_log_return' column not found in data!\n"
                "   This means you're using old data generated before the fix.\n"
                "   Please regenerate data by running the data pipeline:\n"
                "   python -m harlf.data_pipeline.pipeline"
            )

        # Check for NaN values in forward returns
        returns_data = self.feature_pivots['forward_log_return']
        nan_count = returns_data.isna().sum().sum()
        expected_nans = len(self.tickers)  # Last date for each ticker should be NaN

        if nan_count > expected_nans:
            print(f"⚠️  WARNING: Found {nan_count} NaN values in forward_log_return")
            print(f"   Expected only {expected_nans} (last date for each ticker)")

        # Validate return distribution
        valid_returns = returns_data.values.flatten()
        valid_returns = valid_returns[~np.isnan(valid_returns)]

        stats = {
            'mean': valid_returns.mean(),
            'std': valid_returns.std(),
            'min': valid_returns.min(),
            'max': valid_returns.max(),
            'abs_max': np.abs(valid_returns).max(),
        }

        print(f"\n📊 Data Validation - Forward Returns ({self.fold_id}, {self.split}):")
        print(f"   Mean: {stats['mean']:.6f} ({np.exp(stats['mean'])-1:.4%} simple)")
        print(f"   Std:  {stats['std']:.6f}")
        print(f"   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"   Max |return|: {stats['abs_max']:.4f} ({np.exp(stats['abs_max'])-1:.1%} simple)")

        # Sanity checks
        warnings = []
        if stats['abs_max'] > 0.30:  # |30%| log return ≈ ±35% simple return
            warnings.append(f"Very large returns detected (max: {np.exp(stats['abs_max'])-1:.1%})")

        if abs(stats['mean']) > 0.005:  # Mean should be close to zero
            warnings.append(f"Non-zero mean return ({stats['mean']*252:.1%} annualized)")

        if warnings:
            print(f"\n⚠️  Data Quality Warnings:")
            for w in warnings:
                print(f"   - {w}")
        else:
            print(f"   ✅ Data looks reasonable")

        print()

    def _get_observation(self, step: int) -> np.ndarray:
        """
        Get observation for a given step.

        Returns flattened feature vector: [ticker1_feats, ticker2_feats, ...]
        Shape: (23 × 7 = 161,)
        """
        date = self.dates[step]
        obs = []

        for ticker in self.tickers:
            ticker_feats = []
            for feature in self.feature_cols:
                value = self.feature_pivots[feature].loc[date, ticker]
                ticker_feats.append(value)
            obs.extend(ticker_feats)

        return np.array(obs, dtype=np.float32)

    def _get_daily_returns(self, step: int) -> np.ndarray:
        """
        Get forward-looking log returns for each ticker at the given step.

        These are the REALIZED returns from holding assets from step t to t+1.
        Uses log returns for better mathematical properties:
        - Time-additive
        - Symmetric
        - More stable numerically

        Returns:
            Array of log returns for each ticker
        """
        date = self.dates[step]
        log_returns = []

        for ticker in self.tickers:
            # Get the forward log return (calculated in data pipeline)
            log_return = self.feature_pivots['forward_log_return'].loc[date, ticker]

            # Sanity check: daily log returns should typically be in [-0.15, 0.15]
            # (corresponding to roughly ±14% simple return per day)
            if abs(log_return) > 0.20 and not np.isnan(log_return):
                print(f"⚠️  WARNING: Large return: {ticker} on {date}: {log_return:.4f} ({np.exp(log_return)-1:.2%} simple)")

            log_returns.append(log_return)

        return np.array(log_returns, dtype=np.float32)

    def _get_benchmark_return(self, step: int) -> float:
        """
        Get benchmark return (QQQ) for the given step.

        We can approximate using bench_relative_strength feature.
        Alternatively, compute from actual QQQ data if available.

        For simplicity, assume benchmark return = portfolio return / bench_relative_strength
        Or use a fixed benchmark return (e.g., 0 for simplicity in training)

        Returns:
            Benchmark daily return
        """
        # Simplified: use 0 for now (can be enhanced later)
        # TODO: Load actual QQQ returns if needed
        return 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset episode state
        self.current_step = 0
        self.current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets  # Equal weight
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance

        # Reset tracking
        self.episode_returns = []
        self.episode_weights = [self.current_weights.copy()]
        self.episode_turnovers = []

        # Reset reward function
        self.reward_fn.reset()

        # Get initial observation
        obs = self._get_observation(self.current_step)

        info = {
            'step': self.current_step,
            'date': str(self.dates[self.current_step]),
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights.copy(),
        }

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Portfolio weights (sum should be ~1.0)

        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode ended (max drawdown hit)
            truncated: Whether episode reached max steps
            info: Additional information
        """
        # Normalize action to ensure sum = 1.0 (safety check)
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)
        action = action / (action.sum() + 1e-10)

        # Compute turnover (portfolio rebalancing)
        turnover = np.sum(np.abs(action - self.current_weights))
        transaction_cost = turnover * self.transaction_cost_rate

        # Get log returns for each asset
        asset_log_returns = self._get_daily_returns(self.current_step)

        # Portfolio log return (approximate weighted average for small returns)
        # For small returns: log(1 + weighted_sum(r_i)) ≈ weighted_sum(log(1 + r_i))
        portfolio_log_return_gross = np.dot(action, asset_log_returns)

        # Transaction costs in log space
        # If we lose `transaction_cost` fraction of value:
        # new_value = old_value * (1 - transaction_cost)
        # log_return = log(1 - transaction_cost)
        transaction_cost_log = np.log(1.0 - transaction_cost) if transaction_cost > 0 else 0.0
        portfolio_log_return_net = portfolio_log_return_gross + transaction_cost_log

        # Update portfolio value (compound in log space)
        # value_{t+1} = value_t * exp(log_return)
        self.portfolio_value *= np.exp(portfolio_log_return_net)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Compute current drawdown
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # Compute reward (using log return)
        if self.reward_type == 'multi_objective':
            benchmark_return = self._get_benchmark_return(self.current_step)
            reward = self.reward_fn.compute(
                portfolio_return=portfolio_log_return_net,
                benchmark_return=benchmark_return,
                turnover=turnover
            )
        else:
            reward = self.reward_fn.compute(portfolio_return=portfolio_log_return_net)

        # Track episode metrics (store log returns)
        self.episode_returns.append(portfolio_log_return_net)
        self.episode_weights.append(action.copy())
        self.episode_turnovers.append(turnover)

        # Update state
        self.current_weights = action
        self.current_step += 1

        # Check termination conditions
        terminated = False
        truncated = False

        # Terminate if max drawdown exceeded (only if stop-loss enabled)
        if self.enable_stop_loss and current_drawdown >= self.max_drawdown_threshold:
            terminated = True

        # Truncate if reached end of episode
        if self.current_step >= self.episode_length:
            truncated = True

        # Get next observation (if not done)
        if terminated or truncated:
            # Episode ended - return last observation
            obs = self._get_observation(min(self.current_step - 1, self.episode_length - 1))
        else:
            obs = self._get_observation(self.current_step)

        # Prepare info dict
        info = {
            'step': self.current_step,
            'date': str(self.dates[min(self.current_step, self.episode_length - 1)]),
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_log_return_net,  # Log return
            'portfolio_return_simple': np.exp(portfolio_log_return_net) - 1.0,  # Convert to simple for display
            'gross_return': portfolio_log_return_gross,
            'transaction_cost': transaction_cost,
            'turnover': turnover,
            'weights': action.copy(),
            'drawdown': current_drawdown,
            'peak_value': self.peak_value,
        }

        # Add reward function info
        if hasattr(self.reward_fn, 'get_info'):
            if self.reward_type == 'multi_objective':
                reward_info = self.reward_fn.get_info(
                    portfolio_return=portfolio_return_net,
                    benchmark_return=self._get_benchmark_return(self.current_step - 1),
                    turnover=turnover
                )
            else:
                reward_info = self.reward_fn.get_info()

            info.update({f'reward_{k}': v for k, v in reward_info.items()})

        return obs, reward, terminated, truncated, info

    def get_episode_metrics(self) -> Dict[str, Any]:
        """
        Compute episode-level metrics from log returns.

        Returns:
            Dictionary of episode metrics (Sharpe, return, drawdown, etc.)
        """
        if len(self.episode_returns) == 0:
            return {}

        log_returns = np.array(self.episode_returns)  # These are log returns

        # Total return (convert from log to simple)
        total_log_return = log_returns.sum()
        total_return = np.exp(total_log_return) - 1.0

        # Mean and std of log returns
        mean_log_return = log_returns.mean()
        std_log_return = log_returns.std()

        # Sharpe ratio (annualized)
        sharpe = (mean_log_return / (std_log_return + 1e-8)) * np.sqrt(252)

        # Sortino (downside deviation)
        downside_log_returns = log_returns[log_returns < 0]
        if len(downside_log_returns) >= 10:  # Need enough data for reliable estimate
            downside_std = downside_log_returns.std()
            sortino = (mean_log_return / (downside_std + 1e-8)) * np.sqrt(252)
        else:
            sortino = np.nan  # Not enough downside data

        # Max drawdown (convert log returns to cumulative wealth)
        cumulative = np.exp(np.cumsum(log_returns))  # Cumulative wealth from log returns
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = drawdowns.max()

        # Turnover
        mean_turnover = np.mean(self.episode_turnovers) if len(self.episode_turnovers) > 0 else 0.0

        # Sanity checks
        if abs(total_return) > 5.0:  # 500% return is suspicious
            print(f"⚠️  WARNING: Extreme total return: {total_return:.2%} over {len(log_returns)} days")

        return {
            'total_return': total_return,
            'mean_daily_return': mean_log_return,  # Mean log return
            'volatility': std_log_return * np.sqrt(252),  # Annualized volatility
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'mean_turnover': mean_turnover,
            'final_value': self.portfolio_value,
            'episode_length': len(log_returns),
        }


# ============================================================================
# TESTING
# ============================================================================

def test_portfolio_env():
    """Test the PortfolioEnv."""
    print("="*80)
    print("TESTING PORTFOLIO ENVIRONMENT")
    print("="*80)

    # Test with fold 0, train split
    print("\n1. Initializing environment (fold_0, train, ema_sharpe)...")
    env = PortfolioEnv(
        fold_id=0,
        split='train',
        reward_type='ema_sharpe',
        transaction_cost_bps=15.0,
        max_drawdown_threshold=0.20,
    )

    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    print(f"   Episode length: {env.episode_length}")
    print(f"   Number of assets: {env.n_assets}")
    print(f"   Tickers: {env.tickers}")

    # Test reset
    print("\n2. Testing reset...")
    obs, info = env.reset(seed=42)
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation sample: {obs[:5]}")
    print(f"   Info: {info}")
    print(f"   Initial weights: {env.current_weights}")

    # Test step with random actions
    print("\n3. Testing steps (10 random actions)...")
    for i in range(10):
        # Random action (will be normalized by env)
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"   Step {i+1}: Reward={reward:+.4f}, Value={info['portfolio_value']:.2f}, "
              f"Turnover={info['turnover']:.4f}, Drawdown={info['drawdown']:.4f}")

        if terminated or truncated:
            print(f"   Episode ended: terminated={terminated}, truncated={truncated}")
            break

    # Get episode metrics
    print("\n4. Episode metrics:")
    metrics = env.get_episode_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")

    # Test multi-objective reward
    print("\n5. Testing multi-objective reward...")
    env_mo = PortfolioEnv(
        fold_id=0,
        split='train',
        reward_type='multi_objective',
        transaction_cost_bps=15.0,
    )
    obs, info = env_mo.reset(seed=42)
    action = env_mo.action_space.sample()
    obs, reward, terminated, truncated, info = env_mo.step(action)
    print(f"   Multi-objective reward: {reward:+.4f}")
    print(f"   Info keys: {[k for k in info.keys() if 'reward_' in k]}")

    print("\n" + "="*80)
    print("✅ PORTFOLIO ENVIRONMENT TESTS PASSED")
    print("="*80)


if __name__ == '__main__':
    test_portfolio_env()
