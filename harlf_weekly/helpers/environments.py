"""
Portfolio Environment for Multi-Hierarchical RL System.

This module implements the portfolio optimization environment compatible with
OpenAI Gym/Gymnasium interface and Stable-Baselines3.

The environment simulates weekly portfolio rebalancing with:
- Transaction costs
- Position constraints
- Risk management (max drawdown, leverage limits)
- Flexible reward functions
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any

from .rewards import create_reward_function
from .config import EnvironmentConfig, RewardConfig, N_ASSETS, TICKERS


class PortfolioEnv(gym.Env):
    """
    Portfolio Optimization Environment.

    Action Space:
        Continuous, shape (n_assets,): Target portfolio weights for each asset
        Weights are normalized to sum to 1.0 (fully invested)

    Observation Space:
        Continuous, shape (n_features,): Current market features
        Features depend on agent type (technical/sentiment/macro)

    Rewards:
        Configurable via reward_type parameter
        Default: EMA Sharpe ratio
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        reward_type: str = 'ema_sharpe',
        reward_kwargs: Optional[Dict] = None,
        transaction_cost: float = EnvironmentConfig.TRANSACTION_COST,
        initial_capital: float = EnvironmentConfig.INITIAL_CAPITAL,
        max_drawdown: float = EnvironmentConfig.MAX_DRAWDOWN,
        normalize_observations: bool = EnvironmentConfig.NORMALIZE_OBSERVATIONS,
        include_positions: bool = True
    ):
        """
        Initialize Portfolio Environment.

        Args:
            features: Feature array, shape (n_steps, n_assets, n_features)
            returns: Returns array, shape (n_steps, n_assets)
            reward_type: Type of reward function
            reward_kwargs: Keyword arguments for reward function
            transaction_cost: Transaction cost per trade (e.g., 0.001 = 0.1%)
            initial_capital: Initial portfolio value
            max_drawdown: Maximum drawdown before forced liquidation
            normalize_observations: Whether to normalize observations
            include_positions: Include current positions in observation
        """
        super().__init__()

        # Data
        self.features = features
        self.returns = returns
        self.n_steps, self.n_assets, self.n_features = features.shape

        # Validate data shapes
        assert returns.shape == (self.n_steps, self.n_assets), \
            f"Returns shape {returns.shape} doesn't match features {self.n_steps, self.n_assets}"

        # Environment parameters
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.max_drawdown = max_drawdown
        self.normalize_observations = normalize_observations
        self.include_positions = include_positions

        # Reward function
        if reward_kwargs is None:
            reward_kwargs = {}
        self.reward_function = create_reward_function(reward_type, **reward_kwargs)

        # Define action space (portfolio weights)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # Define observation space
        obs_dim = self.n_assets * self.n_features
        if self.include_positions:
            obs_dim += self.n_assets  # Add current positions

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.positions = np.zeros(self.n_assets)  # Current portfolio weights
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.episode_returns = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset state
        self.current_step = 0
        self.positions = np.zeros(self.n_assets)  # Start with no positions
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.episode_returns = []

        # Reset reward function
        self.reward_function.reset()

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Target portfolio weights (n_assets,)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Normalize action to sum to 1.0 (fully invested)
        action = np.clip(action, 0, 1)
        if action.sum() > 0:
            action = action / action.sum()
        else:
            action = np.ones(self.n_assets) / self.n_assets  # Equal weight fallback

        # Calculate transaction costs
        trades = np.abs(action - self.positions)
        transaction_costs = trades.sum() * self.transaction_cost

        # Get current period returns
        period_returns = self.returns[self.current_step]

        # Calculate portfolio return before costs
        portfolio_return_gross = np.dot(self.positions, period_returns)

        # Apply transaction costs
        portfolio_return_net = portfolio_return_gross - transaction_costs

        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return_net)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Update positions (rebalance after return)
        self.positions = action.copy()

        # Store return
        self.episode_returns.append(portfolio_return_net)

        # Calculate reward
        reward = self.reward_function(portfolio_return_net)

        # Move to next step
        self.current_step += 1

        # Check termination conditions
        terminated = False
        truncated = False

        # Max drawdown check
        current_drawdown = (self.portfolio_value - self.peak_value) / self.peak_value
        if current_drawdown < -self.max_drawdown:
            terminated = True
            reward = RewardConfig.BANKRUPTCY_PENALTY

        # End of episode check
        if self.current_step >= self.n_steps:
            truncated = True

        # Get next observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns:
            Observation array
        """
        if self.current_step >= self.n_steps:
            # Episode ended, return zeros
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Get current features (n_assets, n_features)
        current_features = self.features[self.current_step]

        # Flatten features
        obs = current_features.flatten()

        # Add current positions if configured
        if self.include_positions:
            obs = np.concatenate([obs, self.positions])

        # Normalize if configured
        if self.normalize_observations:
            obs = obs / (np.abs(obs).max() + 1e-8)

        return obs.astype(np.float32)

    def _get_info(self) -> Dict:
        """
        Get info dictionary.

        Returns:
            Info dictionary with current state
        """
        info = {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy(),
            'peak_value': self.peak_value
        }

        # Add reward metrics if available
        if hasattr(self.reward_function, 'get_metrics'):
            info.update(self.reward_function.get_metrics())

        return info

    def render(self):
        """Render environment (not implemented)."""
        pass


# ============================================================================
# ENVIRONMENT FACTORY
# ============================================================================

def create_env(
    agent_type: str,
    split: str = 'train',
    reward_type: str = 'ema_sharpe',
    reward_kwargs: Optional[Dict] = None,
    **env_kwargs
) -> PortfolioEnv:
    """
    Factory function to create portfolio environment.

    Args:
        agent_type: Type of agent ('technical', 'sentiment', 'macro')
        split: Data split ('train', 'val', 'test')
        reward_type: Type of reward function
        reward_kwargs: Keyword arguments for reward function
        **env_kwargs: Additional environment keyword arguments

    Returns:
        PortfolioEnv instance
    """
    from .utils import prepare_env_data

    # Load data
    features, returns = prepare_env_data(agent_type, split)

    # Create environment
    env = PortfolioEnv(
        features=features,
        returns=returns,
        reward_type=reward_type,
        reward_kwargs=reward_kwargs,
        **env_kwargs
    )

    return env


def create_vectorized_env(
    agent_type: str,
    split: str = 'train',
    n_envs: int = 4,
    reward_type: str = 'ema_sharpe',
    reward_kwargs: Optional[Dict] = None,
    **env_kwargs
):
    """
    Create vectorized environment for parallel training.

    Args:
        agent_type: Type of agent
        split: Data split
        n_envs: Number of parallel environments
        reward_type: Type of reward function
        reward_kwargs: Reward function kwargs
        **env_kwargs: Environment kwargs

    Returns:
        DummyVecEnv with n_envs parallel environments
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        return create_env(
            agent_type=agent_type,
            split=split,
            reward_type=reward_type,
            reward_kwargs=reward_kwargs,
            **env_kwargs
        )

    # Create vectorized environment
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

    return vec_env


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    """Test portfolio environment."""

    print("Testing Portfolio Environment")
    print("="*70)

    # Create synthetic data
    np.random.seed(42)
    n_steps = 100
    n_assets = 7
    n_features = 20

    features = np.random.randn(n_steps, n_assets, n_features)
    returns = 0.01 + 0.02 * np.random.randn(n_steps, n_assets)

    print(f"\nSynthetic Data:")
    print(f"  Features shape: {features.shape}")
    print(f"  Returns shape: {returns.shape}")

    # Test 1: Basic environment
    print("\n1. Basic Environment Test:")
    env = PortfolioEnv(
        features=features,
        returns=returns,
        reward_type='ema_sharpe'
    )

    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")

    obs, info = env.reset()
    print(f"   Initial obs shape: {obs.shape}")
    print(f"   Initial info: {list(info.keys())}")

    # Take a few random steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"   Steps taken: {i+1}")
    print(f"   Total reward: {total_reward:.4f}")
    print(f"   Final portfolio value: ${info['portfolio_value']:.2f}")

    # Test 2: Different reward functions
    print("\n2. Different Reward Functions:")

    for reward_type in ['ema_sharpe', 'multi_objective', 'simple_return']:
        env = PortfolioEnv(
            features=features,
            returns=returns,
            reward_type=reward_type
        )

        obs, _ = env.reset()
        action = np.ones(n_assets) / n_assets  # Equal weight
        _, reward, _, _, _ = env.step(action)

        print(f"   {reward_type}: reward = {reward:.4f}")

    # Test 3: Episode completion
    print("\n3. Full Episode Test:")
    env = PortfolioEnv(
        features=features,
        returns=returns,
        reward_type='ema_sharpe'
    )

    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    print(f"   Episode length: {step_count} steps")
    print(f"   Final value: ${info['portfolio_value']:.2f}")
    print(f"   Return: {(info['portfolio_value']/env.initial_capital - 1)*100:.2f}%")

    print("\n" + "="*70)
    print("✅ Portfolio environment tested successfully")
