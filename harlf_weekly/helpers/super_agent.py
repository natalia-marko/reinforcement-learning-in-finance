"""
Hierarchical Agent Module for Multi-Hierarchical RL Portfolio System.

This module implements the hierarchical structure:
- Base Agents (Technical & Sentiment) → trained independently
- Super Agent → blends base agent outputs + lagged returns
- Meta Agent → adjusts super agent with macro data

Architecture:
    Technical Agent ─┐
                     ├→ Super Agent → Meta Agent → Final Portfolio
    Sentiment Agent ─┘
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from stable_baselines3 import PPO

from .environments import PortfolioEnv
from .utils import prepare_env_data, load_features, load_returns
from .config import N_ASSETS, TICKERS, DATA_DIR, MODELS_DIR


# ============================================================================
# BASE AGENT WRAPPER
# ============================================================================

class BaseAgentWrapper:
    """
    Wrapper for trained base agents.

    Loads a trained agent and provides interface to get portfolio recommendations.
    """

    def __init__(
        self,
        agent_type: str,
        model_path: Path,
        split: str = 'train'
    ):
        """
        Initialize base agent wrapper.

        Args:
            agent_type: Type of agent ('technical', 'sentiment')
            model_path: Path to trained model (.zip file)
            split: Data split to operate on
        """
        self.agent_type = agent_type
        self.model_path = model_path
        self.split = split

        # Load model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = PPO.load(model_path)

        # Load data
        self.features, self.returns = prepare_env_data(agent_type, split)
        self.n_steps = self.features.shape[0]

        self.current_step = 0
        # Start with zero positions (100% cash) - agent learns to build portfolio from scratch
        # This is intentional and allows the agent to learn optimal entry points
        self.positions = np.zeros(N_ASSETS)

    def reset(self):
        """Reset to beginning of episode."""
        self.current_step = 0
        # state for each episode
        self.positions = np.ones(N_ASSETS) / N_ASSETS

    def get_action(self, step: Optional[int] = None) -> np.ndarray:
        """
        Get portfolio recommendation from base agent.

        Args:
            step: Specific step to get action for (default: current_step)

        Returns:
            Portfolio weights (n_assets,)
        """
        if step is None:
            step = self.current_step

        if step >= self.n_steps:
            raise ValueError(f"Step {step} exceeds data length {self.n_steps}")

        # Get observation: features + current positions
        # Base agents expect: [flattened_features, positions]
        features_flat = self.features[step].flatten().astype(np.float32)
        obs = np.concatenate([features_flat, self.positions]).astype(np.float32)

        # Get action from model
        action, _ = self.model.predict(obs, deterministic=True)

        # Normalize to sum to 1
        if action.sum() > 0:
            action = action / action.sum()
        else:
            action = np.ones(N_ASSETS) / N_ASSETS

        # Update positions for next call
        self.positions = action.copy()
        
        return action

    def step(self) -> np.ndarray:
        """
        Get action for current step and advance.

        Returns:
            Portfolio weights
        """
        action = self.get_action()
        self.current_step += 1
        return action


# ============================================================================
# SUPER AGENT ENVIRONMENT
# ============================================================================

class SuperAgentEnv(gym.Env):
    """
    Super Agent Environment.

    Blends outputs from base agents (Technical & Sentiment) with lagged returns.

    Observation Space:
        - Base agent actions (technical): n_assets
        - Base agent actions (sentiment): n_assets
        - Lagged returns (1w, 2w, 4w): n_assets * 3
        Total: n_assets * 5

    Action Space:
        - Blending weights for agents: 2 (technical_weight, sentiment_weight)
        Normalized to sum to 1.0
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        technical_agent: BaseAgentWrapper,
        sentiment_agent: BaseAgentWrapper,
        returns: np.ndarray,
        reward_type: str = 'ema_sharpe',
        reward_kwargs: Optional[Dict] = None,
        transaction_cost: float = 0.001,
        initial_capital: float = 100000
    ):
        """
        Initialize Super Agent Environment.

        Args:
            technical_agent: Trained technical base agent
            sentiment_agent: Trained sentiment base agent
            returns: Returns array (n_steps, n_assets)
            reward_type: Type of reward function
            reward_kwargs: Reward function kwargs
            transaction_cost: Transaction cost per trade
            initial_capital: Initial portfolio value
        """
        super().__init__()

        self.technical_agent = technical_agent
        self.sentiment_agent = sentiment_agent
        self.returns = returns
        self.n_steps = returns.shape[0]
        self.n_assets = N_ASSETS
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

        # Verify data alignment
        assert technical_agent.n_steps == sentiment_agent.n_steps == self.n_steps, \
            "Agent data lengths must match"

        # Reward function
        from .rewards import create_reward_function
        if reward_kwargs is None:
            reward_kwargs = {}
        self.reward_function = create_reward_function(reward_type, **reward_kwargs)

        # Action space: blending weights for 2 base agents
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Observation space: base actions + lagged returns
        obs_dim = self.n_assets * 5  # 2 agents * n_assets + 3 lags * n_assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # State
        self.current_step = 0
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.positions = np.zeros(self.n_assets)
        self.episode_returns = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        # Reset base agents
        self.technical_agent.reset()
        self.sentiment_agent.reset()

        # Reset state
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.positions = np.zeros(self.n_assets)
        self.episode_returns = []

        # Reset reward
        self.reward_function.reset()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step."""
        # FIXED: Constrain blending weights to force balanced blending, min 0.2/max 0.8
        weights = np.clip(action, 0.2, 0.8)
        weights = weights / weights.sum()

        # Get base agent actions and blend
        tech_action = self.technical_agent.step()
        sent_action = self.sentiment_agent.step()
        blended_action = weights[0] * tech_action + weights[1] * sent_action

        # Normalize blended action
        if blended_action.sum() > 0:
            blended_action = blended_action / blended_action.sum()
        else:
            blended_action = np.ones(self.n_assets) / self.n_assets

        # Calculate transaction costs
        trades = np.abs(blended_action - self.positions)
        transaction_costs = trades.sum() * self.transaction_cost

        # Get returns
        period_returns = self.returns[self.current_step]

        # Calculate portfolio return
        portfolio_return_gross = np.dot(self.positions, period_returns)
        portfolio_return_net = portfolio_return_gross - transaction_costs

        # Update portfolio
        self.portfolio_value *= (1 + portfolio_return_net)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.positions = blended_action.copy()
        self.episode_returns.append(portfolio_return_net)

        # FIXED: Add diversity bonus to reward (entropy of blend weights)
        blend_entropy = -np.sum(weights * np.log(weights + 1e-8))
        reward = self.reward_function(portfolio_return_net) + 0.1 * blend_entropy

        # Advance step
        self.current_step += 1

        # Check termination
        terminated = False
        truncated = False

        if self.current_step >= self.n_steps:
            truncated = True

        # Get next observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def advance_step(self):
        """
        Advance step counters for all nested components.

        This method ensures proper synchronization of step counters
        across the SuperAgentEnv and its nested base agents.
        """
        self.current_step += 1
        self.technical_agent.current_step += 1
        self.sentiment_agent.current_step += 1

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step >= self.n_steps:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Base agent actions (next step)
        tech_action = self.technical_agent.get_action(self.current_step)
        sent_action = self.sentiment_agent.get_action(self.current_step)

        # Lagged returns (1w, 2w, 4w)
        lag_1w = self.returns[max(0, self.current_step - 1)]
        lag_2w = self.returns[max(0, self.current_step - 2)]
        lag_4w = self.returns[max(0, self.current_step - 4)]

        # Concatenate
        obs = np.concatenate([
            tech_action,
            sent_action,
            lag_1w,
            lag_2w,
            lag_4w
        ])

        return obs.astype(np.float32)

    def _get_info(self) -> Dict:
        """Get info dictionary."""
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy()
        }

    def render(self):
        """Render (not implemented)."""
        pass


# ============================================================================
# META AGENT ENVIRONMENT
# ============================================================================

class MetaAgentEnv(gym.Env):
    """
    Meta Agent Environment.

    Adjusts Super Agent output with macro features.

    Observation Space:
        - Super agent action: n_assets
        - Macro features: 12
        Total: n_assets + 12

    Action Space:
        - Adjustment factor for super agent: n_assets
        (applied as: final_action = super_action * adjustment)
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        super_agent: PPO,
        super_env: SuperAgentEnv,
        macro_features: np.ndarray,
        returns: np.ndarray,
        reward_type: str = 'ema_sharpe',
        reward_kwargs: Optional[Dict] = None,
        transaction_cost: float = 0.001,
        initial_capital: float = 100000
    ):
        """
        Initialize Meta Agent Environment.

        Args:
            super_agent: Trained super agent model
            super_env: Super agent environment
            macro_features: Macro features array (n_steps, n_features)
            returns: Returns array (n_steps, n_assets)
            reward_type: Type of reward function
            reward_kwargs: Reward function kwargs
            transaction_cost: Transaction cost
            initial_capital: Initial capital
        """
        super().__init__()

        self.super_agent = super_agent
        self.super_env = super_env
        self.macro_features = macro_features
        self.returns = returns
        self.n_steps = returns.shape[0]
        self.n_assets = N_ASSETS
        self.n_macro_features = macro_features.shape[1] if len(macro_features.shape) > 1 else 12
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

        # Reward function
        from .rewards import create_reward_function
        if reward_kwargs is None:
            reward_kwargs = {}
        self.reward_function = create_reward_function(reward_type, **reward_kwargs)

        # Action space: adjustment multipliers for each asset
        self.action_space = spaces.Box(
            low=0.0,
            high=2.0,  # Can increase or decrease super agent weights
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # Observation space: super action + macro features
        obs_dim = self.n_assets + self.n_macro_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # State
        self.current_step = 0
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.positions = np.zeros(self.n_assets)
        self.episode_returns = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        # Reset super env
        self.super_env.reset()

        # Reset state
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.positions = np.zeros(self.n_assets)
        self.episode_returns = []

        # Reset reward
        self.reward_function.reset()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step."""
        # Get super agent blend weights (2,)
        super_obs = self.super_env._get_observation()
        blend_weights, _ = self.super_agent.predict(super_obs, deterministic=True)

        # Normalize blend weights
        blend_weights = np.clip(blend_weights, 0, 1)
        if blend_weights.sum() > 0:
            blend_weights = blend_weights / blend_weights.sum()
        else:
            blend_weights = np.array([0.5, 0.5])

        # Get base agent actions to create blended portfolio (7,)
        tech_action = self.super_env.technical_agent.get_action(self.current_step)
        sent_action = self.super_env.sentiment_agent.get_action(self.current_step)

        # Blend base agents to get portfolio (7,)
        blended_portfolio = blend_weights[0] * tech_action + blend_weights[1] * sent_action

        # Normalize blended portfolio
        if blended_portfolio.sum() > 0:
            blended_portfolio = blended_portfolio / blended_portfolio.sum()
        else:
            blended_portfolio = np.ones(self.n_assets) / self.n_assets

        # Apply meta agent adjustment to blended portfolio
        adjustment = np.clip(action, 0, 2)
        adjusted_action = blended_portfolio * adjustment

        # Normalize adjusted action
        if adjusted_action.sum() > 0:
            adjusted_action = adjusted_action / adjusted_action.sum()
        else:
            adjusted_action = blended_portfolio  # Fallback to blended portfolio

        # Calculate transaction costs
        trades = np.abs(adjusted_action - self.positions)
        transaction_costs = trades.sum() * self.transaction_cost

        # Get returns
        period_returns = self.returns[self.current_step]

        # Calculate portfolio return
        portfolio_return_gross = np.dot(self.positions, period_returns)
        portfolio_return_net = portfolio_return_gross - transaction_costs

        # Update portfolio
        self.portfolio_value *= (1 + portfolio_return_net)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.positions = adjusted_action.copy()
        self.episode_returns.append(portfolio_return_net)

        # Calculate reward
        reward = self.reward_function(portfolio_return_net)

        # Advance steps (both meta and super env)
        self.current_step += 1
        self.super_env.advance_step()  # Encapsulated state synchronization

        # Check termination
        terminated = False
        truncated = False

        if self.current_step >= self.n_steps:
            truncated = True

        # Get next observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step >= self.n_steps:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Get super agent's blended portfolio output (not the blending weights!)
        # The super agent takes blend weights as action and outputs blended portfolio
        super_obs = self.super_env._get_observation()
        blend_weights, _ = self.super_agent.predict(super_obs, deterministic=True)

        # Normalize blend weights
        blend_weights = np.clip(blend_weights, 0, 1)
        if blend_weights.sum() > 0:
            blend_weights = blend_weights / blend_weights.sum()
        else:
            blend_weights = np.array([0.5, 0.5])

        # Get base agent actions
        tech_action = self.super_env.technical_agent.get_action(self.current_step)
        sent_action = self.super_env.sentiment_agent.get_action(self.current_step)

        # Blend to get portfolio
        blended_portfolio = blend_weights[0] * tech_action + blend_weights[1] * sent_action

        # Normalize portfolio
        if blended_portfolio.sum() > 0:
            blended_portfolio = blended_portfolio / blended_portfolio.sum()
        else:
            blended_portfolio = np.ones(self.n_assets) / self.n_assets

        # Get macro features for current step
        if len(self.macro_features.shape) == 3:
            # (n_steps, n_assets, n_features) - take mean across assets
            macro_feat = self.macro_features[self.current_step].mean(axis=0)
        else:
            # (n_steps, n_features)
            macro_feat = self.macro_features[self.current_step]

        # Concatenate: blended portfolio (n_assets) + macro features
        obs = np.concatenate([blended_portfolio, macro_feat])

        return obs.astype(np.float32)

    def _get_info(self) -> Dict:
        """Get info dictionary."""
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy()
        }

    def render(self):
        """Render (not implemented)."""
        pass


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_super_agent_env(
    technical_model_path: Path,
    sentiment_model_path: Path,
    split: str = 'train',
    reward_type: str = 'ema_sharpe',
    **kwargs
) -> SuperAgentEnv:
    """
    Create Super Agent Environment.

    Args:
        technical_model_path: Path to trained technical agent
        sentiment_model_path: Path to trained sentiment agent
        split: Data split
        reward_type: Reward function type
        **kwargs: Additional kwargs for SuperAgentEnv

    Returns:
        SuperAgentEnv instance
    """
    # Create base agent wrappers
    tech_agent = BaseAgentWrapper('technical', technical_model_path, split)
    sent_agent = BaseAgentWrapper('sentiment', sentiment_model_path, split)

    # Load returns
    returns_df = load_returns(split)
    returns = returns_df[TICKERS].fillna(0).values

    # Create environment
    env = SuperAgentEnv(
        technical_agent=tech_agent,
        sentiment_agent=sent_agent,
        returns=returns,
        reward_type=reward_type,
        **kwargs
    )

    return env


def create_meta_agent_env(
    super_agent_path: Path,
    super_env: SuperAgentEnv,
    split: str = 'train',
    reward_type: str = 'ema_sharpe',
    **kwargs
) -> MetaAgentEnv:
    """
    Create Meta Agent Environment.

    Args:
        super_agent_path: Path to trained super agent
        super_env: Super agent environment
        split: Data split
        reward_type: Reward function type
        **kwargs: Additional kwargs for MetaAgentEnv

    Returns:
        MetaAgentEnv instance
    """
    # Load super agent
    super_agent = PPO.load(super_agent_path)

    # Load macro features
    macro_df = load_features('macro', split)
    feature_cols = [col for col in macro_df.columns if col not in ['date', 'ticker']]

    # Get unique dates and aggregate macro features (they're replicated per ticker)
    dates = sorted(macro_df['date'].unique())
    macro_features = []
    for date in dates:
        date_data = macro_df[macro_df['date'] == date]
        # Take mean across tickers (should be identical)
        macro_features.append(date_data[feature_cols].mean().values)

    macro_features = np.array(macro_features)

    # Load returns
    returns_df = load_returns(split)
    returns = returns_df[TICKERS].fillna(0).values

    # Create environment
    env = MetaAgentEnv(
        super_agent=super_agent,
        super_env=super_env,
        macro_features=macro_features,
        returns=returns,
        reward_type=reward_type,
        **kwargs
    )

    return env


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    """Test hierarchical agent components."""

    print("Hierarchical Agent Module Test")
    print("="*70)
    print("\nNote: This requires trained base agents.")
    print("Run notebooks 02_base_agents.ipynb first.\n")

    # Check if models exist
    tech_model = MODELS_DIR / 'technical_ema_sharpe.zip'
    sent_model = MODELS_DIR / 'sentiment_ema_sharpe.zip'

    if tech_model.exists() and sent_model.exists():
        print("✅ Base agent models found!")

        # Test 1: Base Agent Wrapper
        print("\n1. Testing BaseAgentWrapper...")
        tech_wrapper = BaseAgentWrapper('technical', tech_model, 'train')
        print(f"   Technical agent loaded: {tech_wrapper.n_steps} steps")

        action = tech_wrapper.step()
        print(f"   Sample action: {action[:3]}... (sum={action.sum():.3f})")

        # Test 2: Super Agent Environment
        print("\n2. Testing SuperAgentEnv...")
        sent_wrapper = BaseAgentWrapper('sentiment', sent_model, 'train')
        returns_df = load_returns('train')
        returns = returns_df[TICKERS].fillna(0).values

        super_env = SuperAgentEnv(tech_wrapper, sent_wrapper, returns)
        print(f"   Super env created")
        print(f"   Action space: {super_env.action_space}")
        print(f"   Obs space: {super_env.observation_space}")

        obs, _ = super_env.reset()
        print(f"   Initial obs shape: {obs.shape}")

        action = super_env.action_space.sample()
        obs, reward, _, _, info = super_env.step(action)
        print(f"   Step executed, reward: {reward:.4f}")

        print("\n✅ All tests passed!")

    else:
        print("❌ Base agent models not found.")
        print(f"   Looking for:")
        print(f"   - {tech_model}")
        print(f"   - {sent_model}")

    print("\n" + "="*70)
