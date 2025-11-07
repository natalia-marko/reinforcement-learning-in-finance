"""
Reward functions for Multi-Hierarchical RL Portfolio System.

This module implements various reward functions for portfolio optimization:
1. EMA Sharpe Reward: Online Sharpe ratio using exponential moving averages
2. Multi-Objective Reward: Combines returns, risk, and drawdown
3. Simple Return Reward: Direct portfolio returns
4. Risk-Adjusted Return: Sharpe-like reward with lookback window
"""

import numpy as np
from typing import Dict, Optional, Tuple


class EMASharpeReward:
    """
    EMA-based Sharpe Ratio Reward.

    Computes an online Sharpe ratio using exponential moving averages
    for both returns and volatility. This provides a smooth, responsive
    reward signal suitable for RL training.

    Formula:
        reward = (ema_return - risk_free_rate) / (ema_std + epsilon)

    References:
        Moody & Saffell (2001): "Learning to Trade via Direct Reinforcement"
    """

    def __init__(
        self,
        alpha: float = 0.1,
        risk_free_rate: float = 0.04,
        target_sharpe: float = 2.0,
        annualization_factor: float = 52.0,
        epsilon: float = 1e-8,
        adaptive_target: bool = False,
        adaptive_alpha: float = 0.01
    ):
        """
        Initialize EMA Sharpe reward calculator.

        Args:
            alpha: EMA smoothing factor (0 < alpha <= 1)
                   Lower = smoother, Higher = more responsive
            risk_free_rate: Annual risk-free rate (e.g., 0.04 = 4%)
            target_sharpe: Target Sharpe ratio for normalization (initial value if adaptive)
            annualization_factor: Factor to annualize weekly returns (52 weeks/year)
            epsilon: Small constant to avoid division by zero
            adaptive_target: If True, target_sharpe adapts to observed Sharpe ratios
            adaptive_alpha: Smoothing factor for adaptive target (lower = more stable)
        """
        self.alpha = alpha
        self.risk_free_rate = risk_free_rate / annualization_factor  # Weekly rf rate
        self.initial_target_sharpe = target_sharpe
        self.target_sharpe = target_sharpe
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon
        self.adaptive_target = adaptive_target
        self.adaptive_alpha = adaptive_alpha

        # State variables
        self.ema_return = 0.0
        self.ema_return_sq = 0.0
        self.initialized = False
        self.step_count = 0

    def reset(self):
        """
        Reset state for new episode.

        Note: target_sharpe is NOT reset if adaptive_target=True,
        allowing it to carry learned normalization across episodes.
        """
        self.ema_return = 0.0
        self.ema_return_sq = 0.0
        self.initialized = False
        self.step_count = 0
        # Keep target_sharpe to maintain learned normalization

    def __call__(self, portfolio_return: float) -> float:
        """
        Calculate EMA Sharpe reward for current step.

        Args:
            portfolio_return: Portfolio return for current period (e.g., 0.02 = 2%)

        Returns:
            Sharpe-based reward
        """
        self.step_count += 1

        # Initialize on first call
        if not self.initialized:
            self.ema_return = portfolio_return
            self.ema_return_sq = portfolio_return ** 2
            self.initialized = True
            return 0.0  # No reward on first step

        # Update EMAs
        self.ema_return = (1 - self.alpha) * self.ema_return + self.alpha * portfolio_return
        self.ema_return_sq = (1 - self.alpha) * self.ema_return_sq + self.alpha * (portfolio_return ** 2)

        # Calculate EMA variance and std
        ema_variance = self.ema_return_sq - self.ema_return ** 2
        ema_std = np.sqrt(max(ema_variance, 0)) + self.epsilon

        # Calculate Sharpe ratio
        excess_return = self.ema_return - self.risk_free_rate
        sharpe = excess_return / ema_std

        # Annualize Sharpe
        sharpe_annual = sharpe * np.sqrt(self.annualization_factor)

        # Adaptive target: update target_sharpe based on observed Sharpe
        if self.adaptive_target and self.step_count > 10:  # Wait for stabilization
            # Smoothly adapt target toward observed Sharpe (clipped to reasonable range)
            observed_sharpe = np.clip(np.abs(sharpe_annual), 0.5, 5.0)
            self.target_sharpe = (
                (1 - self.adaptive_alpha) * self.target_sharpe +
                self.adaptive_alpha * observed_sharpe
            )
            # Keep target within reasonable bounds
            self.target_sharpe = np.clip(self.target_sharpe, 0.5, 4.0)

        # Normalize by target Sharpe (so reward ~1.0 at target)
        reward = sharpe_annual / (self.target_sharpe + self.epsilon)

        return reward

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics for logging."""
        if not self.initialized:
            return {
                'ema_return': 0.0,
                'ema_std': 0.0,
                'sharpe': 0.0,
                'target_sharpe': self.target_sharpe
            }

        ema_variance = self.ema_return_sq - self.ema_return ** 2
        ema_std = np.sqrt(max(ema_variance, 0))
        sharpe = (self.ema_return - self.risk_free_rate) / (ema_std + self.epsilon)
        sharpe_annual = sharpe * np.sqrt(self.annualization_factor)

        return {
            'ema_return': self.ema_return * self.annualization_factor,  # Annualized
            'ema_std': ema_std * np.sqrt(self.annualization_factor),
            'sharpe': sharpe_annual,
            'target_sharpe': self.target_sharpe,
            'reward_scaling': sharpe_annual / (self.target_sharpe + self.epsilon)
        }


class MultiObjectiveReward:
    """
    Multi-Objective Reward Function.

    Combines multiple objectives:
    1. Returns (maximize)
    2. Risk-adjusted returns (maximize Sharpe)
    3. Drawdown penalty (minimize drawdown)

    Formula:
        reward = w_return * return + w_risk * sharpe + w_dd * drawdown_penalty
    """

    def __init__(
        self,
        return_weight: float = 0.5,
        risk_weight: float = 0.3,
        drawdown_weight: float = 0.2,
        ema_alpha: float = 0.1,
        risk_free_rate: float = 0.04,
        annualization_factor: float = 52.0,
        reward_scale: float = 100.0
    ):
        """
        Initialize multi-objective reward calculator.

        Args:
            return_weight: Weight for raw returns
            risk_weight: Weight for Sharpe component
            drawdown_weight: Weight for drawdown penalty
            ema_alpha: EMA smoothing for Sharpe calculation
            risk_free_rate: Annual risk-free rate
            annualization_factor: Annualization factor (52 for weekly)
            reward_scale: Scale factor for final reward
        """
        self.return_weight = return_weight
        self.risk_weight = risk_weight
        self.drawdown_weight = drawdown_weight
        self.reward_scale = reward_scale

        # EMA Sharpe calculator
        self.sharpe_calculator = EMASharpeReward(
            alpha=ema_alpha,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor
        )

        # Drawdown tracking
        self.peak_value = 1.0  # Start with $1 (normalized)
        self.current_value = 1.0

    def reset(self):
        """Reset state for new episode."""
        self.sharpe_calculator.reset()
        self.peak_value = 1.0
        self.current_value = 1.0

    def __call__(self, portfolio_return: float) -> float:
        """
        Calculate multi-objective reward.

        Args:
            portfolio_return: Portfolio return for current period

        Returns:
            Multi-objective reward
        """
        # Update portfolio value
        self.current_value *= (1 + portfolio_return)
        self.peak_value = max(self.peak_value, self.current_value)

        # 1. Return component (scaled)
        return_reward = portfolio_return * 100  # Convert to %

        # 2. Sharpe component
        sharpe_reward = self.sharpe_calculator(portfolio_return)

        # 3. Drawdown penalty
        drawdown = (self.current_value - self.peak_value) / self.peak_value
        drawdown_penalty = drawdown * 100  # Negative value

        # Combine objectives
        reward = (
            self.return_weight * return_reward +
            self.risk_weight * sharpe_reward +
            self.drawdown_weight * drawdown_penalty
        )

        # Scale reward
        reward *= self.reward_scale

        return reward

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics for logging."""
        sharpe_metrics = self.sharpe_calculator.get_metrics()
        drawdown = (self.current_value - self.peak_value) / self.peak_value

        return {
            **sharpe_metrics,
            'drawdown': drawdown,
            'portfolio_value': self.current_value
        }


class SimpleReturnReward:
    """
    Simple Return-Based Reward.

    Directly uses portfolio returns as reward. Simple but can be noisy.
    """

    def __init__(self, scale: float = 100.0):
        """
        Initialize simple return reward.

        Args:
            scale: Scale factor for returns (e.g., 100 to convert to %)
        """
        self.scale = scale

    def reset(self):
        """Reset state (nothing to reset for simple returns)."""
        pass

    def __call__(self, portfolio_return: float) -> float:
        """
        Calculate simple return reward.

        Args:
            portfolio_return: Portfolio return for current period

        Returns:
            Scaled return
        """
        return portfolio_return * self.scale

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics (none for simple returns)."""
        return {}


class RiskAdjustedReturnReward:
    """
    Risk-Adjusted Return Reward with Lookback Window.

    Computes Sharpe ratio over a rolling window (e.g., 12 weeks).
    More stable than EMA Sharpe but less responsive.
    """

    def __init__(
        self,
        lookback_window: int = 12,
        risk_free_rate: float = 0.04,
        annualization_factor: float = 52.0,
        epsilon: float = 1e-8
    ):
        """
        Initialize risk-adjusted return reward.

        Args:
            lookback_window: Number of periods for rolling calculation
            risk_free_rate: Annual risk-free rate
            annualization_factor: Annualization factor
            epsilon: Small constant to avoid division by zero
        """
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate / annualization_factor
        self.annualization_factor = annualization_factor
        self.epsilon = epsilon

        # Return history
        self.return_history = []

    def reset(self):
        """Reset state for new episode."""
        self.return_history = []

    def __call__(self, portfolio_return: float) -> float:
        """
        Calculate risk-adjusted return reward.

        Args:
            portfolio_return: Portfolio return for current period

        Returns:
            Sharpe-based reward
        """
        # Add to history
        self.return_history.append(portfolio_return)

        # Keep only lookback window
        if len(self.return_history) > self.lookback_window:
            self.return_history.pop(0)

        # Need at least 2 periods for std calculation
        if len(self.return_history) < 2:
            return 0.0

        # Calculate Sharpe over window
        returns = np.array(self.return_history)
        mean_return = returns.mean()
        std_return = returns.std() + self.epsilon

        sharpe = (mean_return - self.risk_free_rate) / std_return
        sharpe_annual = sharpe * np.sqrt(self.annualization_factor)

        return sharpe_annual

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics for logging."""
        if len(self.return_history) < 2:
            return {
                'mean_return': 0.0,
                'std_return': 0.0,
                'sharpe': 0.0
            }

        returns = np.array(self.return_history)
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe = (mean_return - self.risk_free_rate) / (std_return + self.epsilon)
        sharpe_annual = sharpe * np.sqrt(self.annualization_factor)

        return {
            'mean_return': mean_return * self.annualization_factor,
            'std_return': std_return * np.sqrt(self.annualization_factor),
            'sharpe': sharpe_annual
        }


# ============================================================================
# REWARD FACTORY
# ============================================================================

def create_reward_function(
    reward_type: str = 'ema_sharpe',
    **kwargs
):
    """
    Factory function to create reward functions.

    Args:
        reward_type: Type of reward function
            - 'ema_sharpe': EMA Sharpe reward
            - 'multi_objective': Multi-objective reward
            - 'simple_return': Simple return reward
            - 'risk_adjusted': Risk-adjusted return with lookback
        **kwargs: Additional arguments passed to reward constructor

    Returns:
        Reward function instance
    """
    if reward_type == 'ema_sharpe':
        return EMASharpeReward(**kwargs)
    elif reward_type == 'multi_objective':
        return MultiObjectiveReward(**kwargs)
    elif reward_type == 'simple_return':
        return SimpleReturnReward(**kwargs)
    elif reward_type == 'risk_adjusted':
        return RiskAdjustedReturnReward(**kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    """Test reward functions with synthetic returns."""

    print("Testing Reward Functions")
    print("="*70)

    # Generate synthetic returns (trending upward with noise)
    np.random.seed(42)
    n_steps = 100
    returns = 0.01 + 0.02 * np.random.randn(n_steps)  # Mean 1%, std 2%

    # Test EMA Sharpe
    print("\n1. EMA Sharpe Reward:")
    ema_sharpe = EMASharpeReward(alpha=0.1, risk_free_rate=0.04)
    rewards = []
    for r in returns[:20]:
        reward = ema_sharpe(r)
        rewards.append(reward)
    print(f"   Sample rewards (first 20): {rewards[-5:]}")
    print(f"   Metrics: {ema_sharpe.get_metrics()}")

    # Test Multi-Objective
    print("\n2. Multi-Objective Reward:")
    multi_obj = MultiObjectiveReward()
    multi_obj.reset()
    rewards = []
    for r in returns[:20]:
        reward = multi_obj(r)
        rewards.append(reward)
    print(f"   Sample rewards (first 20): {[f'{r:.2f}' for r in rewards[-5:]]}")
    print(f"   Metrics: {multi_obj.get_metrics()}")

    # Test Simple Return
    print("\n3. Simple Return Reward:")
    simple = SimpleReturnReward(scale=100)
    rewards = [simple(r) for r in returns[:5]]
    print(f"   Sample rewards: {[f'{r:.2f}' for r in rewards]}")

    # Test Risk-Adjusted
    print("\n4. Risk-Adjusted Return Reward:")
    risk_adj = RiskAdjustedReturnReward(lookback_window=12)
    rewards = []
    for r in returns[:20]:
        reward = risk_adj(r)
        rewards.append(reward)
    print(f"   Sample rewards (first 20): {[f'{r:.2f}' for r in rewards[-5:]]}")
    print(f"   Metrics: {risk_adj.get_metrics()}")

    print("\n" + "="*70)
    print("✅ All reward functions tested successfully")
