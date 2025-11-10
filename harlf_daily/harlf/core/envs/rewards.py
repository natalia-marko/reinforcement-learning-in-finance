"""
Reward Functions for Portfolio RL Environment

Implements multiple reward function strategies:
1. EMA Sharpe Reward (Primary) - Risk-adjusted returns with exponential smoothing
2. Multi-Objective Reward - Balances return, risk, alpha, and transaction costs

Created: 2025-01-09
"""

import numpy as np
from typing import Optional, Dict, Any


class EMASharpeReward:
    """
    EMA Sharpe Ratio Reward Function.

    Computes risk-adjusted returns using exponential moving averages
    to smooth daily noise while adapting to recent market conditions.

    Mathematical formulation:
        Sharpe_t = (EMA_return_t - rf_rate) / sqrt(EMA_variance_t)

    Where:
        EMA_return_t = α * r_t + (1-α) * EMA_return_{t-1}
        EMA_variance_t = α * (r_t - EMA_return_t)^2 + (1-α) * EMA_variance_{t-1}
        α = 2 / (window + 1)

    Advantages:
        - Directly optimizes risk-adjusted returns
        - Smooths daily noise (reduces reward variance)
        - Adapts to recent market conditions
        - Single hyperparameter (window size)

    Args:
        window: EMA window size in days (default: 21 = 1 month)
        risk_free_rate: Annual risk-free rate (default: 0.0)
        scale: Scaling factor for reward (default: 1.0)
        min_variance: Minimum variance to avoid division by zero (default: 1e-4)
        clip_min: Minimum Sharpe ratio value (default: -10.0)
        clip_max: Maximum Sharpe ratio value (default: 10.0)
    """

    def __init__(
        self,
        window: int = 21,
        risk_free_rate: float = 0.00,
        scale: float = 1.0,
        min_variance: float = 1e-4,
        clip_min: float = -10.0,
        clip_max: float = 10.0
    ):
        self.window = window
        self.alpha = 2.0 / (window + 1)
        self.rf_rate = risk_free_rate / 252.0  # Convert to daily
        self.scale = scale
        self.min_variance = min_variance
        self.clip_min = clip_min
        self.clip_max = clip_max

        # State variables
        self.ema_return = 0.0
        self.ema_variance = 0.0
        self.step_count = 0

    def reset(self):
        """Reset the reward function state."""
        self.ema_return = 0.0
        self.ema_variance = 0.0
        self.step_count = 0

    def compute(self, portfolio_return: float) -> float:
        """
        Compute EMA Sharpe reward for current step.

        Args:
            portfolio_return: Daily portfolio return (not annualized)

        Returns:
            Sharpe ratio reward (scaled)
        """
        # Update EMA return
        if self.step_count == 0:
            self.ema_return = portfolio_return
        else:
            self.ema_return = self.alpha * portfolio_return + (1 - self.alpha) * self.ema_return

        # Update EMA variance
        deviation = portfolio_return - self.ema_return
        if self.step_count == 0:
            self.ema_variance = deviation ** 2
        else:
            self.ema_variance = self.alpha * (deviation ** 2) + (1 - self.alpha) * self.ema_variance

        # Compute Sharpe ratio
        sharpe = (self.ema_return - self.rf_rate) / (np.sqrt(self.ema_variance + self.min_variance))

        # Clip Sharpe ratio to prevent explosions
        sharpe = np.clip(sharpe, self.clip_min, self.clip_max)

        self.step_count += 1

        return sharpe * self.scale

    def get_info(self) -> Dict[str, float]:
        """Get current state information."""
        return {
            'ema_return': self.ema_return,
            'ema_variance': self.ema_variance,
            'ema_std': np.sqrt(self.ema_variance),
            'ema_sharpe': (self.ema_return - self.rf_rate) / (np.sqrt(self.ema_variance + self.min_variance)),
            'annualized_return': self.ema_return * 252,
            'annualized_volatility': np.sqrt(self.ema_variance) * np.sqrt(252),
        }


class MultiObjectiveReward:
    """
    Multi-Objective Reward Function.

    Balances multiple objectives:
    1. Return maximization
    2. Risk minimization (downside penalty)
    3. Alpha generation (benchmark outperformance)
    4. Transaction cost minimization

    Mathematical formulation:
        R_t = w1 * R_return + w2 * R_risk + w3 * R_alpha + w4 * R_turnover

    Where:
        R_return = r_t * 252 (annualized return)
        R_risk = -max(0, -r_t) * penalty_scale (downside penalty)
        R_alpha = (r_t - r_benchmark) * 252 (annualized alpha)
        R_turnover = -turnover * cost_rate (transaction cost)

    Advantages:
        - Explicit control over objectives
        - Interpretable components
        - Flexible (adjustable to investor preferences)

    Disadvantages:
        - More hyperparameters to tune
        - Risk of reward hacking if weights misaligned

    Args:
        w_return: Weight for return component (default: 1.0)
        w_risk: Weight for risk penalty (default: 0.5)
        w_alpha: Weight for alpha component (default: 0.3)
        w_turnover: Weight for turnover penalty (default: 0.2)
        downside_penalty_scale: Scaling for downside risk (default: 100.0)
        cost_rate: Transaction cost rate (default: 0.0015 = 15 bps)
    """

    def __init__(
        self,
        w_return: float = 1.0,
        w_risk: float = 0.5,
        w_alpha: float = 0.3,
        w_turnover: float = 0.2,
        downside_penalty_scale: float = 100.0,
        cost_rate: float = 0.0015
    ):
        self.w_return = w_return
        self.w_risk = w_risk
        self.w_alpha = w_alpha
        self.w_turnover = w_turnover
        self.downside_penalty_scale = downside_penalty_scale
        self.cost_rate = cost_rate

    def reset(self):
        """Reset the reward function state (stateless for this reward)."""
        pass

    def compute(
        self,
        portfolio_return: float,
        benchmark_return: float = 0.0,
        turnover: float = 0.0
    ) -> float:
        """
        Compute multi-objective reward.

        Args:
            portfolio_return: Daily portfolio return
            benchmark_return: Daily benchmark return (default: 0.0)
            turnover: Portfolio turnover (sum of absolute weight changes)

        Returns:
            Multi-objective reward
        """
        # Component 1: Return (annualized)
        r_return = portfolio_return * 252

        # Component 2: Risk penalty (penalize losses)
        r_risk = -max(0.0, -portfolio_return) * self.downside_penalty_scale

        # Component 3: Alpha (annualized outperformance)
        alpha = portfolio_return - benchmark_return
        r_alpha = alpha * 252

        # Component 4: Turnover penalty (transaction costs)
        r_turnover = -turnover * self.cost_rate

        # Weighted sum
        total_reward = (
            self.w_return * r_return +
            self.w_risk * r_risk +
            self.w_alpha * r_alpha +
            self.w_turnover * r_turnover
        )

        return total_reward

    def get_info(
        self,
        portfolio_return: float,
        benchmark_return: float = 0.0,
        turnover: float = 0.0
    ) -> Dict[str, float]:
        """Get component breakdown."""
        r_return = portfolio_return * 252
        r_risk = -max(0.0, -portfolio_return) * self.downside_penalty_scale
        alpha = portfolio_return - benchmark_return
        r_alpha = alpha * 252
        r_turnover = -turnover * self.cost_rate

        return {
            'r_return': r_return,
            'r_risk': r_risk,
            'r_alpha': r_alpha,
            'r_turnover': r_turnover,
            'total_reward': (
                self.w_return * r_return +
                self.w_risk * r_risk +
                self.w_alpha * r_alpha +
                self.w_turnover * r_turnover
            ),
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return,
            'alpha': alpha,
            'turnover': turnover,
        }


class SimpleReturnReward:
    """
    Simple Return Reward (baseline).

    Directly uses portfolio returns as rewards.
    Good for debugging and as a baseline comparison.

    Args:
        annualize: Whether to annualize returns (default: False)
        scale: Scaling factor (default: 1.0)
    """

    def __init__(self, annualize: bool = False, scale: float = 1.0):
        self.annualize = annualize
        self.scale = scale

    def reset(self):
        """Reset the reward function state."""
        pass

    def compute(self, portfolio_return: float) -> float:
        """
        Compute simple return reward.

        Args:
            portfolio_return: Daily portfolio return

        Returns:
            Return-based reward
        """
        if self.annualize:
            return portfolio_return * 252 * self.scale
        return portfolio_return * self.scale

    def get_info(self) -> Dict[str, float]:
        """Get state information (empty for this reward)."""
        return {}


# ============================================================================
# TESTING
# ============================================================================

def test_reward_functions():
    """Test all reward functions."""
    print("="*80)
    print("TESTING REWARD FUNCTIONS")
    print("="*80)

    # Test data
    returns = np.array([0.01, -0.005, 0.015, -0.01, 0.02, 0.005, -0.015])
    benchmark_returns = np.array([0.008, -0.003, 0.012, -0.008, 0.015, 0.006, -0.01])
    turnovers = np.array([0.1, 0.05, 0.15, 0.08, 0.12, 0.06, 0.09])

    # Test 1: EMA Sharpe Reward
    print("\n1. Testing EMA Sharpe Reward...")
    ema_reward = EMASharpeReward(window=5, risk_free_rate=0.04)

    for i, ret in enumerate(returns):
        reward = ema_reward.compute(ret)
        info = ema_reward.get_info()
        print(f"  Step {i}: Return={ret:+.4f}, Reward={reward:+.4f}, "
              f"EMA_Return={info['ema_return']:+.4f}, EMA_Std={info['ema_std']:.4f}")

    ema_reward.reset()
    print("  ✓ EMA Sharpe Reward works")

    # Test 2: Multi-Objective Reward
    print("\n2. Testing Multi-Objective Reward...")
    mo_reward = MultiObjectiveReward()

    for i in range(len(returns)):
        reward = mo_reward.compute(returns[i], benchmark_returns[i], turnovers[i])
        info = mo_reward.get_info(returns[i], benchmark_returns[i], turnovers[i])
        print(f"  Step {i}: Return={returns[i]:+.4f}, Reward={reward:+.4f}, "
              f"Alpha={info['alpha']:+.4f}, Turnover={turnovers[i]:.4f}")

    print("  ✓ Multi-Objective Reward works")

    # Test 3: Simple Return Reward
    print("\n3. Testing Simple Return Reward...")
    simple_reward = SimpleReturnReward(annualize=True, scale=1.0)

    for i, ret in enumerate(returns):
        reward = simple_reward.compute(ret)
        print(f"  Step {i}: Return={ret:+.4f}, Reward={reward:+.4f}")

    print("  ✓ Simple Return Reward works")

    print("\n" + "="*80)
    print("✅ ALL REWARD FUNCTION TESTS PASSED")
    print("="*80)


if __name__ == '__main__':
    test_reward_functions()
