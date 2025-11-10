"""
Baseline Portfolio Strategies for Comparison

Simple strategies to benchmark against RL agents:
1. Equal Weight (1/N)
2. Random Allocation
3. Momentum (21-day)

Created: 2025-01-09
"""

import numpy as np


class EqualWeightStrategy:
    """
    Equal weight (1/N) portfolio strategy.

    Allocates equal weight to all assets regardless of market conditions.
    This is a common baseline in portfolio optimization.

    Args:
        n_assets: Number of assets in portfolio (default: 7)
    """

    def __init__(self, n_assets=7):
        self.n_assets = n_assets
        self.weights = np.ones(n_assets, dtype=np.float32) / n_assets

    def predict(self, observation, deterministic=True):
        """
        Predict portfolio weights.

        Args:
            observation: Current state (ignored)
            deterministic: Whether to use deterministic policy (ignored, always deterministic)

        Returns:
            weights: Portfolio weights (all equal)
            state: None (stateless strategy)
        """
        return self.weights.copy(), None


class RandomStrategy:
    """
    Random portfolio allocation strategy.

    Samples random weights from a Dirichlet distribution.
    Useful as a baseline to verify that other strategies do better than random.

    Args:
        n_assets: Number of assets in portfolio (default: 7)
        seed: Random seed for reproducibility (default: 42)
    """

    def __init__(self, n_assets=7, seed=42):
        self.n_assets = n_assets
        self.rng = np.random.RandomState(seed)

    def predict(self, observation, deterministic=True):
        """
        Predict random portfolio weights.

        Args:
            observation: Current state (ignored)
            deterministic: Whether to use deterministic policy (ignored)

        Returns:
            weights: Random portfolio weights (sum to 1.0)
            state: None (stateless strategy)
        """
        weights = self.rng.dirichlet(np.ones(self.n_assets)).astype(np.float32)
        return weights, None


class MomentumStrategy:
    """
    Simple momentum strategy based on 21-day returns.

    Allocates weights proportionally to positive momentum assets.
    If no assets have positive momentum, falls back to equal weight.

    This is a simple trend-following strategy commonly used in practice.

    Args:
        n_assets: Number of assets in portfolio (default: 7)
        return_feature_idx: Index of return feature in observation (default: 1 for return_21d)
    """

    def __init__(self, n_assets=7, return_feature_idx=1):
        self.n_assets = n_assets
        self.return_feature_idx = return_feature_idx

    def predict(self, observation, deterministic=True):
        """
        Predict momentum-based portfolio weights.

        Args:
            observation: Flattened state [asset1_features, asset2_features, ...]
            deterministic: Whether to use deterministic policy (ignored, always deterministic)

        Returns:
            weights: Portfolio weights based on positive momentum
            state: None (stateless strategy)
        """
        # Extract return_21d for each asset
        # Observation format: [asset1_feat1, ..., asset1_featN, asset2_feat1, ...]
        n_features = len(observation) // self.n_assets
        returns = np.array([
            observation[i * n_features + self.return_feature_idx]
            for i in range(self.n_assets)
        ])

        # Allocate to positive momentum assets
        positive_returns = np.maximum(returns, 0)

        if positive_returns.sum() > 0:
            weights = positive_returns / positive_returns.sum()
        else:
            # No positive momentum - use equal weight
            weights = np.ones(self.n_assets) / self.n_assets

        return weights.astype(np.float32), None


class InverseMomentumStrategy:
    """
    Contrarian strategy: allocate to assets with negative momentum.

    This is a mean-reversion strategy that bets on reversal of recent trends.

    Args:
        n_assets: Number of assets in portfolio (default: 7)
        return_feature_idx: Index of return feature in observation (default: 1)
    """

    def __init__(self, n_assets=7, return_feature_idx=1):
        self.n_assets = n_assets
        self.return_feature_idx = return_feature_idx

    def predict(self, observation, deterministic=True):
        """Predict contrarian portfolio weights."""
        n_features = len(observation) // self.n_assets
        returns = np.array([
            observation[i * n_features + self.return_feature_idx]
            for i in range(self.n_assets)
        ])

        # Allocate to negative momentum assets (mean reversion)
        negative_returns = np.maximum(-returns, 0)

        if negative_returns.sum() > 0:
            weights = negative_returns / negative_returns.sum()
        else:
            weights = np.ones(self.n_assets) / self.n_assets

        return weights.astype(np.float32), None


class LowVolatilityStrategy:
    """
    Allocate to low-volatility assets.

    Inversely weights assets by their volatility - favors stable assets.

    Args:
        n_assets: Number of assets in portfolio (default: 7)
        volatility_feature_idx: Index of volatility feature (default: 6 for volatility_21d)
    """

    def __init__(self, n_assets=7, volatility_feature_idx=6):
        self.n_assets = n_assets
        self.volatility_feature_idx = volatility_feature_idx

    def predict(self, observation, deterministic=True):
        """Predict low-volatility portfolio weights."""
        n_features = len(observation) // self.n_assets
        volatilities = np.array([
            observation[i * n_features + self.volatility_feature_idx]
            for i in range(self.n_assets)
        ])

        # Inverse volatility weights (lower vol = higher weight)
        # Add small constant to avoid division by zero
        inv_vol = 1.0 / (np.abs(volatilities) + 1e-6)
        weights = inv_vol / inv_vol.sum()

        return weights.astype(np.float32), None


# ============================================================================
# TESTING
# ============================================================================

def test_baseline_strategies():
    """Test all baseline strategies."""
    print("="*80)
    print("TESTING BASELINE STRATEGIES")
    print("="*80)

    # Create dummy observation (22 features × 7 assets = 154)
    n_features = 22
    n_assets = 7
    obs = np.random.randn(n_features * n_assets).astype(np.float32)

    strategies = {
        'Equal Weight': EqualWeightStrategy(n_assets),
        'Random': RandomStrategy(n_assets, seed=42),
        'Momentum': MomentumStrategy(n_assets),
        'Inverse Momentum': InverseMomentumStrategy(n_assets),
        'Low Volatility': LowVolatilityStrategy(n_assets),
    }

    print(f"\nTest observation shape: {obs.shape}")
    print(f"Number of assets: {n_assets}\n")

    for name, strategy in strategies.items():
        print(f"Testing {name}...")

        # Test predict
        weights, state = strategy.predict(obs, deterministic=True)

        # Validate
        assert len(weights) == n_assets, f"Wrong number of weights: {len(weights)}"
        assert np.isclose(weights.sum(), 1.0, atol=1e-5), f"Weights don't sum to 1: {weights.sum()}"
        assert np.all(weights >= 0), f"Negative weights found"
        assert np.all(weights <= 1), f"Weights > 1 found"
        assert not np.isnan(weights).any(), f"NaN weights found"

        print(f"  ✓ Weights: {weights}")
        print(f"  ✓ Sum: {weights.sum():.6f}")
        print(f"  ✓ Min/Max: {weights.min():.4f} / {weights.max():.4f}")
        print()

    print("="*80)
    print("✅ ALL BASELINE STRATEGY TESTS PASSED")
    print("="*80)


if __name__ == '__main__':
    test_baseline_strategies()
