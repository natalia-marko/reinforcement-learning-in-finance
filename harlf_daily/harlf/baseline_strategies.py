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
    ✅ FIXED: Simple momentum strategy based on 21-day returns.

    Allocates weights proportionally to positive momentum assets.
    If no assets have positive momentum, falls back to equal weight.

    This is a simple trend-following strategy commonly used in practice.

    Args:
        n_assets: Number of assets in portfolio (default: 7)
        n_features_per_asset: Number of features per asset (default: 16)
        return_feature_name: Name of return feature to use (default: 'return_21d')
    """

    def __init__(self, n_assets=7, n_features_per_asset=16, return_feature_name='return_21d'):
        self.n_assets = n_assets
        self.n_features_per_asset = n_features_per_asset

        # ✅ FIXED: Dynamically determine feature index from name
        # Feature order: return_5d(0), return_21d(1), price_to_sma_20d(2), ...
        feature_names = [
            'return_5d', 'return_21d', 'price_to_sma_20d', 'macd_histogram',
            'rsi_14d', 'bb_position', 'volatility_21d', 'atr_14d',
            'volume_ratio', 'mfi_14d', 'correlation_to_qqq', 'beta_to_qqq',
            'momentum_vol_interaction', 'regime_0', 'regime_1', 'regime_2'
        ]
        try:
            self.return_feature_idx = feature_names.index(return_feature_name)
        except ValueError:
            print(f"⚠️  Warning: Feature '{return_feature_name}' not found, using index 1")
            self.return_feature_idx = 1

    def predict(self, observation, deterministic=True):
        """
        Predict momentum-based portfolio weights.

        Args:
            observation: Flattened state [asset1_features, ..., asset7_features, current_weights]
            deterministic: Whether to use deterministic policy (ignored, always deterministic)

        Returns:
            weights: Portfolio weights based on positive momentum
            state: None (stateless strategy)
        """
        # ✅ FIXED: Extract features correctly, ignoring portfolio weights at end
        # Observation format: [asset_features (n_features_per_asset × n_assets), current_weights (n_assets)]
        feature_section_length = self.n_features_per_asset * self.n_assets

        # Extract returns for each asset
        returns = np.array([
            observation[i * self.n_features_per_asset + self.return_feature_idx]
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
    ✅ FIXED: Contrarian strategy: allocate to assets with negative momentum.

    This is a mean-reversion strategy that bets on reversal of recent trends.

    Args:
        n_assets: Number of assets in portfolio (default: 7)
        n_features_per_asset: Number of features per asset (default: 16)
        return_feature_name: Name of return feature to use (default: 'return_21d')
    """

    def __init__(self, n_assets=7, n_features_per_asset=16, return_feature_name='return_21d'):
        self.n_assets = n_assets
        self.n_features_per_asset = n_features_per_asset

        # ✅ FIXED: Dynamically determine feature index
        feature_names = [
            'return_5d', 'return_21d', 'price_to_sma_20d', 'macd_histogram',
            'rsi_14d', 'bb_position', 'volatility_21d', 'atr_14d',
            'volume_ratio', 'mfi_14d', 'correlation_to_qqq', 'beta_to_qqq',
            'momentum_vol_interaction', 'regime_0', 'regime_1', 'regime_2'
        ]
        try:
            self.return_feature_idx = feature_names.index(return_feature_name)
        except ValueError:
            print(f"⚠️  Warning: Feature '{return_feature_name}' not found, using index 1")
            self.return_feature_idx = 1

    def predict(self, observation, deterministic=True):
        """Predict contrarian portfolio weights."""
        # ✅ FIXED: Extract features correctly, ignoring portfolio weights at end
        returns = np.array([
            observation[i * self.n_features_per_asset + self.return_feature_idx]
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
    ✅ FIXED: Allocate to low-volatility assets.

    Inversely weights assets by their volatility - favors stable assets.

    Args:
        n_assets: Number of assets in portfolio (default: 7)
        n_features_per_asset: Number of features per asset (default: 16)
        volatility_feature_name: Name of volatility feature (default: 'volatility_21d')
    """

    def __init__(self, n_assets=7, n_features_per_asset=16, volatility_feature_name='volatility_21d'):
        self.n_assets = n_assets
        self.n_features_per_asset = n_features_per_asset

        # ✅ FIXED: Dynamically determine feature index
        feature_names = [
            'return_5d', 'return_21d', 'price_to_sma_20d', 'macd_histogram',
            'rsi_14d', 'bb_position', 'volatility_21d', 'atr_14d',
            'volume_ratio', 'mfi_14d', 'correlation_to_qqq', 'beta_to_qqq',
            'momentum_vol_interaction', 'regime_0', 'regime_1', 'regime_2'
        ]
        try:
            self.volatility_feature_idx = feature_names.index(volatility_feature_name)
        except ValueError:
            print(f"⚠️  Warning: Feature '{volatility_feature_name}' not found, using index 6")
            self.volatility_feature_idx = 6

    def predict(self, observation, deterministic=True):
        """Predict low-volatility portfolio weights."""
        # ✅ FIXED: Extract features correctly, ignoring portfolio weights at end
        volatilities = np.array([
            observation[i * self.n_features_per_asset + self.volatility_feature_idx]
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
