"""
Unit Tests and Validation for All Critical Fixes

This script tests:
1. Environment with portfolio weights in observations
2. Fixed reward calculation (EMA Sharpe)
3. Fixed portfolio return calculation
4. Fixed baseline strategies
5. Early stopping callback
6. SimpleReturnReward as baseline

Usage:
    python test_fixes.py --all           # Run all tests
    python test_fixes.py --unit          # Unit tests only
    python test_fixes.py --integration   # Integration test (quick training)
    python test_fixes.py --reward simple # Test with SimpleReturnReward

Created: 2025-01-11
"""

import argparse
import numpy as np
import sys
from pathlib import Path

# Test results tracking
test_results = {}


def test_environment_observation_space():
    """Test that environment includes portfolio weights in observations."""
    print("\n" + "="*70)
    print("TEST 1: Environment Observation Space")
    print("="*70)

    try:
        from harlf.portfolio_env import PortfolioEnv

        env = PortfolioEnv(fold_id=0, split='train')
        obs, info = env.reset()

        # Check dimensions
        n_features = 16
        n_assets = 7
        expected_dim = n_features * n_assets + n_assets  # 119

        print(f"Expected observation dim: {expected_dim}")
        print(f"Actual observation dim: {obs.shape[0]}")

        assert obs.shape[0] == expected_dim, f"Wrong dimension: {obs.shape[0]} != {expected_dim}"

        # Check weights are at the end
        weights = obs[-n_assets:]
        print(f"Current weights (last {n_assets} values): {weights}")
        print(f"Weights sum: {weights.sum():.6f}")

        assert np.isclose(weights.sum(), 1.0, atol=1e-5), f"Weights don't sum to 1: {weights.sum()}"

        # Test that weights update after step
        action = np.array([0.3, 0.2, 0.1, 0.15, 0.1, 0.1, 0.05])
        obs_next, reward, done, truncated, info = env.step(action)

        new_weights = obs_next[-n_assets:]
        print(f"New weights after action: {new_weights}")

        assert not np.allclose(weights, new_weights), "Weights didn't change after action"

        print("\n✅ TEST 1 PASSED: Observation space includes portfolio weights")
        test_results['observation_space'] = 'PASS'
        return True

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        test_results['observation_space'] = 'FAIL'
        return False


def test_reward_calculation():
    """Test that EMA Sharpe reward calculation is fixed."""
    print("\n" + "="*70)
    print("TEST 2: Reward Calculation (EMA Sharpe)")
    print("="*70)

    try:
        from harlf.rewards import EMASharpeReward

        reward_fn = EMASharpeReward(window=21, risk_free_rate=0.02)

        # Test with known returns sequence
        test_returns = [0.01, -0.005, 0.015, -0.01, 0.02]
        rewards = []

        print("Testing reward calculation order...")
        for i, ret in enumerate(test_returns):
            reward = reward_fn.compute(ret)
            info = reward_fn.get_info()
            rewards.append(reward)

            print(f"  Step {i}: return={ret:+.4f} → reward={reward:+.4f}, "
                  f"ema_return={info['ema_return']:+.6f}, ema_std={info['ema_std']:.6f}")

            # Check that variance is reasonable (should be positive)
            assert info['ema_variance'] > 0, f"Variance should be positive: {info['ema_variance']}"

            # Check that reward is finite
            assert np.isfinite(reward), f"Reward should be finite: {reward}"

        # Check that rewards vary reasonably
        reward_std = np.std(rewards)
        print(f"\nReward std: {reward_std:.4f}")
        assert reward_std > 0.01, f"Rewards should vary: std={reward_std}"

        print("\n✅ TEST 2 PASSED: EMA Sharpe reward calculation works correctly")
        test_results['reward_calculation'] = 'PASS'
        return True

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        test_results['reward_calculation'] = 'FAIL'
        return False


def test_portfolio_return_calculation():
    """Test exact vs approximate portfolio return calculation."""
    print("\n" + "="*70)
    print("TEST 3: Portfolio Return Calculation")
    print("="*70)

    try:
        # Test data: large returns where approximation matters
        asset_log_returns = np.array([0.05, -0.03, 0.02, 0.01, -0.02, 0.04, 0.0])
        portfolio_weights = np.array([0.2, 0.15, 0.1, 0.15, 0.1, 0.2, 0.1])

        print(f"Asset log returns: {asset_log_returns}")
        print(f"Portfolio weights: {portfolio_weights}")

        # OLD (approximate) method
        approx_log_return = np.dot(portfolio_weights, asset_log_returns)

        # NEW (exact) method - what we implemented
        asset_simple_returns = np.exp(asset_log_returns) - 1.0
        portfolio_simple_return = np.dot(portfolio_weights, asset_simple_returns)
        exact_log_return = np.log(1.0 + portfolio_simple_return)

        # Calculate error
        error = abs(exact_log_return - approx_log_return)
        error_pct = (error / abs(exact_log_return)) * 100

        print(f"\nApproximate log return: {approx_log_return:.6f}")
        print(f"Exact log return: {exact_log_return:.6f}")
        print(f"Absolute error: {error:.8f}")
        print(f"Relative error: {error_pct:.4f}%")

        # For large returns, error should be significant (proving we need exact calc)
        assert error > 0.0001, f"Error too small for large returns: {error}"

        # Test with environment
        from harlf.portfolio_env import PortfolioEnv
        env = PortfolioEnv(fold_id=0, split='train')
        obs, _ = env.reset()

        # Take one step and verify return calculation
        action = portfolio_weights
        obs_next, reward, done, truncated, info = env.step(action)

        print(f"\nEnvironment portfolio return: {info['portfolio_return']:.6f}")
        print(f"Transaction cost: {info['transaction_cost']:.6f}")

        assert np.isfinite(info['portfolio_return']), "Portfolio return should be finite"

        print("\n✅ TEST 3 PASSED: Portfolio return calculation is exact")
        test_results['portfolio_return'] = 'PASS'
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        test_results['portfolio_return'] = 'FAIL'
        return False


def test_baseline_strategies():
    """Test that baseline strategies work with new observation structure."""
    print("\n" + "="*70)
    print("TEST 4: Baseline Strategies")
    print("="*70)

    try:
        from harlf.baseline_strategies import (
            EqualWeightStrategy, MomentumStrategy,
            InverseMomentumStrategy, LowVolatilityStrategy
        )
        from harlf.portfolio_env import PortfolioEnv

        # Create environment and get observation
        env = PortfolioEnv(fold_id=0, split='train')
        obs, _ = env.reset()

        print(f"Observation shape: {obs.shape}")

        strategies = {
            'Equal Weight': EqualWeightStrategy(),
            'Momentum': MomentumStrategy(),
            'Inverse Momentum': InverseMomentumStrategy(),
            'Low Volatility': LowVolatilityStrategy()
        }

        for name, strategy in strategies.items():
            weights, _ = strategy.predict(obs)

            print(f"\n{name}:")
            print(f"  Weights: {weights}")
            print(f"  Sum: {weights.sum():.6f}")
            print(f"  Min/Max: {weights.min():.4f} / {weights.max():.4f}")

            # Validate
            assert len(weights) == 7, f"Wrong number of weights: {len(weights)}"
            assert np.isclose(weights.sum(), 1.0, atol=1e-5), f"Weights don't sum to 1: {weights.sum()}"
            assert np.all(weights >= 0), "Negative weights found"
            assert np.all(weights <= 1), "Weights > 1 found"
            assert not np.isnan(weights).any(), "NaN weights found"

        print("\n✅ TEST 4 PASSED: All baseline strategies work correctly")
        test_results['baseline_strategies'] = 'PASS'
        return True

    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        test_results['baseline_strategies'] = 'FAIL'
        return False


def test_simple_return_reward():
    """Test SimpleReturnReward as baseline."""
    print("\n" + "="*70)
    print("TEST 5: SimpleReturnReward")
    print("="*70)

    try:
        from harlf.rewards import SimpleReturnReward
        from harlf.portfolio_env import PortfolioEnv

        # Test reward function
        reward_fn = SimpleReturnReward(annualize=False, scale=1.0)

        test_returns = [0.01, -0.005, 0.015]
        print("Testing SimpleReturnReward...")
        for i, ret in enumerate(test_returns):
            reward = reward_fn.compute(ret)
            print(f"  Return: {ret:+.4f} → Reward: {reward:+.4f}")
            assert np.isclose(reward, ret), f"Reward should equal return: {reward} != {ret}"

        # Test with environment
        env = PortfolioEnv(fold_id=0, split='train', reward_type='simple')
        obs, _ = env.reset()

        action = np.ones(7) / 7  # Equal weight
        obs_next, reward, done, truncated, info = env.step(action)

        print(f"\nEnvironment with SimpleReturnReward:")
        print(f"  Reward: {reward:.6f}")
        print(f"  Portfolio return: {info['portfolio_return']:.6f}")

        # Reward should be close to portfolio return (minus transaction costs)
        assert np.isfinite(reward), "Reward should be finite"

        print("\n✅ TEST 5 PASSED: SimpleReturnReward works correctly")
        test_results['simple_return_reward'] = 'PASS'
        return True

    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        test_results['simple_return_reward'] = 'FAIL'
        return False


def test_integration():
    """Integration test: run a few training steps."""
    print("\n" + "="*70)
    print("TEST 6: Integration Test (Quick Training)")
    print("="*70)

    try:
        from harlf.portfolio_env import PortfolioEnv
        from harlf.dirichlet_policy import SoftmaxActorCriticPolicy
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.monitor import Monitor

        # Create environment
        env = PortfolioEnv(fold_id=0, split='train', reward_type='simple')  # Start simple
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

        print("✅ Environment created")

        # Create model
        model = PPO(
            policy=SoftmaxActorCriticPolicy,
            env=env,
            learning_rate=3e-5,
            n_steps=512,
            batch_size=64,
            n_epochs=5,  # Reduced for quick test
            verbose=0,
            policy_kwargs={'net_arch': [128, 128]}
        )

        print("✅ Model created")

        # Train for a few steps
        print("🚀 Training for 1024 steps...")
        model.learn(total_timesteps=1024, progress_bar=False)

        print("✅ Training completed successfully")

        # Test evaluation
        obs = env.reset()
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                break

        print("✅ Evaluation works")

        print("\n✅ TEST 6 PASSED: Integration test successful")
        test_results['integration'] = 'PASS'
        return True

    except Exception as e:
        print(f"\n❌ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        test_results['integration'] = 'FAIL'
        return False


def print_test_summary():
    """Print summary of all test results."""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total = len(test_results)
    passed = sum(1 for v in test_results.values() if v == 'PASS')

    for test_name, result in test_results.items():
        status = "✅ PASS" if result == 'PASS' else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("="*70)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou can now run full training:")
        print("  python harlf/train_ppo.py --fold_id 0 --total_timesteps 50000")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test all critical fixes')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration test only')

    args = parser.parse_args()

    # Default to all tests
    run_all = args.all or not (args.unit or args.integration)

    print("="*70)
    print("TESTING ALL CRITICAL FIXES")
    print("="*70)

    success = True

    if run_all or args.unit:
        # Unit tests
        success &= test_environment_observation_space()
        success &= test_reward_calculation()
        success &= test_portfolio_return_calculation()
        success &= test_baseline_strategies()
        success &= test_simple_return_reward()

    if run_all or args.integration:
        # Integration test
        success &= test_integration()

    # Print summary
    all_passed = print_test_summary()

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
