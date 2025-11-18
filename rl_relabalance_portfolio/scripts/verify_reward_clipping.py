"""
Verify that reward clipping is working correctly.
This script tests individual step rewards to confirm they're clipped to [-10, 10].
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from rl_system import PortfolioEnv

def test_reward_clipping():
    """Test that step rewards are clipped to [-10, 10]."""

    print("=" * 80)
    print("REWARD CLIPPING VERIFICATION")
    print("=" * 80)

    # Load data
    DATA_DIR = project_root / 'data' / 'processed'
    train_df = pd.read_parquet(DATA_DIR / 'train.parquet')

    with open(DATA_DIR / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    feature_cols = metadata['feature_cols']
    tickers = metadata['tickers']

    # Create environment
    print("\n1. Creating environment...")
    env = PortfolioEnv(
        data=train_df.head(1000),  # Use subset for quick test
        feature_cols=feature_cols,
        tickers=tickers,
        rebalance_frequency=4,
        transaction_cost=0.00025,
        max_weight_per_asset=0.4,
        reward_lookback=4,
        seed=42
    )

    # Run episode and collect rewards
    print("2. Running episode and collecting step rewards...")
    obs, info = env.reset()

    step_rewards = []
    episode_reward = 0
    done = False
    step = 0

    while not done and step < 50:  # Limit to 50 steps for quick test
        # Random action (just for testing)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        step_rewards.append(reward)
        episode_reward += reward
        done = terminated or truncated
        step += 1

    # Analyze rewards
    print(f"\n3. Analyzing {len(step_rewards)} step rewards:")
    print(f"   Min step reward: {min(step_rewards):.4f}")
    print(f"   Max step reward: {max(step_rewards):.4f}")
    print(f"   Mean step reward: {np.mean(step_rewards):.4f}")
    print(f"   Std step reward: {np.std(step_rewards):.4f}")

    # Check clipping
    print("\n4. Verifying clipping to [-10, 10]:")
    clipped = all(-10.0 <= r <= 10.0 for r in step_rewards)

    if clipped:
        print("   ✅ ALL step rewards are within [-10, 10]")
        print("   ✅ Clipping is working correctly!")
    else:
        out_of_range = [r for r in step_rewards if r < -10 or r > 10]
        print(f"   ❌ Found {len(out_of_range)} rewards outside [-10, 10]:")
        for r in out_of_range[:5]:  # Show first 5
            print(f"      {r:.4f}")

    # Show episode reward
    print(f"\n5. Episode reward calculation:")
    print(f"   Sum of step rewards: {sum(step_rewards):.4f}")
    print(f"   Episode reward: {episode_reward:.4f}")
    print(f"   Match: {'✅' if abs(sum(step_rewards) - episode_reward) < 0.01 else '❌'}")

    # Show distribution
    print(f"\n6. Step reward distribution:")
    bins = [-10, -5, 0, 5, 10]
    for i in range(len(bins)-1):
        count = sum(1 for r in step_rewards if bins[i] <= r < bins[i+1])
        pct = 100 * count / len(step_rewards)
        bar = '█' * int(pct / 2)
        print(f"   [{bins[i]:+3.0f}, {bins[i+1]:+3.0f}): {count:3d} ({pct:5.1f}%) {bar}")

    # Final count at upper bound
    count_at_10 = sum(1 for r in step_rewards if r == 10.0)
    count_at_minus10 = sum(1 for r in step_rewards if r == -10.0)

    print(f"\n7. Clipping statistics:")
    print(f"   Rewards at upper bound (+10): {count_at_10}")
    print(f"   Rewards at lower bound (-10): {count_at_minus10}")

    if count_at_10 > 0 or count_at_minus10 > 0:
        print("   ✅ Clipping is being triggered (rewards hitting bounds)")
    else:
        print("   ℹ️  No rewards hit the bounds (all within range naturally)")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

    return clipped


if __name__ == "__main__":
    success = test_reward_clipping()
    sys.exit(0 if success else 1)
