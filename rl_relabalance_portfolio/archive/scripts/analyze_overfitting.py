#!/usr/bin/env python3
"""
Analyze validation performance during training to detect overfitting.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 80)
print("VALIDATION PERFORMANCE ANALYSIS (Overfitting Detection)")
print("=" * 80)

folds = [0, 1, 2]
output_dir = Path('models_minimal_features')

for fold_idx in folds:
    fold_dir = output_dir / f'fold_{fold_idx}'
    eval_file = fold_dir / 'evaluations.npz'

    if not eval_file.exists():
        print(f"\nFold {fold_idx}: No evaluation file found")
        continue

    # Load evaluation data
    with np.load(eval_file) as data:
        timesteps = data['timesteps']
        results = data['results']  # Shape: (n_evals, n_episodes)
        ep_lengths = data['ep_lengths']

    # Calculate mean reward per evaluation
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    # Find best and final performance
    best_idx = np.argmax(mean_rewards)
    best_reward = mean_rewards[best_idx]
    best_step = timesteps[best_idx]
    final_reward = mean_rewards[-1]

    # Calculate performance drop
    drop = (best_reward - final_reward) / abs(best_reward) * 100

    # Check for overfitting patterns
    print(f"\nFold {fold_idx}:")
    print(f"  Evaluations: {len(timesteps)}")
    print(f"  Best val reward: {best_reward:.4f} at step {best_step:,}")
    print(f"  Final val reward: {final_reward:.4f} at step {timesteps[-1]:,}")
    print(f"  Performance drop: {drop:.1f}%")

    # Analyze trend after peak
    if best_idx < len(mean_rewards) - 1:
        rewards_after_peak = mean_rewards[best_idx:]
        if len(rewards_after_peak) > 1:
            # Calculate slope after peak (negative = degrading)
            steps_after_peak = timesteps[best_idx:] - timesteps[best_idx]
            if len(steps_after_peak) > 1 and steps_after_peak[-1] > 0:
                slope = (rewards_after_peak[-1] - rewards_after_peak[0]) / steps_after_peak[-1]
                print(f"  Trend after peak: {slope*100000:.2f} per 100k steps")
                if slope < -0.0001:
                    print(f"  ⚠️ Degrading after peak (overfitting)")
                elif slope > 0.0001:
                    print(f"  ✓ Improving after peak (continued learning)")
                else:
                    print(f"  ≈ Stable after peak")

    # Check if early stopping triggered
    if drop > 10:
        print(f"  ⚠️ Significant performance drop ({drop:.1f}%)")
        print(f"     Early stopping should have triggered!")
    else:
        print(f"  ✓ Stable performance (drop < 10%)")

    # Plot validation curve
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Val Reward')
    plt.fill_between(timesteps,
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     alpha=0.3, color='b')
    plt.axhline(y=best_reward, color='g', linestyle='--', label=f'Best ({best_reward:.3f})')
    plt.axhline(y=final_reward, color='r', linestyle='--', label=f'Final ({final_reward:.3f})')
    plt.axvline(x=best_step, color='g', linestyle=':', alpha=0.5, label=f'Best step ({best_step:,})')

    plt.xlabel('Training Steps')
    plt.ylabel('Validation Reward')
    plt.title(f'Fold {fold_idx}: Validation Performance During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fold_dir / f'validation_curve_fold{fold_idx}.png', dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {fold_dir / f'validation_curve_fold{fold_idx}.png'}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
Key Observations:
1. Check if validation rewards degrade after peak (overfitting)
2. Check if early stopping worked properly
3. Compare validation performance to test performance

If validation stayed stable but test performance varies widely,
this suggests models are fitting to validation-specific patterns.
""")
