#!/usr/bin/env python3
"""
Deep investigation of early stopping bug

Analyzes evaluation logs to understand exactly when callbacks are triggered
"""
import numpy as np
from pathlib import Path
import pandas as pd

def analyze_fold_evaluations(fold_dir, fold_idx):
    """Analyze evaluation logs for a single fold"""
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx} - DETAILED ANALYSIS")
    print(f"{'='*80}")

    eval_file = fold_dir / 'evaluations.npz'
    if not eval_file.exists():
        print(f"  No evaluations.npz found")
        return

    # Load evaluation data
    with np.load(eval_file) as data:
        timesteps = data['timesteps']
        results = data['results']  # Shape: (n_evals, n_episodes)

    # Calculate mean reward per evaluation
    mean_rewards = results.mean(axis=1)

    print(f"\nEvaluation Timeline:")
    print(f"{'Eval':<6} {'Step':<10} {'Mean Reward':<15} {'Best So Far':<15} {'Improved?':<12} {'No Improve Count':<18}")
    print("-" * 90)

    best_reward = -np.inf
    no_improvement_count = 0
    best_eval_idx = 0

    for i, (step, reward) in enumerate(zip(timesteps, mean_rewards)):
        improved = reward > best_reward

        if improved:
            best_reward = reward
            best_eval_idx = i
            no_improvement_count = 0
            improved_str = "✓ YES"
        else:
            no_improvement_count += 1
            improved_str = f"  No ({no_improvement_count})"

        marker = "  ⭐ PEAK" if i == best_eval_idx and i == len(mean_rewards) - 1 else ""
        marker = "  ⭐ PEAK" if improved and i > 0 and all(mean_rewards[i] >= mean_rewards[j] for j in range(len(mean_rewards))) else marker

        print(f"{i+1:<6} {step:<10} {reward:<15.4f} {best_reward:<15.4f} {improved_str:<12} {no_improvement_count:<18} {marker}")

    # Analysis
    print(f"\n{'='*80}")
    print("EARLY STOPPING ANALYSIS")
    print(f"{'='*80}")

    best_step = timesteps[best_eval_idx]
    final_step = timesteps[-1]
    evals_after_peak = len(timesteps) - best_eval_idx - 1
    steps_after_peak = final_step - best_step

    print(f"Best evaluation: #{best_eval_idx + 1} at step {best_step:,}")
    print(f"Final evaluation: #{len(timesteps)} at step {final_step:,}")
    print(f"Evaluations after peak: {evals_after_peak}")
    print(f"Steps after peak: {steps_after_peak:,}")
    print(f"Final no-improvement count: {no_improvement_count}")

    print(f"\nExpected behavior (max_no_improvement_evals=5, min_evals=3):")
    print(f"  - Start checking after eval #{3 + 1} (n_calls > min_evals)")
    print(f"  - Stop when no_improvement_evals > 5 (i.e., at 6)")
    print(f"  - Should stop at eval #{best_eval_idx + 1 + 6}")

    expected_stop_eval = best_eval_idx + 1 + 6  # +1 for 1-indexing, +6 for the bug
    actual_stop_eval = len(timesteps)

    if actual_stop_eval == expected_stop_eval:
        print(f"  ✓ Stopped as expected at eval #{actual_stop_eval}")
    elif actual_stop_eval < expected_stop_eval:
        print(f"  ⚠️ Stopped EARLY at eval #{actual_stop_eval} (expected #{expected_stop_eval})")
    else:
        print(f"  ❌ Stopped LATE at eval #{actual_stop_eval} (expected #{expected_stop_eval})")
        print(f"  → {actual_stop_eval - expected_stop_eval} extra evaluations!")

    # Check for multiple peaks (reward hacking)
    print(f"\nReward Pattern Analysis:")
    if best_eval_idx == 0:
        print(f"  ⚠️ PEAKED AT FIRST EVALUATION - Possible reward hacking!")
    elif best_eval_idx < 3:
        print(f"  ⚠️ PEAKED VERY EARLY (eval #{best_eval_idx + 1}) - Suspicious")

    # Check if reward degraded significantly
    peak_reward = mean_rewards[best_eval_idx]
    final_reward = mean_rewards[-1]
    degradation = (peak_reward - final_reward) / abs(peak_reward) * 100
    print(f"  Peak reward: {peak_reward:.4f}")
    print(f"  Final reward: {final_reward:.4f}")
    print(f"  Degradation: {degradation:.1f}%")

    if degradation > 50:
        print(f"  ❌ SEVERE degradation (>{degradation:.0f}%)")
    elif degradation > 20:
        print(f"  ⚠️ Significant degradation ({degradation:.0f}%)")
    else:
        print(f"  ✓ Acceptable degradation ({degradation:.0f}%)")

    return {
        'fold': fold_idx,
        'total_evals': len(timesteps),
        'best_eval': best_eval_idx + 1,
        'best_step': best_step,
        'final_step': final_step,
        'evals_after_peak': evals_after_peak,
        'steps_after_peak': steps_after_peak,
        'final_no_improve_count': no_improvement_count,
        'degradation_pct': degradation,
        'peak_reward': peak_reward,
        'final_reward': final_reward
    }

def main():
    print("="*80)
    print("EARLY STOPPING BUG INVESTIGATION")
    print("="*80)
    print("\nConfiguration:")
    print("  max_no_improvement_evals = 5")
    print("  min_evals = 3")
    print("  eval_freq = 2500")
    print("\nExpected behavior:")
    print("  - Start checking after 3 evaluations")
    print("  - Stop when no_improvement_evals > 5 (at 6 due to SB3 bug)")
    print("  - Should run 6 evals past peak, then stop")

    models_dir = Path('models_minimal_features')
    results = []

    for fold_idx in [0, 1, 2]:
        fold_dir = models_dir / f'fold_{fold_idx}'
        if fold_dir.exists():
            result = analyze_fold_evaluations(fold_dir, fold_idx)
            if result:
                results.append(result)

    # Summary table
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        df = pd.DataFrame(results)
        print(df.to_string(index=False))

        print(f"\n{'='*80}")
        print("ROOT CAUSE ANALYSIS")
        print(f"{'='*80}")

        print("\n1. Stable-Baselines3 Bug:")
        print("   Code: if self.no_improvement_evals > self.max_no_improvement_evals:")
        print("   Issue: Uses '>' instead of '>='")
        print("   Impact: Stops at 6 evals instead of 5")

        # Check if all folds have 6 evals after peak
        evals_after = df['evals_after_peak'].values
        if all(e == 6 for e in evals_after):
            print("\n2. All folds stopped at 6 evals past peak ✓")
            print("   → Behavior is CONSISTENT with SB3 '>' bug")
            print("   → Fix: Change max_no_improvement_evals=5 to 4")
        else:
            print(f"\n2. Inconsistent stopping: {evals_after}")
            print("   → Additional issue beyond SB3 bug!")

            for i, result in enumerate(results):
                if result['evals_after_peak'] != 6:
                    print(f"\n   Fold {result['fold']}:")
                    print(f"     - Expected: 6 evals after peak")
                    print(f"     - Actual: {result['evals_after_peak']} evals after peak")

                    if result['best_eval'] == 1:
                        print(f"     - ⚠️ Peaked at FIRST eval - reward hacking likely")
                        print(f"     - After first eval, ALL subsequent evals are 'no improvement'")
                        print(f"     - Started counting from eval #2, stopped at eval #{2 + 6} = #{8}")
                    elif result['best_eval'] < 4:
                        print(f"     - ⚠️ Peaked at eval #{result['best_eval']} (before min_evals)")
                        print(f"     - Didn't start checking until eval #4")
                        print(f"     - This could explain extra evaluations")

        print(f"\n{'='*80}")
        print("RECOMMENDED FIX")
        print(f"{'='*80}")
        print("\nOption 1: Compensate for SB3 bug")
        print("  Change: max_no_improvement_evals=4 (instead of 5)")
        print("  Result: Will stop at 5 evals past peak (4+1)")
        print("  Pro: Simple one-line fix")
        print("  Con: Doesn't fix underlying bug")

        print("\nOption 2: Custom callback")
        print("  Create our own callback with '>=' instead of '>'")
        print("  Result: Exact control over stopping logic")
        print("  Pro: Full control, can add logging")
        print("  Con: More code to maintain")

        print("\nOption 3: Verbose mode + manual monitoring")
        print("  Change: verbose=1 in callback")
        print("  Result: See when stopping occurs")
        print("  Pro: Better visibility")
        print("  Con: Doesn't prevent overtraining")

if __name__ == '__main__':
    main()
