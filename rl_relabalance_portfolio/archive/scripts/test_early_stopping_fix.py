#!/usr/bin/env python3
"""
Test the fixed early stopping callback

Simulates evaluation scenarios to verify callback stops at correct time
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MockEvalCallback:
    """Mock EvalCallback for testing"""
    def __init__(self):
        self.best_mean_reward = -np.inf
        self.evaluation_num = 0

    def evaluate(self, reward):
        """Simulate an evaluation"""
        self.evaluation_num += 1
        if reward > self.best_mean_reward:
            self.best_mean_reward = reward
        return reward


class StopTrainingOnNoModelImprovementFixed(BaseCallback):
    """Fixed version - test implementation"""

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0
        self.parent = None

    def _on_step(self) -> bool:
        continue_training = True

        if self.n_calls > self.min_evals:
            current_best = self.parent.best_mean_reward

            if current_best > self.last_best_mean_reward:
                self.no_improvement_evals = 0
                if self.verbose >= 2:
                    print(f"  Eval {self.n_calls}: New best {current_best:.4f}, reset counter")
            else:
                self.no_improvement_evals += 1
                if self.verbose >= 2:
                    print(f"  Eval {self.n_calls}: No improvement, counter = {self.no_improvement_evals}")

                # FIXED: Use '>=' instead of '>'
                if self.no_improvement_evals >= self.max_no_improvement_evals:
                    continue_training = False

            self.last_best_mean_reward = current_best
        else:
            if self.verbose >= 2:
                print(f"  Eval {self.n_calls}: Warmup (min_evals={self.min_evals})")
            self.last_best_mean_reward = self.parent.best_mean_reward

        if self.verbose >= 1 and not continue_training:
            print(f"\n  STOPPED at eval {self.n_calls}, no improvement for {self.no_improvement_evals} evals")

        return continue_training


def test_scenario(name, rewards, max_no_improvement=5, min_evals=3, verbose=2):
    """Test a reward scenario"""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")
    print(f"Config: max_no_improvement={max_no_improvement}, min_evals={min_evals}")
    print(f"Rewards: {rewards}\n")

    # Create mock parent
    parent = MockEvalCallback()

    # Create callback
    callback = StopTrainingOnNoModelImprovementFixed(
        max_no_improvement_evals=max_no_improvement,
        min_evals=min_evals,
        verbose=verbose
    )
    callback.parent = parent
    callback._init_callback()

    # Simulate evaluations
    for i, reward in enumerate(rewards):
        parent.evaluate(reward)
        callback.n_calls = i + 1

        continue_training = callback._on_step()

        if not continue_training:
            print(f"\nResult: Stopped at evaluation {i + 1}/{len(rewards)}")
            break
    else:
        print(f"\nResult: Completed all {len(rewards)} evaluations without stopping")

    # Analyze
    peak_idx = np.argmax(rewards)
    peak_reward = rewards[peak_idx]
    evals_after_peak = callback.n_calls - (peak_idx + 1)

    print(f"\nAnalysis:")
    print(f"  Peak reward: {peak_reward:.4f} at eval #{peak_idx + 1}")
    print(f"  Final eval: #{callback.n_calls}")
    print(f"  Evals after peak: {evals_after_peak}")
    print(f"  Expected evals after peak: {max_no_improvement} (if peak before min_evals)")
    print(f"  or {max_no_improvement} (if peak after min_evals)")

    # Check correctness
    # After warmup (min_evals), callback starts checking at eval (min_evals + 1)
    checking_start_eval = min_evals + 1  # First eval where checking happens

    if peak_idx + 1 < checking_start_eval:
        # Peak during warmup - patience counted from when checking starts
        # checking_start eval has count=1, next has count=2, ...
        # Stops when count reaches patience, which is at checking_start + (patience - 1)
        expected_stop = checking_start_eval + (max_no_improvement - 1)
    else:
        # Peak at or after checking start - patience counted from NEXT eval after peak
        # If peak at eval #5, then eval #6 has count=1, #7 has count=2, ...
        # Stops at eval #(5 + patience)
        expected_stop = peak_idx + 1 + max_no_improvement

    # Special case: continuous improvement (no peak)
    if callback.n_calls == len(rewards) and all(rewards[i] <= rewards[i+1] for i in range(len(rewards)-1)):
        print(f"  ✓ CORRECT: No stopping (continuous improvement)")
        return True

    if callback.n_calls == expected_stop:
        print(f"  ✓ CORRECT: Stopped at eval {callback.n_calls} (expected {expected_stop})")
        return True
    else:
        print(f"  ❌ INCORRECT: Stopped at eval {callback.n_calls} (expected {expected_stop})")
        return False


def main():
    print("="*80)
    print("EARLY STOPPING CALLBACK TEST SUITE")
    print("="*80)

    results = []

    # Test 1: Normal case - peak in middle
    results.append(test_scenario(
        "Normal case - peak at eval 5",
        rewards=[0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        max_no_improvement=5,
        min_evals=3
    ))

    # Test 2: Fold 1 scenario - peak at eval 1
    results.append(test_scenario(
        "Fold 1 scenario - peak at eval 1 (reward hacking)",
        rewards=[1.0, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0],
        max_no_improvement=5,
        min_evals=3
    ))

    # Test 3: Peak at eval 3 (at min_evals boundary)
    results.append(test_scenario(
        "Boundary case - peak at eval 3 (at min_evals)",
        rewards=[0.2, 0.5, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0, -0.1, -0.2],
        max_no_improvement=5,
        min_evals=3
    ))

    # Test 4: Peak at eval 4 (just after min_evals)
    results.append(test_scenario(
        "Peak at eval 4 (just after warmup)",
        rewards=[0.2, 0.4, 0.6, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0, -0.1],
        max_no_improvement=5,
        min_evals=3
    ))

    # Test 5: Continuous improvement (no stopping)
    results.append(test_scenario(
        "Continuous improvement - should NOT stop",
        rewards=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        max_no_improvement=5,
        min_evals=3
    ))

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")

    if all(results):
        print("\n✓ ALL TESTS PASSED - Callback is working correctly!")
    else:
        print("\n❌ SOME TESTS FAILED - Callback needs fixing")

    # Compare with SB3 bug
    print(f"\n{'='*80}")
    print("SB3 BUG COMPARISON")
    print(f"{'='*80}")
    print("With SB3 bug (using '>'), max_no_improvement=5:")
    print("  - Stops when no_improvement_evals > 5 (i.e., at 6)")
    print("  - Result: 1 extra evaluation past peak")
    print("\nWith our fix (using '>='), max_no_improvement=5:")
    print("  - Stops when no_improvement_evals >= 5 (i.e., at 5)")
    print("  - Result: Exactly as configured")

    return all(results)


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
