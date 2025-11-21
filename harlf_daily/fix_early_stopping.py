"""
Quick Fix for EarlyStoppingCallback Metric Lookup

This script shows the corrected logic for the EarlyStoppingCallback._on_step method.
Apply these changes to harlf/callbacks.py lines 365-430.

Issues Fixed:
1. Metric lookup tries adding 'eval/' prefix before removing it
2. NaN/None metric values are handled gracefully
3. Better debugging output

"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EarlyStoppingCallbackFixed(BaseCallback):
    """
    ✅ FIXED VERSION: Monitors evaluation metrics and stops training when:
    1. No improvement for 'patience' evaluations
    2. Overfitting detected (train-val gap too large)
    """

    def __init__(
        self,
        eval_freq: int = 5000,
        patience: int = 5,
        min_delta: float = 0.01,
        metric: str = 'sharpe_ratio',  # No prefix needed
        min_evaluations: int = 3,
        check_overfitting: bool = True,
        max_train_val_gap: float = 0.5,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.min_evaluations = min_evaluations
        self.check_overfitting = check_overfitting
        self.max_train_val_gap = max_train_val_gap
        self.best_metric = -np.inf
        self.wait = 0
        self.evaluations_count = 0
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        """
        ✅ FIXED: Proper metric lookup with fallback logic.
        Returns False to stop training, True to continue.
        """
        # Check if it's time to evaluate (aligned with validation callback)
        if self.n_calls % self.eval_freq != 0:
            return True

        # Skip if we haven't done enough evaluations
        if self.evaluations_count < self.min_evaluations:
            self.evaluations_count += 1
            if self.verbose > 1:
                print(f"   Early stopping: Evaluation {self.evaluations_count}/{self.min_evaluations} (waiting)")
            return True

        # Check if logger exists
        if self.model.logger is None:
            if self.verbose > 0:
                print("⚠️  Early stopping: No logger available")
            return True

        # ✅ FIX 1: Try multiple metric name variations
        metric_key = None
        current_metric = None

        # Try with 'eval/' prefix first (most common)
        for prefix in ['eval/', 'validation/', '']:
            candidate_key = f"{prefix}{self.metric}" if prefix else self.metric

            if candidate_key in self.model.logger.name_to_value:
                metric_key = candidate_key
                current_metric = self.model.logger.name_to_value[metric_key]
                break

        # Metric not found
        if metric_key is None:
            if self.verbose > 0 and self.evaluations_count == self.min_evaluations:
                available_keys = list(self.model.logger.name_to_value.keys())
                print(f"⚠️  Early stopping: Metric '{self.metric}' not found in logger")
                print(f"   Available metrics: {[k for k in available_keys if 'eval' in k or 'val' in k]}")
            self.evaluations_count += 1
            return True

        # ✅ FIX 2: Handle NaN/None values
        if current_metric is None or (isinstance(current_metric, float) and np.isnan(current_metric)):
            if self.verbose > 0:
                print(f"⚠️  Early stopping: Invalid metric value: {current_metric}")
            self.evaluations_count += 1
            return True  # Don't count this as an evaluation

        # ✅ FIX 3: Better logging
        if self.verbose > 1:
            print(f"\n   Early stopping check @ step {self.num_timesteps}:")
            print(f"   Metric ({metric_key}): {current_metric:.4f}")
            print(f"   Best: {self.best_metric:.4f}")

        # Check for improvement
        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.wait = 0
            if self.verbose > 0:
                print(f"\n✅ Early stopping: Improved {metric_key}: {current_metric:.4f} (best: {self.best_metric:.4f})")
        else:
            self.wait += 1
            if self.verbose > 0:
                print(f"\n⏸️  Early stopping: No improvement ({self.wait}/{self.patience})")
                print(f"   Current: {current_metric:.4f}, Best: {self.best_metric:.4f}, Delta: {current_metric - self.best_metric:.4f}")

        # Check overfitting (if train metrics available)
        if self.check_overfitting:
            # Try to find corresponding train metric
            train_metric_key = metric_key.replace('eval/', 'train/').replace('validation/', 'train/')

            if train_metric_key in self.model.logger.name_to_value:
                train_val = self.model.logger.name_to_value[train_metric_key]
                val_val = current_metric

                # Handle NaN in train metric
                if train_val is not None and not (isinstance(train_val, float) and np.isnan(train_val)):
                    gap = train_val - val_val

                    if self.verbose > 1:
                        print(f"   Overfitting check: Train={train_val:.3f}, Val={val_val:.3f}, Gap={gap:.3f}")

                    if gap > self.max_train_val_gap:
                        if self.verbose > 0:
                            print(f"\n🛑 Early stopping: Overfitting detected!")
                            print(f"   Train: {train_val:.3f}, Val: {val_val:.3f}, Gap: {gap:.3f} > {self.max_train_val_gap}")
                        return False  # Stop due to overfitting

        # Check patience
        if self.wait >= self.patience:
            if self.verbose > 0:
                print(f"\n🛑 Early stopping triggered at step {self.num_timesteps}")
                print(f"   Best {metric_key}: {self.best_metric:.4f}")
                print(f"   No improvement for {self.patience} evaluations")
            return False  # Stop training

        self.evaluations_count += 1
        return True  # Continue training


# ============================================================================
# TESTING
# ============================================================================

def test_fixed_callback():
    """Test the fixed callback logic."""
    print("="*80)
    print("TESTING FIXED EARLY STOPPING CALLBACK")
    print("="*80)

    # Create mock logger
    class MockLogger:
        def __init__(self):
            self.name_to_value = {}

    # Create mock model
    class MockModel:
        def __init__(self):
            self.logger = MockLogger()

    # Create callback
    callback = EarlyStoppingCallbackFixed(
        eval_freq=5000,
        patience=3,
        min_delta=0.01,
        metric='sharpe_ratio',
        min_evaluations=2,
        check_overfitting=False,
        verbose=2
    )

    # Setup callback
    callback.model = MockModel()
    callback.n_calls = 0
    callback.num_timesteps = 0

    # Simulate training
    scenarios = [
        # (step, eval/sharpe_ratio, should_continue, description)
        (4999, None, True, "Before first eval"),
        (5000, 1.5, True, "First eval - below min"),
        (10000, 1.8, True, "Second eval - improving"),
        (15000, 2.0, True, "Third eval - improving (best so far)"),
        (20000, 1.95, True, "Fourth eval - no improvement (1/3)"),
        (25000, 1.97, True, "Fifth eval - no significant improvement (2/3)"),
        (30000, 1.96, False, "Sixth eval - no improvement (3/3) → STOP"),
    ]

    print("\n" + "-"*80)
    print("SCENARIO: Testing patience mechanism")
    print("-"*80)

    for step, metric_value, expected_continue, description in scenarios:
        callback.n_calls = step
        callback.num_timesteps = step

        # Set metric in logger
        if metric_value is not None:
            callback.model.logger.name_to_value['eval/sharpe_ratio'] = metric_value

        # Check callback
        should_continue = callback._on_step()

        # Verify
        status = "✅" if should_continue == expected_continue else "❌"
        print(f"\n{status} Step {step}: {description}")
        print(f"   Metric: {metric_value}, Continue: {should_continue} (expected: {expected_continue})")

        if should_continue != expected_continue:
            print(f"   ERROR: Expected {expected_continue}, got {should_continue}")
            return False

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    return True


def test_nan_handling():
    """Test NaN handling."""
    print("\n" + "="*80)
    print("TESTING NaN HANDLING")
    print("="*80)

    # Setup
    class MockLogger:
        name_to_value = {'eval/sharpe_ratio': np.nan}

    class MockModel:
        logger = MockLogger()

    callback = EarlyStoppingCallbackFixed(
        eval_freq=1,
        patience=1,
        metric='sharpe_ratio',
        min_evaluations=0,
        verbose=1
    )

    callback.model = MockModel()
    callback.n_calls = 1
    callback.num_timesteps = 1

    # Should continue despite NaN
    should_continue = callback._on_step()

    if should_continue:
        print("✅ Correctly handled NaN metric")
        return True
    else:
        print("❌ Incorrectly stopped on NaN metric")
        return False


if __name__ == '__main__':
    # Run tests
    test1 = test_fixed_callback()
    test2 = test_nan_handling()

    if test1 and test2:
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - READY TO APPLY FIX")
        print("="*80)
        print("\nTo apply:")
        print("1. Backup harlf/callbacks.py")
        print("2. Replace EarlyStoppingCallback class with EarlyStoppingCallbackFixed")
        print("3. Rename class back to EarlyStoppingCallback")
        print("4. Re-run training")
    else:
        print("\n❌ SOME TESTS FAILED")
