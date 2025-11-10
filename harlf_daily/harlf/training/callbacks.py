"""
Custom Callbacks for RL Training

Implements:
1. Early stopping based on validation performance
2. Portfolio weights logging
3. Best model checkpointing
4. Performance metric tracking

Created: 2025-01-09
"""

import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import json

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback based on validation performance.

    Stops training if validation Sharpe ratio does not improve
    for a specified number of evaluations.

    Args:
        patience: Number of evaluations to wait for improvement (default: 5)
        min_delta: Minimum improvement threshold (default: 0.01)
        metric: Metric to monitor ('sharpe_ratio', 'total_return', etc.)
        verbose: Verbosity level
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.01,
        metric: str = 'sharpe_ratio',
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_metric = -np.inf
        self.wait = 0
        self.stopped_epoch = 0

    def _on_step(self) -> bool:
        """Called at each step. Returns False to stop training."""
        # This will be triggered by evaluation callback
        return True

    def on_evaluation_end(self, eval_metrics: Dict[str, float]) -> bool:
        """
        Called after evaluation.

        Args:
            eval_metrics: Dictionary of evaluation metrics

        Returns:
            False if training should stop, True otherwise
        """
        if self.metric not in eval_metrics:
            if self.verbose > 0:
                print(f"Warning: Metric '{self.metric}' not found in eval_metrics")
            return True

        current_metric = eval_metrics[self.metric]

        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.wait = 0
            if self.verbose > 0:
                print(f"Early stopping: Improved {self.metric}={current_metric:.4f}")
        else:
            self.wait += 1
            if self.verbose > 0:
                print(f"Early stopping: No improvement ({self.wait}/{self.patience})")

            if self.wait >= self.patience:
                self.stopped_epoch = self.num_timesteps
                if self.verbose > 0:
                    print(f"Early stopping triggered at step {self.stopped_epoch}")
                return False  # Stop training

        return True


class PortfolioWeightsLogger(BaseCallback):
    """
    Logs portfolio weights at the end of each episode.

    Useful for analyzing learned allocation strategies.

    Args:
        log_freq: Frequency of logging (in timesteps)
        verbose: Verbosity level
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        """Log weights when episode ends."""
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            # Get the last observation info
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]

                if 'weights' in info:
                    weights = info['weights']

                    # Log each ticker weight
                    from harlf.config import TICKERS
                    for i, ticker in enumerate(TICKERS):
                        self.logger.record(f'weights/{ticker}', weights[i])

                # Log portfolio metrics if available
                metrics_to_log = [
                    'portfolio_value', 'portfolio_return', 'turnover',
                    'drawdown', 'transaction_cost'
                ]
                for metric in metrics_to_log:
                    if metric in info:
                        self.logger.record(f'episode/{metric}', info[metric])

        return True


class BestModelCallback(BaseCallback):
    """
    Saves the best model based on a specified metric.

    Args:
        save_path: Directory to save the best model
        metric: Metric to use for comparison (default: 'sharpe_ratio')
        verbose: Verbosity level
    """

    def __init__(
        self,
        save_path: Path,
        metric: str = 'sharpe_ratio',
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.best_metric = -np.inf
        self.best_model_path = None

    def _on_step(self) -> bool:
        """Check if current model is best after evaluation."""
        return True

    def save_best_model(self, current_metric: float, model_name: str = "best_model"):
        """
        Save model if it's the best so far.

        Args:
            current_metric: Current value of the tracked metric
            model_name: Name for the saved model file
        """
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_model_path = self.save_path / f"{model_name}.zip"

            self.model.save(self.best_model_path)

            # Save metadata
            metadata = {
                'best_metric': float(self.best_metric),
                'metric_name': self.metric,
                'timesteps': int(self.num_timesteps),
            }

            metadata_path = self.save_path / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            if self.verbose > 0:
                print(f"Best model saved: {self.metric}={current_metric:.4f} "
                      f"(previous best: {self.best_metric:.4f})")


class MetricsTrackingCallback(BaseCallback):
    """
    Tracks and logs comprehensive training metrics.

    Computes episode-level metrics (Sharpe, returns, drawdown, etc.)
    and logs them to TensorBoard.

    Args:
        log_freq: Frequency of logging (in episodes)
        verbose: Verbosity level
    """

    def __init__(self, log_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0

    def _on_step(self) -> bool:
        """Compute and log metrics when episode ends."""
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1

            # Get environment (unwrap if vectorized)
            env = self.training_env
            if isinstance(env, VecEnv):
                env = env.envs[0]

            # Get episode metrics
            if hasattr(env, 'get_episode_metrics'):
                metrics = env.get_episode_metrics()

                # Log metrics
                for key, value in metrics.items():
                    self.logger.record(f'episode_metrics/{key}', value)

                if self.verbose > 0 and self.episode_count % self.log_freq == 0:
                    print(f"Episode {self.episode_count}: "
                          f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
                          f"Return={metrics.get('total_return', 0):.4f}, "
                          f"Drawdown={metrics.get('max_drawdown', 0):.4f}")

        return True


class PeriodicCheckpointCallback(BaseCallback):
    """
    Saves model checkpoints at regular intervals.

    Args:
        save_freq: Frequency of checkpoints (in timesteps)
        save_path: Directory to save checkpoints
        name_prefix: Prefix for checkpoint filenames
        verbose: Verbosity level
    """

    def __init__(
        self,
        save_freq: int = 10000,
        save_path: Path = Path("./checkpoints"),
        name_prefix: str = "rl_model",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        """Save checkpoint if frequency reached."""
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            self.model.save(checkpoint_path)

            if self.verbose > 0:
                print(f"Checkpoint saved: {checkpoint_path}")

        return True


# ============================================================================
# TESTING
# ============================================================================

def test_callbacks():
    """Test callbacks (basic instantiation)."""
    print("="*80)
    print("TESTING CALLBACKS")
    print("="*80)

    # Test 1: Early Stopping
    print("\n1. Testing EarlyStoppingCallback...")
    early_stop = EarlyStoppingCallback(patience=3, min_delta=0.01, verbose=1)
    print("   ✓ EarlyStoppingCallback instantiated")

    # Test 2: Portfolio Weights Logger
    print("\n2. Testing PortfolioWeightsLogger...")
    weights_logger = PortfolioWeightsLogger(log_freq=100, verbose=1)
    print("   ✓ PortfolioWeightsLogger instantiated")

    # Test 3: Best Model Callback
    print("\n3. Testing BestModelCallback...")
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        best_model = BestModelCallback(save_path=Path(tmpdir), metric='sharpe_ratio', verbose=1)
        print("   ✓ BestModelCallback instantiated")

    # Test 4: Metrics Tracking
    print("\n4. Testing MetricsTrackingCallback...")
    metrics = MetricsTrackingCallback(log_freq=1, verbose=1)
    print("   ✓ MetricsTrackingCallback instantiated")

    # Test 5: Periodic Checkpoint
    print("\n5. Testing PeriodicCheckpointCallback...")
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = PeriodicCheckpointCallback(
            save_freq=5000,
            save_path=Path(tmpdir),
            verbose=1
        )
        print("   ✓ PeriodicCheckpointCallback instantiated")

    print("\n" + "="*80)
    print("✅ ALL CALLBACK TESTS PASSED")
    print("="*80)


if __name__ == '__main__':
    test_callbacks()
