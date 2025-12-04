"""
Custom Monitoring and Logging Utilities for RL Portfolio Training

This module provides:
- PortfolioMonitor: Gym wrapper that tracks portfolio statistics across episodes
- TrainingLogger: Comprehensive callback for logging training/validation metrics with early stopping

These utilities work together to properly track portfolio performance during RL training,
fixing issues with stats being lost after environment resets.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Dict, List
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback


class PortfolioMonitor(gym.Wrapper):
    """
    Custom Monitor that stores portfolio stats before environment resets.

    This fixes the issue where get_portfolio_stats() returns zeros after reset.
    Wraps a PortfolioEnv and tracks episode statistics properly across resets.

    How it works:
    1. Tracks rewards during episode in current_episode_rewards
    2. When episode ends (terminated/truncated), stores stats from env.get_portfolio_stats()
    3. Stores stats in info['portfolio_stats'] for callbacks to access
    4. On reset(), stores final stats if not already stored (edge case handling)

    Features:
    - Tracks episode returns, lengths, and portfolio statistics
    - Stores stats before reset to prevent data loss
    - Provides methods to retrieve episode-specific or summary statistics
    """

    def __init__(self, env):
        """
        Initialize PortfolioMonitor.

        Parameters:
        -----------
        env : gym.Env
            The environment to wrap (typically PortfolioEnv or PortfolioEnv2)
        """
        super().__init__(env)
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_portfolio_stats = []
        self.current_episode_rewards = []
        self._episode_ended = False  # Track if current episode has ended

    def reset(self, **kwargs):
        """
        Reset environment and store final stats from previous episode if needed.
        
        Note: Stats are typically stored in step() when episode ends, but this
        handles edge cases where reset() is called without step() detecting the end.
        """
        # Store final stats before reset (if episode ended but stats not stored yet)
        if hasattr(self.env, 'get_portfolio_stats'):
            stats = self.env.get_portfolio_stats()
            # Only store if:
            # 1. Stats are meaningful (portfolio_value > 0 or we had rewards)
            # 2. Episode ended but stats weren't stored yet (edge case)
            if (stats.get('portfolio_value', 0) > 0 or len(self.current_episode_rewards) > 0) and \
               self._episode_ended and \
               len(self.episode_portfolio_stats) < len(self.episode_returns):
                # This handles edge case where reset() is called but step() didn't store stats
                self.episode_portfolio_stats.append(stats)

        # Reset tracking for new episode
        self.current_episode_rewards = []
        self._episode_ended = False

        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Execute step and track rewards/stats.
        
        When episode ends, stores portfolio stats from environment and adds them to info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_episode_rewards.append(reward)
        
        episode_done = terminated or truncated

        # If episode done, store stats in info and track episode metrics
        if episode_done:
            self._episode_ended = True
            
            # Get final portfolio statistics from environment
            stats = self.env.get_portfolio_stats()
            
            # Add stats to info dict for callbacks to access
            info['portfolio_stats'] = stats
            
            # Calculate episode metrics
            episode_return = sum(self.current_episode_rewards)
            episode_length = len(self.current_episode_rewards)
            
            # Store episode metrics (synchronized lists)
            self.episode_returns.append(episode_return)
            self.episode_lengths.append(episode_length)
            
            # Store portfolio stats for this episode
            # This is synchronized with episode_returns and episode_lengths
            self.episode_portfolio_stats.append(stats)

        return obs, reward, terminated, truncated, info

    def get_episode_stats(self, episode_idx: int = -1) -> Dict:
        """
        Get portfolio stats for a specific episode.
        
        Parameters:
        -----------
        episode_idx : int
            Episode index (default: -1 for most recent)
            
        Returns:
        --------
        Dict
            Portfolio statistics dictionary with keys:
            - total_return: Cumulative return (e.g., 0.5 = 50% return)
            - sharpe_ratio: Annualized Sharpe ratio
            - volatility: Annualized volatility
            - max_drawdown: Maximum drawdown (negative value)
            - portfolio_value: Final portfolio value
        """
        if len(self.episode_portfolio_stats) == 0:
            return {}
        return self.episode_portfolio_stats[episode_idx]
    
    def get_all_episode_stats(self) -> List[Dict]:
        """
        Get all stored episode portfolio statistics.
        
        Returns:
        --------
        List[Dict]
            List of portfolio statistics dictionaries, one per episode
        """
        return self.episode_portfolio_stats.copy()
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics across all episodes.
        
        Computes mean and std of portfolio metrics across all completed episodes.
        
        Returns:
        --------
        Dict
            Summary statistics including:
            - num_episodes: Number of completed episodes
            - mean_sharpe, std_sharpe: Mean and std of Sharpe ratios
            - mean_return, std_return: Mean and std of total returns
            - mean_volatility: Mean volatility across episodes
            - mean_max_drawdown: Mean maximum drawdown
            - mean_episode_length: Average episode length in steps
            - mean_episode_return: Average cumulative reward per episode
        """
        if len(self.episode_portfolio_stats) == 0:
            return {}
        
        # Extract metrics from all episodes
        sharpe_ratios = [s.get('sharpe_ratio', 0.0) for s in self.episode_portfolio_stats]
        total_returns = [s.get('total_return', 0.0) for s in self.episode_portfolio_stats]
        volatilities = [s.get('volatility', 0.0) for s in self.episode_portfolio_stats]
        max_drawdowns = [s.get('max_drawdown', 0.0) for s in self.episode_portfolio_stats]
        
        return {
            'num_episodes': len(self.episode_portfolio_stats),
            'mean_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0.0,
            'std_sharpe': np.std(sharpe_ratios) if sharpe_ratios else 0.0,
            'mean_return': np.mean(total_returns) if total_returns else 0.0,
            'std_return': np.std(total_returns) if total_returns else 0.0,
            'mean_volatility': np.mean(volatilities) if volatilities else 0.0,
            'mean_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0.0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'mean_episode_return': np.mean(self.episode_returns) if self.episode_returns else 0.0
        }


class TrainingLogger(BaseCallback):
    """
    Comprehensive logging callback for PPO training.

    Tracks training and validation metrics, implements early stopping,
    and saves logs to disk. Works with PortfolioMonitor to properly
    extract portfolio statistics.

    How it works:
    1. Training episodes: Monitors info['portfolio_stats'] from PortfolioMonitor
       and logs metrics for each completed training episode
    2. Validation: Periodically runs full validation episode and logs metrics
    3. Early stopping: Stops training if validation Sharpe doesn't improve for
       'patience' consecutive evaluations
    4. Model checkpointing: Saves best model based on validation Sharpe ratio

    Features:
    - Episode-level training metrics (logged from PortfolioMonitor)
    - Periodic validation evaluation (runs full episode)
    - Early stopping based on validation Sharpe ratio
    - CSV logging of all metrics (train_episodes.csv, val_evaluations.csv)
    - Best model checkpointing (saves to fold_dir/best_model.zip)

    Usage:
    ------
    ```python
    from monitoring import TrainingLogger
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Wrap validation env with PortfolioMonitor
    def make_val_env():
        env = PortfolioEnv2(...)
        return PortfolioMonitor(env)
    
    val_env = DummyVecEnv([make_val_env])
    
    logger = TrainingLogger(
        eval_env=val_env,
        eval_freq=5000,  # Evaluate every 5000 steps
        fold_dir=Path('models/fold_0'),
        patience=20,  # Stop if no improvement for 20 evaluations
        verbose=1
    )
    
    model.learn(total_timesteps=100000, callback=logger)
    best_sharpe = logger.get_best_sharpe()
    ```
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        fold_dir: Path,
        patience: int = 20,
        verbose: int = 1
    ):
        """
        Initialize TrainingLogger.

        Parameters:
        -----------
        eval_env : VecEnv
            Vectorized validation environment (should be wrapped with PortfolioMonitor)
        eval_freq : int
            Evaluate every N steps (0 to disable validation)
        fold_dir : Path
            Directory to save logs and models
        patience : int
            Early stopping patience in number of evaluations (default: 20)
            Training stops if validation Sharpe doesn't improve for 'patience' evaluations
        verbose : int
            Verbosity level (0=silent, 1=info)
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.fold_dir = Path(fold_dir)
        self.patience = patience

        # Tracking state
        self.best_val_sharpe = -np.inf  # Track best validation Sharpe
        self.no_improvement_count = 0  # Count consecutive evaluations without improvement
        self.early_stopped = False  # Flag if early stopping triggered

        # Logging storage
        self.train_episodes = []  # List of training episode metrics
        self.val_evaluations = []  # List of validation evaluation metrics

        # Create log directory
        (self.fold_dir / 'logs').mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at every training step.
        
        Returns:
        --------
        bool
            False to stop training (early stopping), True to continue
        """
        # Track training episode completion
        # PortfolioMonitor stores stats in info['portfolio_stats'] when episode ends
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'portfolio_stats' in info:
                stats = info['portfolio_stats']

                # Log training episode metrics
                self.train_episodes.append({
                    'step': self.num_timesteps,
                    'episode': len(self.train_episodes),
                    'total_return': stats.get('total_return', 0.0),
                    'sharpe_ratio': stats.get('sharpe_ratio', 0.0),
                    'volatility': stats.get('volatility', 0.0),
                    'max_drawdown': stats.get('max_drawdown', 0.0),
                    'portfolio_value': stats.get('portfolio_value', 0.0)
                })

                # Print progress periodically
                if self.verbose > 0 and len(self.train_episodes) % 20 == 0:
                    print(f"  Train Episode {len(self.train_episodes)}: "
                          f"Sharpe={stats.get('sharpe_ratio', 0.0):.3f}, "
                          f"Return={stats.get('total_return', 0.0)*100:.2f}%")

        # Run validation evaluation periodically
        should_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if should_eval:
            self._run_validation()

            # Check early stopping after validation
            if self.no_improvement_count >= self.patience:
                print(f"\n  Early stopping at step {self.num_timesteps}")
                self.early_stopped = True
                return False  # Stop training

        return True  # Continue training

    def _run_validation(self):
        """
        Run full validation episode and log results.
        
        This method:
        1. Resets validation environment
        2. Runs complete episode with deterministic policy
        3. Extracts portfolio stats (from PortfolioMonitor if available, else from env)
        4. Logs validation metrics
        5. Updates best model if validation Sharpe improved
        6. Updates early stopping counter
        """
        # Reset and run complete validation episode
        obs = self.eval_env.reset()
        episode_reward = 0.0  # Cumulative reward (sum of step rewards)
        episode_length = 0
        done = False
        stats = None

        # Run episode to completion
        while not done:
            # Use deterministic policy for validation (no exploration)
            action, _ = self.model.predict(obs, deterministic=True)

            # Step environment (VecEnv returns arrays)
            obs, reward, dones, info = self.eval_env.step(action)

            # VecEnv returns arrays, access first element [0]
            episode_reward += reward[0]
            episode_length += 1

            # Check if episode ended
            if dones[0]:
                done = True
                # Try to get stats from PortfolioMonitor (stored in info)
                if len(info) > 0 and 'portfolio_stats' in info[0]:
                    stats = info[0]['portfolio_stats']
                else:
                    # Fallback: unwrap VecEnv and get stats directly from environment
                    val_env_unwrapped = self.eval_env.envs[0]
                    while hasattr(val_env_unwrapped, 'env'):
                        val_env_unwrapped = val_env_unwrapped.env
                    if hasattr(val_env_unwrapped, 'get_portfolio_stats'):
                        stats = val_env_unwrapped.get_portfolio_stats()

        # Ensure stats is defined (fallback if loop exited unexpectedly)
        if stats is None:
            val_env_unwrapped = self.eval_env.envs[0]
            while hasattr(val_env_unwrapped, 'env'):
                val_env_unwrapped = val_env_unwrapped.env
            if hasattr(val_env_unwrapped, 'get_portfolio_stats'):
                stats = val_env_unwrapped.get_portfolio_stats()
            else:
                # Last resort: return empty stats
                stats = {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'volatility': 0.0,
                    'max_drawdown': 0.0,
                    'portfolio_value': 0.0
                }

        # Extract validation Sharpe ratio (primary metric for early stopping)
        val_sharpe = stats.get('sharpe_ratio', 0.0)
        
        # Log validation results
        self.val_evaluations.append({
            'step': self.num_timesteps,
            'evaluation': len(self.val_evaluations),
            'reward': episode_reward,  # Sum of step rewards
            'length': episode_length,
            'total_return': stats.get('total_return', 0.0),  # Portfolio return
            'sharpe_ratio': val_sharpe,  # Portfolio Sharpe ratio
            'volatility': stats.get('volatility', 0.0),
            'max_drawdown': stats.get('max_drawdown', 0.0),
            'portfolio_value': stats.get('portfolio_value', 0.0)
        })

        # Print validation update
        if self.verbose > 0:
            print(f"\n  Validation at step {self.num_timesteps}: "
                  f"Sharpe={val_sharpe:.3f}, "
                  f"Return={stats.get('total_return', 0.0)*100:.2f}%, "
                  f"Drawdown={stats.get('max_drawdown', 0.0)*100:.2f}%")

        # Check if this is the best model (based on validation Sharpe)
        if val_sharpe > self.best_val_sharpe:
            # New best model found
            self.best_val_sharpe = val_sharpe
            self.no_improvement_count = 0  # Reset counter

            # Save best model checkpoint
            model_path = self.fold_dir / 'best_model.zip'
            self.model.save(str(model_path))
            print(f"  -> New best model saved (Sharpe: {val_sharpe:.3f})")
        else:
            # No improvement
            self.no_improvement_count += 1
            if self.verbose > 0:
                print(f"  -> No improvement for {self.no_improvement_count}/{self.patience} evaluations")

    def _on_training_end(self) -> None:
        """
        Called at end of training (or early stopping).
        
        Saves all logged metrics to CSV files.
        """
        # Save training episode logs
        if self.train_episodes:
            train_df = pd.DataFrame(self.train_episodes)
            train_df.to_csv(self.fold_dir / 'logs' / 'train_episodes.csv', index=False)

        # Save validation evaluation logs
        if self.val_evaluations:
            val_df = pd.DataFrame(self.val_evaluations)
            val_df.to_csv(self.fold_dir / 'logs' / 'val_evaluations.csv', index=False)

        # Print summary
        print(f"\n  Training Summary:")
        print(f"    Total episodes: {len(self.train_episodes)}")
        print(f"    Total validations: {len(self.val_evaluations)}")
        print(f"    Best validation Sharpe: {self.best_val_sharpe:.3f}")
        print(f"    Early stopped: {self.early_stopped}")

    def get_best_sharpe(self) -> float:
        """
        Return the best validation Sharpe ratio achieved during training.
        
        Returns:
        --------
        float
            Best validation Sharpe ratio (or -inf if no validations run)
        """
        return self.best_val_sharpe

