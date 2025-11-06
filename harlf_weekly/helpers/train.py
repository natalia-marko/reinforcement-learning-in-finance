"""
Training module for Multi-Hierarchical RL Portfolio System.

This module provides training functions for base agents (Technical & Sentiment)
using Stable-Baselines3 PPO algorithm with:
- Early stopping based on validation performance
- Periodic evaluation and checkpointing
- Training history tracking
- Model saving/loading
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import time
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .environments import create_env
from .config import BaseAgentConfig, AGENT_MODELS_DIR, ACTIVATION_FUNCTIONS
from .utils import calculate_all_metrics, print_metrics

# ============================================================================
# EVALUATION CALLBACK
# ============================================================================

class EvaluationCallback(BaseCallback):
    """
    Callback for periodic evaluation during training.

    Evaluates agent on validation set and implements early stopping.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        patience: int = 5,
        min_delta: float = 0.01,
        save_path: Optional[Path] = None,
        verbose: int = 1
    ):
        """
        Initialize evaluation callback.

        Args:
            eval_env: Environment for evaluation
            eval_freq: Evaluate every N steps
            n_eval_episodes: Number of episodes per evaluation
            patience: Number of evaluations without improvement before stopping
            min_delta: Minimum improvement threshold
            save_path: Path to save best model
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path

        # Tracking
        self.best_mean_reward = -np.inf
        self.evaluations_since_best = 0
        self.evaluation_history = []
        self.early_stopped = False

    def _on_step(self) -> bool:
        """
        Called after each step.

        Returns:
            False to stop training (early stopping)
        """
        # Only evaluate at the correct schedule (avoid step 0 which can occur in some SB3 versions)
        if self.eval_freq > 0 and self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            eval_results = self._evaluate()  # Now returns dict

            # Avoid duplicated output if called very rapidly/recursively due to some SB3 callback bugs
            if hasattr(self, '_last_eval_call') and self._last_eval_call == self.n_calls:
                return True
            self._last_eval_call = self.n_calls

            mean_reward = eval_results['mean_reward']
            std_reward = eval_results['std_reward']

            # Print evaluation
            if self.verbose > 0:
                print(f"\n--- Evaluation at step {self.n_calls} ---")
                print(f"  Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
                if 'sharpe' in eval_results:
                    print(f"  Sharpe: {eval_results['sharpe']:.3f}")

            # Check for improvement
            if mean_reward > self.best_mean_reward + self.min_delta:
                if self.verbose > 0:
                    print(f"  🎉 New best! (previous: {self.best_mean_reward:.4f})")

                self.best_mean_reward = mean_reward
                self.evaluations_since_best = 0

                # Save best model
                if self.save_path is not None:
                    self.model.save(self.save_path)
                    if self.verbose > 0:
                        print(f"  💾 Model saved to {self.save_path}")
            else:
                self.evaluations_since_best += 1
                if self.verbose > 0:
                    print(f"  No improvement ({self.evaluations_since_best}/{self.patience})")

            # Store history with all metrics
            self.evaluation_history.append({
                'step': self.n_calls,
                'best_reward': self.best_mean_reward,
                'split': 'val',
                **eval_results  # Include all metrics from evaluation
            })

            # Early stopping check
            if self.evaluations_since_best >= self.patience:
                if self.verbose > 0:
                    print(f"\n⚠️  Early stopping triggered (no improvement for {self.patience} evaluations)")
                self.early_stopped = True
                return False

        return True

    def _evaluate(self) -> Dict:
        """
        Evaluate agent on validation set and calculate comprehensive metrics.

        Returns:
            Dictionary with mean_reward, std_reward, and portfolio metrics
        """
        episode_rewards = []
        all_returns = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_returns = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated

                # Collect per-step returns if available in info
                if 'return' in info:
                    episode_returns.append(info['return'])

            episode_rewards.append(episode_reward)

            # If returns not in info, try to get from environment attribute
            if not episode_returns and hasattr(self.eval_env, 'episode_returns'):
                episode_returns = self.eval_env.episode_returns

            all_returns.extend(episode_returns)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        # Calculate portfolio metrics if we have returns
        metrics = {}
        if all_returns:
            returns_array = np.array(all_returns)
            portfolio_metrics = calculate_all_metrics(returns_array)
            metrics = {
                'sharpe': float(portfolio_metrics.get('sharpe_ratio', 0.0)),  # ← Fixed: was 'sharpe', should be 'sharpe_ratio'
                'annual_return': float(portfolio_metrics.get('annual_return', 0.0)),
                'annual_volatility': float(portfolio_metrics.get('annual_volatility', 0.0)),
                'max_drawdown': float(portfolio_metrics.get('max_drawdown', 0.0)),
                'calmar': float(portfolio_metrics.get('calmar_ratio', 0.0)),  # ← Fixed: was 'calmar', should be 'calmar_ratio'
                'win_rate': float(portfolio_metrics.get('win_rate', 0.0))
            }

        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            **metrics
        }

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_base_agent(
    agent_type: str,
    agent_name: str,
    reward_type: str = 'ema_sharpe',
    config: Optional[BaseAgentConfig] = None,
    save_model: bool = True,
    verbose: int = 1
) -> Tuple[PPO, Dict]:
    """
    Train a base agent (Technical or Sentiment).

    Args:
        agent_type: Type of agent ('technical', 'sentiment')
        agent_name: Name for saving model (e.g., 'technical_ema_sharpe')
        reward_type: Type of reward function
        config: Training configuration (default: BaseAgentConfig)
        save_model: Whether to save trained model
        verbose: Verbosity level

    Returns:
        Tuple of (trained_model, training_history)
    """
    if config is None:
        config = BaseAgentConfig

    if verbose > 0:
        print("="*70)
        print(f"TRAINING BASE AGENT: {agent_name}")
        print("="*70)
        print(f"Agent Type: {agent_type}")
        print(f"Reward Type: {reward_type}")
        print(f"Total Timesteps: {config.TOTAL_TIMESTEPS:,}")
        print("="*70)

    start_time = time.time()

    # Create environments
    if verbose > 0:
        print("\n📦 Creating environments...")

    train_env = create_env(
        agent_type=agent_type,
        split='train',
        reward_type=reward_type
    )

    val_env = create_env(
        agent_type=agent_type,
        split='val',
        reward_type=reward_type
    )

    if verbose > 0:
        print(f"   Train env: {train_env.n_steps} steps")
        print(f"   Val env: {val_env.n_steps} steps")

    # Create model
    if verbose > 0:
        print("\n🤖 Creating PPO model...")

    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        policy_kwargs={
            'net_arch': config.NET_ARCH,
            'activation_fn': ACTIVATION_FUNCTIONS.get(config.ACTIVATION, nn.Tanh)
        },
        verbose=0
    )

    if verbose > 0:
        print(f"   Network architecture: {config.NET_ARCH}")
        print(f"   Learning rate: {config.LEARNING_RATE}")

    # Create callback
    save_path = AGENT_MODELS_DIR / f'{agent_name}.zip' if save_model else None

    eval_callback = EvaluationCallback(
        eval_env=val_env,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
        save_path=save_path,
        verbose=verbose
    )

    # Train
    if verbose > 0:
        print(f"\n🚀 Starting training for {config.TOTAL_TIMESTEPS:,} steps...")
        print(f"   Evaluation frequency: every {config.EVAL_FREQ:,} steps")
        print(f"   Early stopping patience: {config.PATIENCE} evaluations\n")

    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=eval_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        if verbose > 0:
            print("\n⚠️  Training interrupted by user")

    training_time = time.time() - start_time

    if verbose > 0:
        print(f"\n✅ Training completed in {training_time:.1f} seconds")
        print(f"   Best validation reward: {eval_callback.best_mean_reward:.4f}")
        if eval_callback.early_stopped:
            print(f"   Early stopped at step {eval_callback.n_calls}")

    # Compile training history
    training_history = {
        'agent_type': agent_type,
        'agent_name': agent_name,
        'reward_type': reward_type,
        'training_time': training_time,
        'total_steps': eval_callback.n_calls,
        'best_reward': eval_callback.best_mean_reward,
        'early_stopped': eval_callback.early_stopped,
        'evaluation_history': eval_callback.evaluation_history,
        'config': config.to_dict()
    }

    return model, training_history

def evaluate_agent(
    model: PPO,
    agent_type: str,
    split: str = 'test',
    reward_type: str = 'ema_sharpe',
    n_episodes: int = 1,
    verbose: int = 1
) -> Dict:
    """
    Evaluate trained agent on test set.

    Args:
        model: Trained PPO model
        agent_type: Type of agent
        split: Data split to evaluate on
        reward_type: Type of reward function
        n_episodes: Number of episodes (usually 1 for test)
        verbose: Verbosity level

    Returns:
        Dictionary with evaluation results
    """
    if verbose > 0:
        print("\n" + "="*70)
        print(f"EVALUATING AGENT ON {split.upper()} SET")
        print("="*70)

    # Create environment
    env = create_env(
        agent_type=agent_type,
        split=split,
        reward_type=reward_type
    )

    # Run episodes
    all_returns = []
    all_actions = []
    all_portfolio_values = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_actions = []
        episode_values = []

        while not done:
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            # Store action
            episode_actions.append(action.copy())
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Store info
            episode_values.append(info['portfolio_value'])
        # Store episode data
        episode_returns = env.episode_returns
        all_returns.extend(episode_returns)
        all_actions.append(np.array(episode_actions))
        all_portfolio_values.extend(episode_values)

    # Calculate metrics
    returns_array = np.array(all_returns)
    metrics = calculate_all_metrics(returns_array)

    # Add additional info
    metrics['final_value'] = all_portfolio_values[-1]
    metrics['initial_value'] = env.initial_capital

    if verbose > 0:
        print_metrics(metrics, title=f"{split.upper()} Set Performance")

    # Compile results
    results = {
        'metrics': metrics,
        'returns': returns_array.tolist(),
        'actions': [a.tolist() for a in all_actions],
        'portfolio_values': all_portfolio_values,
        'split': split
    }

    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(
    history: Dict,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot training history.

    Args:
        history: Training history dictionary
        figsize: Figure size
    """
    eval_history = history['evaluation_history']

    if len(eval_history) == 0:
        print("No evaluation history to plot")
        return

    df = pd.DataFrame(eval_history)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Mean reward over time
    ax = axes[0]
    ax.plot(df['step'], df['mean_reward'], marker='o', label='Mean Reward', linewidth=2)
    ax.plot(df['step'], df['best_reward'], linestyle='--', label='Best Reward',
            linewidth=2, alpha=0.7)
    ax.fill_between(df['step'],
                     df['mean_reward'] - df['std_reward'],
                     df['mean_reward'] + df['std_reward'],
                     alpha=0.2)
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title('Training Progress', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Improvement over evaluations
    ax = axes[1]
    improvements = df['mean_reward'].diff()
    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax.bar(range(len(improvements)), improvements, color=colors, alpha=0.6)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Evaluation Number', fontsize=11)
    ax.set_ylabel('Reward Change', fontsize=11)
    ax.set_title('Evaluation-to-Evaluation Improvement', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_agent_performance(
    results: Dict,
    agent_name: str,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Plot agent performance metrics.

    Args:
        results: Evaluation results dictionary
        agent_name: Name of agent for title
        figsize: Figure size
    """
    returns = np.array(results['returns'])
    values = results['portfolio_values']
    actions = np.array(results['actions'][0])  # First episode

    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle(f'{agent_name} - Performance Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Cumulative returns
    ax = axes[0]
    cumulative = (1 + returns).cumprod()
    ax.plot(cumulative, linewidth=2, color='steelblue')
    ax.set_ylabel('Cumulative Return', fontsize=11)
    ax.set_title('Portfolio Growth', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Drawdown
    ax = axes[1]
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    ax.fill_between(range(len(drawdown)), drawdown * 100, 0, color='red', alpha=0.3)
    ax.plot(drawdown * 100, color='darkred', linewidth=1.5)
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.set_title('Drawdown Analysis', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Portfolio weights over time
    ax = axes[2]
    from .config import TICKERS
    for i, ticker in enumerate(TICKERS):
        ax.plot(actions[:, i], label=ticker, linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Weight', fontsize=11)
    ax.set_title('Portfolio Allocation Over Time', fontsize=13, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ============================================================================
# HIERARCHICAL AGENT TRAINING (Super & Meta Agents)
# ============================================================================

def train_super_agent(
    train_env,
    val_env,
    model_name: str = 'super_agent',
    config=None,
    save_model: bool = True,
    verbose: int = 1
) -> Tuple[PPO, Dict]:
    """
    Train Super Agent (Level 2 of hierarchy).

    Args:
        train_env: Training environment
        val_env: Validation environment
        model_name: Name for saved model
        config: Configuration class (default: SuperAgentConfig)
        save_model: Whether to save trained model
        verbose: Verbosity level

    Returns:
        Tuple of (trained_model, training_history)
    """
    if config is None:
        from .config import SuperAgentConfig
        config = SuperAgentConfig

    if verbose > 0:
        print(f"\nTraining Super Agent: {model_name}")
        print("="*70)
        print(f"Config: {config.__name__}")
        print(f"Total timesteps: {config.TOTAL_TIMESTEPS:,}")

    # Create model
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        policy_kwargs=dict(
            net_arch=config.NET_ARCH,
            activation_fn=eval(f"torch.nn.{config.ACTIVATION.capitalize()}")
        ),
        verbose=0
    )

    # Create callback
    from .config import MODELS_DIR
    save_path = MODELS_DIR / f'{model_name}.zip' if save_model else None

    callback = EvaluationCallback(
        eval_env=val_env,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
        save_path=save_path,
        verbose=verbose
    )

    # Train
    start_time = time.time()
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback)
    training_time = time.time() - start_time

    if verbose > 0:
        print(f"\n✅ Training completed in {training_time:.1f}s")
        if callback.early_stopped:
            print(f"   Early stopped at step {callback.n_calls}")
        print(f"   Best mean reward: {callback.best_mean_reward:.4f}")

    # Prepare history - metrics now properly calculated by EvaluationCallback
    # Note: train evaluation not implemented yet, so train metrics will be empty
    history = {
        'train_rewards': [],  # Not tracked (callback only evaluates on val_env)
        'val_rewards': [eval_data['mean_reward'] for eval_data in callback.evaluation_history if eval_data.get('split') == 'val'],
        'train_sharpe': [],  # Not tracked
        'val_sharpe': [eval_data.get('sharpe', 0.0) for eval_data in callback.evaluation_history if eval_data.get('split') == 'val'],
        'train_returns': [],  # Not tracked
        'val_returns': [eval_data.get('annual_return', 0.0) for eval_data in callback.evaluation_history if eval_data.get('split') == 'val'],
        'train_drawdown': [],  # Not tracked
        'val_drawdown': [eval_data.get('max_drawdown', 0.0) for eval_data in callback.evaluation_history if eval_data.get('split') == 'val'],
        'best_reward': callback.best_mean_reward,
        'early_stopped': callback.early_stopped,
        'training_time': training_time
    }

    return model, history

def evaluate_super_agent(
    model: PPO,
    env,
    split: str = 'test',
    n_episodes: int = 1,
    deterministic: bool = True
) -> Dict:
    """
    Evaluate Super Agent on given environment.

    Returns:
        Dictionary with evaluation results
    """
    all_returns = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # After episode, collect all episode returns (from env.episode_returns attribute)
        episode_returns = env.episode_returns
        all_returns.extend(episode_returns)

    # Calculate metrics using all episode returns
    returns_array = np.array(all_returns)
    metrics = calculate_all_metrics(returns_array)

    # Normalize field names to match notebook expectations
    results = {
        'sharpe': float(metrics.get('sharpe_ratio', 0.0)),
        'annual_return': float(metrics.get('annual_return', 0.0)),
        'annual_volatility': float(metrics.get('annual_volatility', 0.0)),
        'max_drawdown': float(metrics.get('max_drawdown', 0.0)),
        'calmar': float(metrics.get('calmar_ratio', 0.0)),
        'sortino': float(metrics.get('sortino_ratio', 0.0)),
        'win_rate': float(metrics.get('win_rate', 0.0)),
        'total_return': float(metrics.get('total_return', 0.0)),
        'returns': returns_array.tolist(),
        'split': split
    }
    return results

def train_meta_agent(
    train_env,
    val_env,
    model_name: str = 'meta_agent',
    config=None,
    save_model: bool = True,
    verbose: int = 1
) -> Tuple[PPO, Dict]:
    """
    Train Meta Agent (Level 3 of hierarchy).

    Args:
        train_env: Training environment
        val_env: Validation environment
        model_name: Name for saved model
        config: Configuration class (default: MetaAgentConfig)
        save_model: Whether to save trained model
        verbose: Verbosity level

    Returns:
        Tuple of (trained_model, training_history)
    """
    if config is None:
        from .config import MetaAgentConfig
        config = MetaAgentConfig

    if verbose > 0:
        print(f"\nTraining Meta Agent: {model_name}")
        print("="*70)
        print(f"Config: {config.__name__}")
        print(f"Total timesteps: {config.TOTAL_TIMESTEPS:,}")

    # Create model
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        policy_kwargs=dict(
            net_arch=config.NET_ARCH,
            activation_fn=eval(f"torch.nn.{config.ACTIVATION.capitalize()}")
        ),
        verbose=0
    )

    # Create callback
    from .config import MODELS_DIR
    save_path = MODELS_DIR / f'{model_name}.zip' if save_model else None

    callback = EvaluationCallback(
        eval_env=val_env,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
        save_path=save_path,
        verbose=verbose
    )

    # Train
    start_time = time.time()
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback)
    training_time = time.time() - start_time

    if verbose > 0:
        print(f"\n✅ Training completed in {training_time:.1f}s")
        if callback.early_stopped:
            print(f"   Early stopped at step {callback.n_calls}")
        print(f"   Best mean reward: {callback.best_mean_reward:.4f}")

    # Prepare history - metrics now properly calculated by EvaluationCallback
    # Note: train evaluation not implemented yet, so train metrics will be empty
    history = {
        'train_rewards': [],  # Not tracked (callback only evaluates on val_env)
        'val_rewards': [eval_data['mean_reward'] for eval_data in callback.evaluation_history if eval_data.get('split') == 'val'],
        'train_sharpe': [],  # Not tracked
        'val_sharpe': [eval_data.get('sharpe', 0.0) for eval_data in callback.evaluation_history if eval_data.get('split') == 'val'],
        'train_returns': [],  # Not tracked
        'val_returns': [eval_data.get('annual_return', 0.0) for eval_data in callback.evaluation_history if eval_data.get('split') == 'val'],
        'train_drawdown': [],  # Not tracked
        'val_drawdown': [eval_data.get('max_drawdown', 0.0) for eval_data in callback.evaluation_history if eval_data.get('split') == 'val'],
        'best_reward': callback.best_mean_reward,
        'early_stopped': callback.early_stopped,
        'training_time': training_time
    }

    return model, history

def evaluate_meta_agent(
    model: PPO,
    env,
    split: str = 'test',
    n_episodes: int = 1,
    deterministic: bool = True
) -> Dict:
    """
    Evaluate Meta Agent on given environment.

    Returns:
        Dictionary with evaluation results
    """
    all_returns = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # After episode, collect episode returns
        episode_returns = env.episode_returns
        all_returns.extend(episode_returns)

    # Calculate metrics
    returns_array = np.array(all_returns)
    metrics = calculate_all_metrics(returns_array)

    # Normalize field names to match notebook expectations
    results = {
        'sharpe': float(metrics.get('sharpe_ratio', 0.0)),
        'annual_return': float(metrics.get('annual_return', 0.0)),
        'annual_volatility': float(metrics.get('annual_volatility', 0.0)),
        'max_drawdown': float(metrics.get('max_drawdown', 0.0)),
        'calmar': float(metrics.get('calmar_ratio', 0.0)),
        'sortino': float(metrics.get('sortino_ratio', 0.0)),
        'win_rate': float(metrics.get('win_rate', 0.0)),
        'total_return': float(metrics.get('total_return', 0.0)),
        'returns': returns_array.tolist(),
        'split': split
    }

    return results

# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    """Test training functions (requires data to be prepared)."""

    print("Training Module Test")
    print("="*70)
    print("\nNote: This requires prepared data in data_hierarchical/")
    print("Run 01_data_prep.ipynb first to generate data.\n")

    # Check if data exists
    from .config import DATA_DIR

    technical_train = DATA_DIR / 'technical' / 'train.csv'

    if technical_train.exists():
        print("✅ Data found! Running training test...")

        # Test with minimal config for speed
        class TestConfig:
            LEARNING_RATE = 3e-4
            N_STEPS = 512
            BATCH_SIZE = 64
            N_EPOCHS = 5
            GAMMA = 0.99
            GAE_LAMBDA = 0.95
            CLIP_RANGE = 0.2
            ENT_COEF = 0.01
            VF_COEF = 0.5
            MAX_GRAD_NORM = 0.5
            TOTAL_TIMESTEPS = 5000  # Very short for testing
            NET_ARCH = [64, 64]
            ACTIVATION = 'tanh'
            EVAL_FREQ = 2000
            N_EVAL_EPISODES = 2
            PATIENCE = 2
            MIN_DELTA = 0.01

            @classmethod
            def to_dict(cls):
                return {k: v for k, v in cls.__dict__.items() if not k.startswith('_')}

        # Train a small test agent
        model, history = train_base_agent(
            agent_type='technical',
            agent_name='test_agent',
            reward_type='ema_sharpe',
            config=TestConfig,
            save_model=False,
            verbose=1
        )

        print("\n✅ Training test completed!")
        print(f"   Final step: {history['total_steps']}")
        print(f"   Best reward: {history['best_reward']:.4f}")

    else:
        print("❌ Data not found. Please run 01_data_prep.ipynb first.")
        print(f"   Looking for: {technical_train}")

    print("\n" + "="*70)
