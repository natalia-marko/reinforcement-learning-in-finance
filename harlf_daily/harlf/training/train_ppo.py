"""
PPO Training Script for Portfolio Optimization

Implements walk-forward training with:
- Custom Dirichlet policy for portfolio weights
- EMA Sharpe or Multi-Objective rewards
- Early stopping and checkpointing
- TensorBoard logging
- Performance evaluation

Usage:
    python training/train_ppo.py --fold_id 0 --total_timesteps 50000

Created: 2025-01-09
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from harlf.core.envs.portfolio_env import PortfolioEnv
from harlf.core.agents.dirichlet_policy import SoftmaxActorCriticPolicy
from harlf.training.callbacks import (
    PortfolioWeightsLogger,
    BestModelCallback,
    MetricsTrackingCallback,
    PeriodicCheckpointCallback,
)
from harlf.config.defaults import TICKERS


def make_env(fold_id: int, split: str, reward_type: str = 'ema_sharpe', **reward_kwargs):
    """
    Create and wrap environment.

    Args:
        fold_id: Fold number for walk-forward validation
        split: Data split ('train', 'val', 'test')
        reward_type: Reward function type
        **reward_kwargs: Additional kwargs for reward function

    Returns:
        Wrapped environment
    """
    env = PortfolioEnv(
        fold_id=fold_id,
        split=split,
        reward_type=reward_type,
        reward_kwargs=reward_kwargs,
    )
    env = Monitor(env)  # Wrap with Monitor for episode tracking
    return env


def train_ppo(
    fold_id: int = 0,
    total_timesteps: int = 50000,
    reward_type: str = 'ema_sharpe',
    learning_rate: float = 2.5e-5,
    gamma: float = 0.985,
    ent_coef: float = 0.001,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    n_steps: int = 2048,
    batch_size: int = 128,
    n_epochs: int = 10,
    clip_range: float = 0.2,
    net_arch: list = [256, 256],
    tensorboard_log: str = "./tensorboard_logs",
    save_path: str = "./models",
    checkpoint_freq: int = 10000,
    verbose: int = 1,
    seed: int = 42,
):
    """
    Train PPO agent on a single fold.

    Args:
        fold_id: Fold number (0-49)
        total_timesteps: Total training timesteps
        reward_type: Reward function ('ema_sharpe', 'multi_objective', 'simple')
        learning_rate: Learning rate
        gamma: Discount factor
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
        n_steps: Number of steps per update
        batch_size: Minibatch size
        n_epochs: Number of optimization epochs
        clip_range: PPO clipping range
        net_arch: Network architecture (list of hidden layer sizes)
        tensorboard_log: TensorBoard log directory
        save_path: Model save directory
        checkpoint_freq: Checkpoint frequency (in timesteps)
        verbose: Verbosity level
        seed: Random seed

    Returns:
        Trained model, training info
    """
    print("="*80)
    print(f"TRAINING PPO - FOLD {fold_id}")
    print("="*80)
    print(f"Reward type: {reward_type}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Network: {net_arch}")
    print("="*80)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environments
    print("\nCreating environments...")
    train_env = DummyVecEnv([lambda: make_env(fold_id, 'train', reward_type)])
    val_env = make_env(fold_id, 'val', reward_type)

    print(f"Train environment: {train_env.observation_space.shape}")
    print(f"Action space: {train_env.action_space.shape}")

    # Reward function kwargs
    reward_kwargs = {}
    if reward_type == 'ema_sharpe':
        reward_kwargs = {'window': 21, 'risk_free_rate': 0.02}
    elif reward_type == 'multi_objective':
        reward_kwargs = {
            'w_return': 1.0,
            'w_risk': 0.5,
            'w_alpha': 0.3,
            'w_turnover': 0.2,
        }

    # Policy kwargs
    policy_kwargs = {
        'net_arch': net_arch,
        'activation_fn': torch.nn.Tanh,
        'ortho_init': True,
    }

    # Create model
    print("\nCreating PPO model...")
    model = PPO(
        policy=SoftmaxActorCriticPolicy,
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=clip_range,
        clip_range_vf=100.0,  # Clip value function updates to prevent explosions
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=seed,
    )

    # Setup callbacks
    save_path = Path(save_path) / f"fold_{fold_id}"
    save_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        PortfolioWeightsLogger(log_freq=100, verbose=0),
        MetricsTrackingCallback(log_freq=1, verbose=1),
        PeriodicCheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=save_path / "checkpoints",
            name_prefix=f"ppo_fold_{fold_id}",
            verbose=1
        ),
        BestModelCallback(
            save_path=save_path,
            metric='sharpe_ratio',
            verbose=1
        ),
    ]

    # Train
    print("\nStarting training...")
    start_time = datetime.now()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"PPO_fold_{fold_id}_{reward_type}",
        progress_bar=True,
    )

    train_time = (datetime.now() - start_time).total_seconds()
    print(f"\nTraining completed in {train_time:.2f} seconds")

    # Save final model
    final_model_path = save_path / f"ppo_fold_{fold_id}_final.zip"
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, val_env, n_episodes=1, deterministic=True)

    print("\nValidation Metrics:")
    print("-" * 40)
    for key, value in val_metrics.items():
        if isinstance(value, float):
            print(f"{key:.<30} {value:.4f}")
        else:
            print(f"{key:.<30} {value}")

    # Cleanup
    train_env.close()
    val_env.close()

    return model, {
        'fold_id': fold_id,
        'train_time': train_time,
        'total_timesteps': total_timesteps,
        'val_metrics': val_metrics,
    }


def evaluate_model(model, env, n_episodes: int = 1, deterministic: bool = True):
    """
    Evaluate trained model on an environment.

    Args:
        model: Trained PPO model
        env: Environment to evaluate on
        n_episodes: Number of episodes to run
        deterministic: Use deterministic actions

    Returns:
        Dictionary of evaluation metrics
    """
    all_metrics = []

    # Unwrap Monitor if needed
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)

        # Get episode metrics from unwrapped environment
        if hasattr(unwrapped_env, 'get_episode_metrics'):
            episode_metrics = unwrapped_env.get_episode_metrics()
            all_metrics.append(episode_metrics)

    if len(all_metrics) == 0:
        return {'error': 'No metrics collected'}

    # Average across episodes
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)

    return avg_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train PPO for Portfolio Optimization')

    # Required arguments
    parser.add_argument('--fold_id', type=int, default=0, help='Fold ID (0-49)')

    # Training arguments
    parser.add_argument('--total_timesteps', type=int, default=50000, help='Total training timesteps')
    parser.add_argument('--reward_type', type=str, default='ema_sharpe',
                        choices=['ema_sharpe', 'multi_objective', 'simple'],
                        help='Reward function type')

    # PPO hyperparameters
    parser.add_argument('--learning_rate', type=float, default=2.5e-5, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.985, help='Discount factor')
    parser.add_argument('--ent_coef', type=float, default=0.001, help='Entropy coefficient')
    parser.add_argument('--n_steps', type=int, default=2048, help='Steps per update')
    parser.add_argument('--batch_size', type=int, default=128, help='Minibatch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Optimization epochs')

    # Paths
    parser.add_argument('--tensorboard_log', type=str, default='./tensorboard_logs', help='TensorBoard log dir')
    parser.add_argument('--save_path', type=str, default='./models', help='Model save directory')

    # Other
    parser.add_argument('--checkpoint_freq', type=int, default=10000, help='Checkpoint frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')

    args = parser.parse_args()

    # Train
    model, info = train_ppo(
        fold_id=args.fold_id,
        total_timesteps=args.total_timesteps,
        reward_type=args.reward_type,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        tensorboard_log=args.tensorboard_log,
        save_path=args.save_path,
        checkpoint_freq=args.checkpoint_freq,
        verbose=args.verbose,
        seed=args.seed,
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Fold {info['fold_id']}: Sharpe={info['val_metrics']['sharpe_ratio']:.4f}, "
          f"Return={info['val_metrics']['total_return']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
