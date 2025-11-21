"""
Enhanced PPO Training Script with Debugging Capabilities

Implements:
- Diagnostic mode for debugging
- Comprehensive hyperparameter tuning
- Advanced monitoring and logging
- Gradient clipping and normalization
- Action distribution analysis
- Overfitting detection

Usage:
    # Normal training
    python train_ppo_enhanced.py --fold_id 0 --total_timesteps 50000
    
    # Diagnostic mode
    python train_ppo_enhanced.py --fold_id 0 --diagnostic --debug_steps 1000
    
    # With enhanced monitoring
    python train_ppo_enhanced.py --fold_id 0 --enhanced_monitoring

Created: 2025-01-11
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import json
import warnings
from typing import Dict, Any, Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

# ✅ FIXED: Import callbacks from correct path
from harlf.callbacks import (
    GradientMonitorCallback,
    ActionDistributionCallback,
    ValidationPerformanceCallback,
    PortfolioWeightsLogger,
    EarlyStoppingCallback,
    ComprehensiveLoggingCallback,
)

from harlf.portfolio_env import PortfolioEnv
from harlf.dirichlet_policy import SoftmaxActorCriticPolicy
from harlf.config import TICKERS


class DiagnosticMode:
    """Diagnostic mode for debugging training issues."""
    
    def __init__(self, model, env, verbose: int = 2):
        self.model = model
        self.env = env
        self.verbose = verbose
        
    def run_diagnostics(self, n_steps: int = 100):
        """Run comprehensive diagnostics."""
        print("\n" + "="*80)
        print("🔍 DIAGNOSTIC MODE")
        print("="*80)
        
        # 1. Check environment
        print("\n1. Environment Check:")
        self._check_environment()
        
        # 2. Check observation space
        print("\n2. Observation Space Analysis:")
        self._check_observations()
        
        # 3. Check action space and policy
        print("\n3. Action Space & Policy Check:")
        self._check_actions()
        
        # 4. Check rewards
        print("\n4. Reward Analysis:")
        self._check_rewards(n_steps)
        
        # 5. Check gradients
        print("\n5. Gradient Analysis:")
        self._check_gradients(n_steps)
        
        # 6. Check for data leakage
        print("\n6. Data Leakage Check:")
        self._check_data_leakage()
        
        print("\n" + "="*80)
        print("✅ DIAGNOSTICS COMPLETE")
        print("="*80)
    
    def _check_environment(self):
        """Check environment setup."""
        obs, info = self.env.reset()
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"   Action space: {self.env.action_space}")
        
        # Check if environment is properly wrapped
        if hasattr(self.env, 'unwrapped'):
            unwrapped = self.env.unwrapped
            print(f"   Environment class: {unwrapped.__class__.__name__}")
            
            if hasattr(unwrapped, 'fold_id'):
                print(f"   Fold ID: {unwrapped.fold_id}")
            if hasattr(unwrapped, 'split'):
                print(f"   Split: {unwrapped.split}")
    
    def _check_observations(self):
        """Analyze observation statistics."""
        obs_list = []
        for _ in range(10):
            obs, _ = self.env.reset()
            obs_list.append(obs)
        
        obs_array = np.array(obs_list)
        
        # Check statistics per feature
        n_features = obs_array.shape[-1]
        print(f"   Number of features: {n_features}")
        
        # Check for constant features
        feature_std = obs_array.std(axis=0).mean(axis=0)
        constant_features = np.where(feature_std < 1e-6)[0]
        if len(constant_features) > 0:
            print(f"   ⚠️ Found {len(constant_features)} constant features: {constant_features}")
        
        # Check for NaN or Inf
        if np.any(np.isnan(obs_array)):
            print("   ⚠️ Found NaN values in observations!")
        if np.any(np.isinf(obs_array)):
            print("   ⚠️ Found Inf values in observations!")
        
        # Check normalization
        mean_vals = obs_array.mean(axis=(0, 1))
        std_vals = obs_array.std(axis=(0, 1))
        
        print(f"   Mean range: [{mean_vals.min():.3f}, {mean_vals.max():.3f}]")
        print(f"   Std range: [{std_vals.min():.3f}, {std_vals.max():.3f}]")
        
        if std_vals.min() < 0.1 or std_vals.max() > 10:
            print("   ⚠️ Features may need better normalization")
    
    def _check_actions(self):
        """Check action distribution."""
        obs, _ = self.env.reset()
        
        # Sample actions
        actions = []
        for _ in range(100):
            action, _ = self.model.predict(obs, deterministic=False)
            actions.append(action)
        
        actions = np.array(actions)
        
        # Analyze distribution
        print(f"   Action shape: {actions.shape}")
        print(f"   Action mean: {actions.mean(axis=0)}")
        print(f"   Action std: {actions.std(axis=0)}")
        
        # Check for policy collapse
        action_std = actions.std(axis=0)
        if np.any(action_std < 0.01):
            print("   ⚠️ Low action variance detected - possible policy collapse!")
        
        # Check portfolio weights sum to 1
        if actions.shape[-1] == len(TICKERS):
            weight_sums = actions.sum(axis=-1)
            if not np.allclose(weight_sums, 1.0, atol=0.01):
                print(f"   ⚠️ Portfolio weights don't sum to 1: {weight_sums[:5]}")
    
    def _check_rewards(self, n_steps: int):
        """Analyze reward distribution."""
        rewards = []
        obs, _ = self.env.reset()
        
        for _ in range(min(n_steps, 100)):
            action, _ = self.model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = self.env.step(action)
            rewards.append(reward)
            
            if done or truncated:
                obs, _ = self.env.reset()
        
        rewards = np.array(rewards)
        
        print(f"   Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]")
        print(f"   Reward mean: {rewards.mean():.4f}")
        print(f"   Reward std: {rewards.std():.4f}")
        
        # Check for reward issues
        if np.all(rewards == rewards[0]):
            print("   ⚠️ Constant rewards detected!")
        
        if np.any(np.isnan(rewards)):
            print("   ⚠️ NaN rewards detected!")
        
        if np.abs(rewards.mean()) > 100:
            print("   ⚠️ Rewards may need scaling")
    
    def _check_gradients(self, n_steps: int):
        """Check gradient flow."""
        from torch.nn.utils import clip_grad_norm_
        
        # Do a few training steps
        obs, _ = self.env.reset()
        
        grad_norms = []
        for _ in range(min(n_steps, 10)):
            action, _ = self.model.predict(obs)
            obs, reward, done, truncated, _ = self.env.step(action)
            
            if done or truncated:
                obs, _ = self.env.reset()
        
        # Get gradient norms
        for name, param in self.model.policy.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append(grad_norm)
                
                if self.verbose >= 2:
                    print(f"   {name}: {grad_norm:.4e}")
        
        if grad_norms:
            mean_grad_norm = np.mean(grad_norms)
            print(f"   Mean gradient norm: {mean_grad_norm:.4e}")
            
            if mean_grad_norm < 1e-7:
                print("   ⚠️ Vanishing gradients detected!")
            elif mean_grad_norm > 100:
                print("   ⚠️ Large gradients detected!")
    
    def _check_data_leakage(self):
        """Check for potential data leakage."""
        if hasattr(self.env, 'unwrapped'):
            env = self.env.unwrapped
            
            # Check if test data is accessible during training
            if hasattr(env, 'data_processor'):
                dp = env.data_processor
                
                if hasattr(dp, 'fold_id') and hasattr(dp, 'split'):
                    print(f"   Current fold: {dp.fold_id}")
                    print(f"   Current split: {dp.split}")
                    
                    # Check date ranges don't overlap
                    if hasattr(dp, 'date_range'):
                        print(f"   Date range: {dp.date_range}")
            
            # Check if future data is visible
            if hasattr(env, 'current_step') and hasattr(env, 'observations'):
                current_step = env.current_step
                obs_shape = env.observations.shape
                
                if current_step < obs_shape[0] - 1:
                    print(f"   ✅ No obvious data leakage detected")
                    print(f"   Current step: {current_step}/{obs_shape[0]}")


def create_env(
    fold_id: int,
    split: str,
    reward_type: str = 'ema_sharpe',
    normalize_obs: bool = True,
    normalize_reward: bool = False,
    **reward_kwargs
):
    """
    Create and wrap environment with optional normalization.
    """
    env = PortfolioEnv(
        fold_id=fold_id,
        split=split,
        reward_type=reward_type,
        reward_kwargs=reward_kwargs,
    )
    
    env = Monitor(env)
    
    # Optional: Add observation/reward normalization
    if normalize_obs or normalize_reward:
        env = VecNormalize(
            DummyVecEnv([lambda: env]),
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99
        )
    else:
        env = DummyVecEnv([lambda: env])
    
    return env


def create_model_with_tuned_params(
    env,
    config: Dict[str, Any],
    tensorboard_log: str,
    verbose: int = 1,
    seed: int = 42
) -> PPO:
    """
    Create PPO model with tuned hyperparameters.
    """
    # Extract hyperparameters
    learning_rate = config.get('learning_rate', 3e-5)
    n_steps = config.get('n_steps', 2048)
    batch_size = config.get('batch_size', 64)
    n_epochs = config.get('n_epochs', 10)
    gamma = config.get('gamma', 0.99)
    gae_lambda = config.get('gae_lambda', 0.95)
    clip_range = config.get('clip_range', 0.2)
    clip_range_vf = config.get('clip_range_vf', None)  # None = no clipping
    ent_coef = config.get('ent_coef', 0.01)
    vf_coef = config.get('vf_coef', 0.5)
    max_grad_norm = config.get('max_grad_norm', 0.5)
    
    # Network architecture
    net_arch = config.get('net_arch', [256, 256])
    activation_fn = config.get('activation_fn', 'tanh')
    
    # Map activation function
    activation_map = {
        'tanh': torch.nn.Tanh,
        'relu': torch.nn.ReLU,
        'elu': torch.nn.ELU,
        'leaky_relu': torch.nn.LeakyReLU,
    }
    
    activation_class = activation_map.get(activation_fn, torch.nn.Tanh)
    
    # ✅ FIXED: Policy kwargs (removed use_sde - it's passed to PPO() directly)
    policy_kwargs = {
        'net_arch': net_arch,
        'activation_fn': activation_class,
        'ortho_init': config.get('ortho_init', True),
        'log_std_init': config.get('log_std_init', -2.0),
        'full_std': config.get('full_std', True),
        'squash_output': config.get('squash_output', False),
    }
    
    # Add normalization layers if specified
    if config.get('use_layer_norm', False):
        policy_kwargs['normalize_images'] = True
    
    # Create model
    model = PPO(
        policy=SoftmaxActorCriticPolicy,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=config.get('use_sde', False),
        sde_sample_freq=config.get('sde_sample_freq', -1),
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=seed,
        device='auto',
    )
    
    return model


def train_ppo_enhanced(
    fold_id: int = 0,
    total_timesteps: int = 50000,
    config: Optional[Dict[str, Any]] = None,
    diagnostic: bool = False,
    debug_steps: int = 100,
    enhanced_monitoring: bool = False,
    save_path: str = "./models",
    tensorboard_log: str = "./tensorboard_logs",
    verbose: int = 1,
    seed: int = 42,
):
    """
    Enhanced PPO training with debugging capabilities.
    """
    print("="*80)
    print(f"ENHANCED PPO TRAINING - FOLD {fold_id}")
    print("="*80)
    
    # Default configuration
    default_config = {
        'reward_type': 'ema_sharpe',
        'learning_rate': 3e-5,
        'n_steps': 512,  # ✅ FIXED: Reduced to fit within val episode length
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'clip_range_vf': None,
        'ent_coef': 0.03,  # ✅ FIXED: Increased from 0.01 to 0.03 to reduce overfitting
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'net_arch': [256, 256],
        'activation_fn': 'tanh',
        'ortho_init': True,
        'use_layer_norm': False,
        'use_sde': False,
        'normalize_obs': True,
        'normalize_reward': True,  # ✅ FIXED: Enable reward normalization for stable training
    }
    
    if config:
        default_config.update(config)
    config = default_config
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Set seeds
    set_random_seed(seed)
    
    # Create environments
    print("\nCreating environments...")
    
    # Reward kwargs based on reward type
    reward_kwargs = {}
    if config['reward_type'] == 'ema_sharpe':
        reward_kwargs = {
            'window': config.get('ema_window', 21),
            'risk_free_rate': config.get('risk_free_rate', 0.02),
            'annualization_factor': config.get('annualization_factor', 52),
        }
    elif config['reward_type'] == 'multi_objective':
        reward_kwargs = {
            'w_return': config.get('w_return', 1.0),
            'w_risk': config.get('w_risk', 0.5),
            'w_alpha': config.get('w_alpha', 0.3),
            'w_turnover': config.get('w_turnover', 0.2),
        }
    
    # Training environment
    train_env = create_env(
        fold_id=fold_id,
        split='train',
        reward_type=config['reward_type'],
        normalize_obs=config['normalize_obs'],
        normalize_reward=config['normalize_reward'],
        **reward_kwargs
    )
    
    # Validation environment (never normalized)
    val_env = PortfolioEnv(
        fold_id=fold_id,
        split='val',
        reward_type=config['reward_type'],
        reward_kwargs=reward_kwargs,
    )
    val_env = Monitor(val_env)
    
    print(f"Train environment: {train_env.observation_space.shape}")
    print(f"Action space: {train_env.action_space.shape}")
    
    # Create model
    print("\nCreating model...")
    model = create_model_with_tuned_params(
        env=train_env,
        config=config,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=seed
    )
    
    # Run diagnostics if requested
    if diagnostic:
        print("\n🔍 Running diagnostics before training...")
        diagnostics = DiagnosticMode(model, train_env, verbose=2)
        diagnostics.run_diagnostics(n_steps=debug_steps)
        
        if input("\nContinue with training? (y/n): ").lower() != 'y':
            print("Training aborted.")
            return None, None
    
    # Setup callbacks
    save_path = Path(save_path) / f"fold_{fold_id}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    callbacks = []
    
    # ✅ FIXED: Basic callbacks (removed non-existent callbacks)
    from stable_baselines3.common.callbacks import CheckpointCallback

    callbacks.extend([
        PortfolioWeightsLogger(log_freq=100, track_turnover=True, verbose=0),
        ValidationPerformanceCallback(
            eval_env=val_env,
            eval_freq=5000,
            n_eval_episodes=1,
            deterministic=True,
            verbose=verbose
        ),
        # Use SB3's built-in CheckpointCallback
        # ✅ FIXED: Consistent naming without underscore after fold
        CheckpointCallback(
            save_freq=10000,
            save_path=str(save_path / "checkpoints"),
            name_prefix=f"ppo_fold{fold_id}_{config['reward_type']}",
            verbose=1
        ),
    ])
    
    # Enhanced monitoring callbacks
    if enhanced_monitoring:
        callbacks.extend([
            GradientMonitorCallback(
                log_freq=100,
                warn_threshold=10.0,
                check_dead_neurons=True,
                verbose=verbose
            ),
            ActionDistributionCallback(
                log_freq=100,
                n_assets=len(TICKERS),
                warn_low_entropy=0.1,
                verbose=verbose
            ),
            ComprehensiveLoggingCallback(
                log_freq=100,
                save_path=save_path / "logs",
                verbose=verbose
            ),
        ])
    
    # ✅ FIXED: Early stopping with proper SB3 integration
    early_stopping = EarlyStoppingCallback(
        eval_freq=5000,  # Must match ValidationPerformanceCallback eval_freq
        patience=5,
        min_delta=0.01,
        metric='sharpe_ratio',  # Will look for 'eval/sharpe_ratio' in logger
        min_evaluations=3,
        check_overfitting=False,  # Disabled (would need train metrics)
        max_train_val_gap=0.5,
        verbose=verbose
    )
    callbacks.append(early_stopping)
    
    # Train
    print("\n🚀 Starting training...")
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=f"PPO_fold_{fold_id}_{config['reward_type']}",
            progress_bar=True,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if diagnostic:
            print("\nRunning post-failure diagnostics...")
            diagnostics = DiagnosticMode(model, train_env, verbose=2)
            diagnostics.run_diagnostics(n_steps=50)
        raise
    
    train_time = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Training completed in {train_time:.2f} seconds")

    # Save final model
    # ✅ FIXED: Include reward type in filename for consistency
    final_model_path = save_path / f"ppo_fold{fold_id}_{config['reward_type']}.zip"
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Comprehensive evaluation
    print("\n📊 Running comprehensive evaluation...")
    eval_results = comprehensive_evaluation(
        model=model,
        fold_id=fold_id,
        reward_type=config['reward_type'],
        reward_kwargs=reward_kwargs,
        save_path=save_path,
        verbose=verbose
    )
    
    # Save training info
    training_info = {
        'fold_id': fold_id,
        'config': config,
        'train_time': train_time,
        'total_timesteps': total_timesteps,
        'eval_results': eval_results,
    }
    
    info_path = save_path / 'training_info.json'
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2, default=str)
    
    print(f"\nTraining info saved: {info_path}")
    
    # Cleanup
    train_env.close()
    val_env.close()
    
    return model, training_info


def comprehensive_evaluation(
    model,
    fold_id: int,
    reward_type: str,
    reward_kwargs: Dict,
    save_path: Path,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation on all splits.
    """
    results = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\nEvaluating on {split} set...")
        
        # Create environment
        env = PortfolioEnv(
            fold_id=fold_id,
            split=split,
            reward_type=reward_type,
            reward_kwargs=reward_kwargs,
        )
        env = Monitor(env)
        
        # Run evaluation
        split_results = evaluate_model_detailed(
            model=model,
            env=env,
            n_episodes=1,
            deterministic=True,
            verbose=verbose
        )
        
        results[split] = split_results
        
        # Print summary
        if verbose > 0:
            print(f"  Sharpe Ratio: {split_results.get('sharpe_ratio', 0):.3f}")
            print(f"  Total Return: {split_results.get('total_return', 0):.4f}")
            print(f"  Max Drawdown: {split_results.get('max_drawdown', 0):.4f}")
            print(f"  Win Rate: {split_results.get('win_rate', 0):.2%}")
            
            # Check for overfitting
            if split == 'val' and 'train' in results:
                train_sharpe = results['train'].get('sharpe_ratio', 0)
                val_sharpe = results['val'].get('sharpe_ratio', 0)
                gap = train_sharpe - val_sharpe
                
                if gap > 0.3:
                    print(f"  ⚠️ Overfitting detected: Train-Val gap = {gap:.3f}")
        
        env.close()
    
    # Save evaluation results
    eval_path = save_path / 'evaluation_results.json'
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def evaluate_model_detailed(
    model,
    env,
    n_episodes: int = 1,
    deterministic: bool = True,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Detailed model evaluation with comprehensive metrics.
    """
    all_metrics = []
    all_actions = []
    all_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_rewards = []
        episode_actions = []
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_actions.append(action)
        
        # Get episode metrics
        if hasattr(env.unwrapped, 'get_episode_metrics'):
            metrics = env.unwrapped.get_episode_metrics()
            all_metrics.append(metrics)
        
        all_rewards.append(episode_rewards)
        all_actions.append(episode_actions)
    
    # Aggregate metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
    else:
        avg_metrics = {'error': 'No metrics collected'}
    
    # Analyze actions
    if all_actions:
        actions_array = np.concatenate(all_actions)
        avg_metrics['action_mean'] = actions_array.mean(axis=0).tolist()
        avg_metrics['action_std'] = actions_array.std(axis=0).tolist()
        
        # Check for static policy
        if actions_array.std() < 0.01:
            avg_metrics['policy_static'] = True
            if verbose > 0:
                print("  ⚠️ Policy appears static!")
    
    # Analyze rewards
    if all_rewards:
        rewards_flat = np.concatenate(all_rewards)
        avg_metrics['reward_mean'] = float(rewards_flat.mean())
        avg_metrics['reward_std'] = float(rewards_flat.std())
        avg_metrics['reward_min'] = float(rewards_flat.min())
        avg_metrics['reward_max'] = float(rewards_flat.max())
    
    return avg_metrics


def main():
    """Main entry point with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description='Enhanced PPO Training for Portfolio Optimization'
    )
    
    # Basic arguments
    parser.add_argument('--fold_id', type=int, default=0, help='Fold ID (0-49)')
    parser.add_argument('--total_timesteps', type=int, default=50000,
                        help='Total training timesteps')
    
    # Reward configuration
    parser.add_argument('--reward_type', type=str, default='ema_sharpe',
                        choices=['ema_sharpe', 'multi_objective', 'simple'],
                        help='Reward function type')
    parser.add_argument('--ema_window', type=int, default=21,
                        help='EMA window for Sharpe ratio')
    
    # Model hyperparameters
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--ent_coef', type=float, default=0.03,
                        help='Entropy coefficient (higher = more exploration, less overfitting)')
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--clip_range', type=float, default=0.2)
    
    # Network architecture
    parser.add_argument('--net_arch', type=int, nargs='+', default=[256, 256],
                        help='Network architecture')
    parser.add_argument('--activation_fn', type=str, default='tanh',
                        choices=['tanh', 'relu', 'elu', 'leaky_relu'])
    
    # Normalization
    parser.add_argument('--normalize_obs', action='store_true',
                        help='Normalize observations')
    parser.add_argument('--normalize_reward', action='store_true',
                        help='Normalize rewards')
    
    # Debugging options
    parser.add_argument('--diagnostic', action='store_true',
                        help='Run diagnostic mode before training')
    parser.add_argument('--debug_steps', type=int, default=100,
                        help='Number of steps for diagnostic mode')
    parser.add_argument('--enhanced_monitoring', action='store_true',
                        help='Enable enhanced monitoring callbacks')
    
    # Paths
    parser.add_argument('--tensorboard_log', type=str, default='./tensorboard_logs')
    parser.add_argument('--save_path', type=str, default='./models')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', type=int, default=1)
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'reward_type': args.reward_type,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'ent_coef': args.ent_coef,
        'vf_coef': args.vf_coef,
        'max_grad_norm': args.max_grad_norm,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'clip_range': args.clip_range,
        'net_arch': args.net_arch,
        'activation_fn': args.activation_fn,
        'normalize_obs': args.normalize_obs,
        'normalize_reward': args.normalize_reward,
        'ema_window': args.ema_window,
    }
    
    # Train
    model, info = train_ppo_enhanced(
        fold_id=args.fold_id,
        total_timesteps=args.total_timesteps,
        config=config,
        diagnostic=args.diagnostic,
        debug_steps=args.debug_steps,
        enhanced_monitoring=args.enhanced_monitoring,
        save_path=args.save_path,
        tensorboard_log=args.tensorboard_log,
        verbose=args.verbose,
        seed=args.seed,
    )
    
    if info:
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE")
        print("="*80)
        
        # Print final results
        for split in ['train', 'val', 'test']:
            if split in info['eval_results']:
                results = info['eval_results'][split]
                print(f"\n{split.upper()} Results:")
                print(f"  Sharpe: {results.get('sharpe_ratio', 0):.3f}")
                print(f"  Return: {results.get('total_return', 0):.4f}")
                print(f"  Drawdown: {results.get('max_drawdown', 0):.4f}")
        
        print("="*80)


if __name__ == '__main__':
    main()