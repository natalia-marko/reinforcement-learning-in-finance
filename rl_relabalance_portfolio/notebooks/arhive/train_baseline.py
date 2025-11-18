"""
Training script for baseline SimpleActor model with walking forward validation.
Uses stable-baselines3 PPO algorithm.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
import torch
import torch.nn as nn
from typing import Dict
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from portfolio_env import PortfolioEnv
from walk_forward_validation import create_walk_forward_folds, print_fold_summary


def train_fold(
    fold_idx: int,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    feature_cols: list,
    tickers: list,
    output_dir: Path,
    total_timesteps: int = 100000,
    eval_freq: int = 5000,
    n_eval_episodes: int = 10
) -> Dict:
    """
    Train PPO agent on a single fold.
    
    Parameters:
    -----------
    fold_idx : int
        Fold index
    train_data : pd.DataFrame
        Training data for this fold
    val_data : pd.DataFrame
        Validation data for this fold
    feature_cols : list
        List of feature column names
    tickers : list
        List of asset tickers
    output_dir : Path
        Directory to save model and results
    total_timesteps : int
        Total training timesteps (default: 100000)
    eval_freq : int
        Evaluation frequency (default: 5000)
    n_eval_episodes : int
        Number of episodes for evaluation (default: 10)
    
    Returns:
    --------
    Dict
        Training results and metrics
    """
    print(f"\n{'='*80}")
    print(f"TRAINING FOLD {fold_idx}")
    print(f"{'='*80}")
    
    # Create environments
    def make_train_env():
        env = PortfolioEnv(
            data=train_data,
            feature_cols=feature_cols,
            tickers=tickers,
            rebalance_frequency=4,
            transaction_cost=0.00025,
            max_weight_per_asset=0.4,
            reward_lookback=4,
            seed=42 + fold_idx
        )
        return Monitor(env)
    
    def make_val_env():
        env = PortfolioEnv(
            data=val_data,
            feature_cols=feature_cols,
            tickers=tickers,
            rebalance_frequency=4,
            transaction_cost=0.00025,
            max_weight_per_asset=0.4,
            reward_lookback=4,
            seed=42 + fold_idx
        )
        return Monitor(env)
    
    train_env = DummyVecEnv([make_train_env])
    val_env = DummyVecEnv([make_val_env])
    
    # Create model directory
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup early stopping callback
    # Patience=50 episodes means wait 50 evaluation periods before stopping
    # Stops if validation Sharpe ratio doesn't improve for 50 consecutive evaluations
    # Since eval_freq=5000, this means 50 * 5000 = 250,000 timesteps of patience
    early_stopping = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=50,  # Patience=50 episodes as per architecture
        min_evals=10,  # Minimum evaluations before early stopping can trigger
        verbose=1
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(fold_dir / "best_model"),
        log_path=str(fold_dir / "logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        callback_on_new_best=early_stopping  # Early stopping based on validation performance
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=str(fold_dir / "checkpoints"),
        name_prefix="ppo_model"
    )
    
    # Create PPO model with SimpleActor architecture
    # Policy network: [256, 128] layers matching SimpleActor from models.py
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),  # pi=actor, vf=critic
        activation_fn=nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(fold_dir / "tensorboard")
    )
    
    # Train model
    print(f"\nTraining for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(str(fold_dir / "final_model"))
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_episode_rewards = []
    val_episode_lengths = []

    # Run for exactly n_eval_episodes complete episodes
    for episode_idx in range(n_eval_episodes):
        val_obs = val_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(val_obs, deterministic=True)
            val_obs, reward, done, info = val_env.step(action)
            episode_reward += reward[0]
            episode_length += 1

            if done[0]:
                done = True

        val_episode_rewards.append(episode_reward)
        val_episode_lengths.append(episode_length)

    # Get final portfolio stats from the last episode
    # Access the underlying environment through Monitor wrapper
    val_env_unwrapped = val_env.envs[0].env if hasattr(val_env.envs[0], 'env') else val_env.envs[0]
    val_stats = val_env_unwrapped.get_portfolio_stats() if hasattr(val_env_unwrapped, 'get_portfolio_stats') else {}
    
    results = {
        'fold_idx': fold_idx,
        'val_mean_reward': float(np.mean(val_episode_rewards)) if val_episode_rewards else 0.0,
        'val_std_reward': float(np.std(val_episode_rewards)) if val_episode_rewards else 0.0,
        'val_total_return': val_stats['total_return'],
        'val_sharpe_ratio': val_stats['sharpe_ratio'],
        'val_volatility': val_stats['volatility'],
        'val_max_drawdown': val_stats['max_drawdown'],
        'val_portfolio_value': val_stats['portfolio_value'],
        'model_path': str(fold_dir / "best_model")
    }
    
    print(f"\nFold {fold_idx} Results:")
    print(f"  Validation Mean Reward: {results['val_mean_reward']:.4f} ± {results['val_std_reward']:.4f}")
    print(f"  Validation Sharpe Ratio: {results['val_sharpe_ratio']:.4f}")
    print(f"  Validation Total Return: {results['val_total_return']:.4f}")
    print(f"  Validation Max Drawdown: {results['val_max_drawdown']:.4f}")
    
    return results


def main():
    """Main training function."""
    # Configuration
    DATA_DIR = project_root / 'data' / 'processed'
    OUTPUT_DIR = project_root / 'models' / 'baseline'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_parquet(DATA_DIR / 'train.parquet')
    test_df = pd.read_parquet(DATA_DIR / 'test.parquet')
    
    with open(DATA_DIR / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    feature_cols = metadata['feature_cols']
    tickers = metadata['tickers']
    
    print(f"Loaded {len(train_df)} training samples")
    print(f"Features: {len(feature_cols)}")
    print(f"Tickers: {tickers}")
    
    # Create walking forward folds
    print("\nCreating walking forward validation folds...")
    folds = create_walk_forward_folds(
        train_data=train_df,
        n_folds=5,
        min_train_size=None,
        date_col='date'
    )
    print_fold_summary(folds)
    
    # Train on each fold
    all_results = []
    
    for fold in folds:
        results = train_fold(
            fold_idx=fold['fold_idx'],
            train_data=fold['train'],
            val_data=fold['val'],
            feature_cols=feature_cols,
            tickers=tickers,
            output_dir=OUTPUT_DIR,
            total_timesteps=50000,
            eval_freq=5000,
            n_eval_episodes=10
        )
        all_results.append(results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'fold_results.csv', index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"\nAverage Validation Sharpe Ratio: {results_df['val_sharpe_ratio'].mean():.4f}")
    print(f"Average Validation Total Return: {results_df['val_total_return'].mean():.4f}")
    print(f"Average Validation Max Drawdown: {results_df['val_max_drawdown'].mean():.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR / 'fold_results.csv'}")
    print("\nNote: Test data is kept untouched for final out-of-sample evaluation.")


if __name__ == "__main__":
    main()

