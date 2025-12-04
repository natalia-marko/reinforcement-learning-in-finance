"""
Quick A/B/C/D test for n_steps parameter optimization

Tests three configurations:
- A: n_steps=615 (5 episodes × 123)  
- B: n_steps=738 (6 episodes × 123)
- C: n_steps=984 (8 episodes × 123)
"""

import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.rl_system import PortfolioEnv
from core.data_loading_preprocessing import create_features, get_lean_features
from core.config import *

def test_n_steps_config(n_steps, test_name, timesteps=10000):
    """Train a quick model with specific n_steps and evaluate"""
    print(f"\n{'='*60}")
    print(f"TEST {test_name}: n_steps={n_steps}")
    print(f"{'='*60}")
    
    # Load data
    paths = get_data_paths(expanded_mode=True)
    raw_data = pd.read_csv(paths['train'], index_col=0, parse_dates=True, header=[0, 1])
    raw_prices = raw_data['prices']
    
    # Take first 60% for quick test
    split_idx = int(len(raw_prices) * 0.6)
    train_prices = raw_prices.iloc[:split_idx]
    val_prices = raw_prices.iloc[split_idx:]
    
    # Create features
    train_features = create_features(train_prices, None, qqq=None, lean=False, drop_zero_var=False, verbose=False)
    train_features = get_lean_features(train_features, verbose=False)
    
    val_features = create_features(val_prices, None, qqq=None, lean=False, drop_zero_var=False, verbose=False)
    val_features = get_lean_features(val_features, verbose=False)
    
    print(f"Train features: {train_features.shape}")
    print(f"Val features: {val_features.shape}")
    
    # Calculate optimal batch_size (should divide n_steps evenly)
    # Aim for 6-8 mini-batches
    if n_steps == 615:
        batch_size = 123  # 5 batches
    elif n_steps == 738:
        batch_size = 123  # 6 batches
    elif n_steps == 984:
        batch_size = 123  # 8 batches
    else:
        batch_size = 64
    
    print(f"Batch size: {batch_size} ({n_steps // batch_size} mini-batches)")
    
    # Create environments
    env_train = DummyVecEnv([
        lambda: PortfolioEnv(train_features, prices_df=train_prices, tickers=TICKERS, use_correlation=False)
    ])
    
    env_val = DummyVecEnv([
        lambda: PortfolioEnv(val_features, prices_df=val_prices, tickers=TICKERS, use_correlation=False)
    ])
    
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)
    env_val = VecNormalize(env_val, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    
    # Model configuration
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        activation_fn=nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=0,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        vf_coef=0.5,
    )
    
    # Train
    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    # Evaluate
    print("Evaluating...")
    obs = env_val.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env_val.step(action)
        total_reward += reward[0]
        done = done[0]
    
    final_info = info[0]
    
    results = {
        'test_name': test_name,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'mini_batches': n_steps // batch_size,
        'total_reward': total_reward,
        'sharpe': final_info.get('sharpe', 0),
        'total_return': final_info.get('total_return', 0),
        'max_drawdown': final_info.get('max_drawdown', 0),
    }
    
    print(f"\nResults:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Sharpe: {results['sharpe']:.2f}")
    print(f"  Return: {results['total_return']:.2%}")
    print(f"  Max DD: {results['max_drawdown']:.2%}")
    
    return results

if __name__ == "__main__":
    print("="*60)
    print("N_STEPS PARAMETER OPTIMIZATION TEST")
    print("="*60)
    print("\nEpisode length: ~123 steps")
    print("Test duration: 10,000 timesteps each")
    print()
    
    results = []
    
    # Test A: 5 episodes
    results.append(test_n_steps_config(615, "A", timesteps=10000))
    
    # Test B: 6 episodes
    results.append(test_n_steps_config(738, "B", timesteps=10000))
    
    # Test C: 8 episodes
    results.append(test_n_steps_config(984, "C", timesteps=10000))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Find best
    best_idx = df_results['sharpe'].idxmax()
    best_config = df_results.iloc[best_idx]
    
    print(f"\n🏆 WINNER: Test {best_config['test_name']} (n_steps={best_config['n_steps']})")
    print(f"   Sharpe: {best_config['sharpe']:.2f}")
    print(f"   Return: {best_config['total_return']:.2%}")
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    df_results.to_csv('outputs/n_steps_test_results.csv', index=False)
    print(f"\n✅ Results saved to outputs/n_steps_test_results.csv")
