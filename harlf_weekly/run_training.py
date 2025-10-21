#!/usr/bin/env python3
"""
Simple training script for Level 1 agents.
Fixes the argparse issue in Jupyter notebooks.
"""

import numpy as np
import pandas as pd
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from level_1_trainer import PortfolioEnvWeekly, load_data, get_feature_indices
from stable_baselines3 import PPO

def main():
    # Configuration
    data_dir = "data"
    algo = "ppo"
    steps = 50000
    eval_freq = 5000
    patience = 5
    rolling_vol = 12
    lr = 3e-4
    gamma = 0.99
    seed = 42
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    print("="*70)
    print("LEVEL 1 TRAINING - TECH & MOMENTUM AGENTS")
    print("="*70)
    
    # Load data
    print("Loading data...")
    train_features, train_returns = load_data('train')
    val_features, val_returns = load_data('val')
    
    # Load tech dataframe to get feature indices
    tech_df = pd.read_csv('data/train/technical.csv')
    tech_features_idx, momentum_features_idx = get_feature_indices(tech_df)
    
    print(f"Tech features: {len(tech_features_idx)}")
    print(f"Momentum features: {len(momentum_features_idx)}")
    
    # Create environments
    tech_features = train_features[:, tech_features_idx, :]
    momentum_features = train_features[:, momentum_features_idx, :]
    
    tech_env = PortfolioEnvWeekly(tech_features, train_returns, cost_rate=0.0005)
    momentum_env = PortfolioEnvWeekly(momentum_features, train_returns, cost_rate=0.0005)
    
    print(f"\nTech Environment: {tech_features.shape}")
    print(f"Momentum Environment: {momentum_features.shape}")
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Train Tech Agent
    print("\n" + "="*50)
    print("TRAINING TECH AGENT")
    print("="*50)
    tech_agent = PPO('MlpPolicy', tech_env, learning_rate=lr, verbose=1, seed=seed)
    tech_agent.learn(total_timesteps=steps)
    tech_agent.save('models/tech_agent')
    print("✓ Tech agent saved")
    
    # Train Momentum Agent
    print("\n" + "="*50)
    print("TRAINING MOMENTUM AGENT")
    print("="*50)
    momentum_agent = PPO('MlpPolicy', momentum_env, learning_rate=lr, verbose=1, seed=seed)
    momentum_agent.learn(total_timesteps=steps)
    momentum_agent.save('models/momentum_agent')
    print("✓ Momentum agent saved")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print("Models saved to: models/")
    print("Ready for Level 2 training!")

if __name__ == '__main__':
    main()
