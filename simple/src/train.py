import gymnasium as gym
import numpy as np
import pandas as pd
import os
import sys
import argparse
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
import torch.nn as nn

# Add parent directory to path to import existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from existing system (reuse data engineering and environment)
from core.rl_system import PortfolioEnv
from core.data_eng_simple import create_features_no_leakage, validate_no_leakage, get_lean_features
from core.data_eng_expanded import create_expanded_features
from core.config import *

# Optimized Hyperparameters
MLP_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 64,
    "ent_coef": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
}

def train_model(lean_mode=False, expanded_mode=False):
    """
    Train RL agent with proper walk-forward validation
    lean_mode: If True, use reduced feature set
    expanded_mode: If True, use 175-feature expanded set instead of 83
    """
    # Determine mode string for logging
    if expanded_mode and lean_mode:
        mode_str = "EXPANDED-LEAN"
    elif expanded_mode:
        mode_str = "EXPANDED-FULL"
    elif lean_mode:
        mode_str = "LEAN"
    else:
        mode_str = "FULL"
        
    print("="*60)
    print(f"STARTING TRAINING ({mode_str} MODE)")
    print("="*60)
    
    # Set directories based on mode
    if expanded_mode and lean_mode:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_expanded_lean')
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs_expanded_lean')
    elif expanded_mode:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_expanded')
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs_expanded')
    elif lean_mode:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_lean')
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs_lean')
    else:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Get mode-specific data paths
    from core.config import get_data_paths
    data_paths = get_data_paths(expanded_mode=expanded_mode)
    
    # 1. Load RAW prices
    print("\nLoading RAW price data...")
    print(f"   Mode: {mode_str}")
    print(f"   Train data: {data_paths['train']}")
    print(f"   Test data: {data_paths['test']}")
    
    # Check if data exists
    if not os.path.exists(data_paths['train']):
        print(f"\n❌ Data files not found for {mode_str} mode!")
        print(f"   Expected: {data_paths['train']}")
        print(f"\n   Please run: python core/data_eng.py{'--expanded' if expanded_mode else ''}")
        sys.exit(1)
    
    raw_prices = pd.read_csv(data_paths['train'], index_col=0, parse_dates=True)
    
    # Walk-Forward Validation Setup
    n_samples = len(raw_prices)
    fold_size = n_samples // (N_FOLDS + 1)
    
    print(f"\nTotal samples: {n_samples}")
    print(f"Fold size: {fold_size}")
    
    best_fold_score = -np.inf
    best_fold_model_path = None
    
    for i in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {i+1}/{N_FOLDS}")
        print(f"{'='*60}")
        
        # Define splits
        train_start_idx = 0
        train_end_idx = (i + 1) * fold_size
        val_start_idx = train_end_idx + GAP
        val_end_idx = min(val_start_idx + fold_size, n_samples)
        
        if val_end_idx - val_start_idx < 20:
            print(f"⚠️ Insufficient validation data for fold {i+1}, skipping")
            continue
            
        # Create features (using existing robust pipeline)
        print(f"\nCreating features for fold {i+1}...")
        train_prices = raw_prices.iloc[train_start_idx:train_end_idx]
        val_prices = raw_prices.iloc[val_start_idx:val_end_idx]
        
        all_prices_up_to_val = raw_prices.iloc[:val_end_idx]
        
        # Use expanded or standard feature set
        if expanded_mode:
            print(f"   Using EXPANDED feature set (~175 features)")
            all_features_full = create_expanded_features(all_prices_up_to_val, None, TICKERS)
        else:
            all_features_full = create_features_no_leakage(all_prices_up_to_val, TICKERS)
        
        # Apply Lean Filter if requested
        if lean_mode:
            all_features = get_lean_features(all_features_full)
        else:
            all_features = all_features_full
        
        feature_train_end = train_end_idx - MIN_HISTORY
        feature_val_start = val_start_idx - MIN_HISTORY
        feature_val_end = val_end_idx - MIN_HISTORY
        
        if feature_train_end <= 0 or feature_val_start <= 0:
            continue
            
        train_features = all_features.iloc[:feature_train_end]
        val_features = all_features.iloc[feature_val_start:feature_val_end]
        
        print(f"Train features shape: {train_features.shape}")
        print(f"Val features shape: {val_features.shape}")
        
        # Create environments
        env_train = DummyVecEnv([
            lambda: PortfolioEnv(train_features, prices_df=train_prices, tickers=TICKERS)
        ])
        env_val = DummyVecEnv([
            lambda: PortfolioEnv(val_features, prices_df=val_prices, tickers=TICKERS)
        ])
        
        env_train = VecMonitor(env_train, filename=os.path.join(logs_dir, f'train_fold_{i}'))
        env_val = VecMonitor(env_val, filename=os.path.join(logs_dir, f'val_fold_{i}'))
        
        # Initialize PPO
        print(f"\nInitializing PPO ({mode_str}) for fold {i+1}...")
        
        # Custom Policy Network
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 64], vf=[128, 64]), # 2 hidden layers as requested
            activation_fn=nn.ReLU
        )
        
        model = PPO(
            "MlpPolicy",
            env_train,
            verbose=0,
            tensorboard_log=os.path.join(logs_dir, 'tensorboard'),
            policy_kwargs=policy_kwargs,
            **MLP_PARAMS
        )
        
        # Callbacks
        eval_callback = EvalCallback(
            env_val,
            best_model_save_path=os.path.join(models_dir, f'fold_{i}'),
            log_path=os.path.join(logs_dir, f'fold_{i}'),
            eval_freq=1000,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )
        
        # Train
        print(f"\n🚀 Training fold {i+1}...")
        model.learn(
            total_timesteps=TRAIN_TIMESTEPS, 
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(models_dir, f'final_model_fold_{i}.zip')
        model.save(final_path)
        print(f"✅ Fold {i+1} complete.")
        
        # Track best
        eval_path = os.path.join(logs_dir, f'fold_{i}', 'evaluations.npz')
        if os.path.exists(eval_path):
            data = np.load(eval_path)
            fold_best_score = np.max(data['results'].mean(axis=1))
            if fold_best_score > best_fold_score:
                best_fold_score = fold_best_score
                best_fold_model_path = os.path.join(models_dir, f'fold_{i}', 'best_model.zip')

    # Save overall best
    if best_fold_model_path and os.path.exists(best_fold_model_path):
        best_overall_path = os.path.join(models_dir, 'best_overall_model.zip')
        shutil.copy(best_fold_model_path, best_overall_path)
        print(f"\n🏆 Best overall model ({mode_str}) saved to {best_overall_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Agent")
    parser.add_argument("--lean", action="store_true", help="Use lean feature set (optimized)")
    parser.add_argument("--expanded", action="store_true", help="Use expanded feature set (175 features)")
    args = parser.parse_args()
    
    train_model(lean_mode=args.lean, expanded_mode=args.expanded)
