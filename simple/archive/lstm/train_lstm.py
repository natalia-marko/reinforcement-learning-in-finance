import gymnasium as gym
import numpy as np
import pandas as pd
import os
import sys
import argparse
import shutil
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# Add parent dir to path to find 'core' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rl_system import PortfolioEnv
from archive.data_eng_simple import create_features_no_leakage, validate_no_leakage, get_lean_features
from core.data_eng_expanded import create_expanded_features
from core.config import *

def train_model(lean_mode=False, expanded_mode=False):
    """
    Train model with proper walk-forward validation
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
    print(f"STARTING LSTM TRAINING ({mode_str} MODE)")
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
    
    # 1. Load RAW prices only (not features!)
    print("\nLoading RAW price data...")
    print(f"   Mode: {mode_str}")
    print(f"   Train data: {data_paths['train']}")
    print(f"   Test data: {data_paths['test']}")
    
    # Check if data exists
    if not os.path.exists(data_paths['train']):
        print(f"\n❌ Data files not found for {mode_str} mode!")
        print(f"   Expected: {data_paths['train']}")
        print(f"\n   Please run: python core/data_eng.py{' --expanded' if expanded_mode else ''}")
        exit(1)
    
    raw_prices = pd.read_csv(data_paths['train'], index_col=0, parse_dates=True)
    
    # Walk-Forward Validation Setup
    n_samples = len(raw_prices)
    fold_size = n_samples // (N_FOLDS + 1)
    
    print(f"\nTotal samples: {n_samples}")
    print(f"Fold size: {fold_size}")
    
    # Track best model across folds
    best_fold_score = -np.inf
    best_fold_model_path = None
    
    for i in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {i+1}/{N_FOLDS}")
        print(f"{'='*60}")
        
        # Define splits with proper embargo
        train_start_idx = 0
        train_end_idx = (i + 1) * fold_size
        val_start_idx = train_end_idx + GAP
        val_end_idx = min(val_start_idx + fold_size, n_samples)
        
        # Check if we have enough validation data
        if val_end_idx - val_start_idx < 20:
            print(f"⚠️ Insufficient validation data for fold {i+1}, skipping")
            continue
        
        # CRITICAL: Create features using ONLY training data
        print(f"\nCreating features for fold {i+1}...")
        
        # Get raw price splits
        train_prices = raw_prices.iloc[train_start_idx:train_end_idx]
        val_prices = raw_prices.iloc[val_start_idx:val_end_idx]
        
        # Create features using expanding window (no leakage)
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
        
        # Split features for train and validation
        feature_train_end = train_end_idx - MIN_HISTORY
        feature_val_start = val_start_idx - MIN_HISTORY
        feature_val_end = val_end_idx - MIN_HISTORY
        
        if feature_train_end <= 0 or feature_val_start <= 0:
            print(f"⚠️ Not enough data after feature creation for fold {i+1}, skipping")
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
        
        # Monitor environments
        env_train = VecMonitor(env_train, filename=os.path.join(logs_dir, f'train_fold_{i}'))
        env_val = VecMonitor(env_val, filename=os.path.join(logs_dir, f'val_fold_{i}'))
        
        # Initialize RecurrentPPO
        print(f"\nInitializing RecurrentPPO ({mode_str}) for fold {i+1}...")
        
        # Architecture settings
        policy_kwargs = dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            lstm_hidden_size=128, # Reduced size for stability
            enable_critic_lstm=True
        )
            
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env_train,
            verbose=0,
            tensorboard_log=os.path.join(logs_dir, 'tensorboard'),
            policy_kwargs=policy_kwargs,
            **PPO_PARAMS
        )
        
        # Callbacks for early stopping
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=10,
            min_evals=5,
            verbose=1
        )
        
        eval_callback = EvalCallback(
            env_val,
            best_model_save_path=os.path.join(models_dir, f'fold_{i}'),
            log_path=os.path.join(logs_dir, f'fold_{i}'),
            eval_freq=2000,
            deterministic=True,
            render=False,
            callback_on_new_best=stop_callback,
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
        print(f"✅ Fold {i+1} complete. Model saved to {final_path}")
        
        # Track best model across folds
        eval_path = os.path.join(logs_dir, f'fold_{i}', 'evaluations.npz')
        if os.path.exists(eval_path):
            data = np.load(eval_path)
            fold_best_score = np.max(data['results'].mean(axis=1))
            print(f"Fold {i+1} best validation score: {fold_best_score:.4f}")
            
            if fold_best_score > best_fold_score:
                best_fold_score = fold_best_score
                best_fold_model_path = os.path.join(models_dir, f'fold_{i}', 'best_model.zip')
    
    # Save the overall best model
    if best_fold_model_path and os.path.exists(best_fold_model_path):
        best_overall_path = os.path.join(models_dir, 'best_overall_model.zip')
        shutil.copy(best_fold_model_path, best_overall_path)
        print(f"\n🏆 Best overall model ({mode_str}) saved to {best_overall_path}")
        print(f"Best validation score: {best_fold_score:.4f}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

def train_single_model(df_features, df_prices, tickers, total_timesteps=50000):
    """
    Train a single model (for web app or quick testing)
    Args:
        df_features: Pre-computed features
        df_prices: Raw prices
        tickers: List of tickers
        total_timesteps: Training steps
    """
    print(f"Training single model on {len(df_features)} samples...")
    
    # Create environment using your original PortfolioEnv
    env = DummyVecEnv([
        lambda: PortfolioEnv(df_features, prices_df=df_prices, tickers=tickers)
    ])
    env = VecMonitor(env)
    
    # Initialize and train model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            lstm_hidden_size=128,
            enable_critic_lstm=True
        ),
        **PPO_PARAMS
    )
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Agent")
    parser.add_argument("--lean", action="store_true", help="Use lean feature set (optimized)")
    parser.add_argument("--expanded", action="store_true", help="Use expanded feature set (175 features)")
    args = parser.parse_args()
    
    train_model(lean_mode=args.lean, expanded_mode=args.expanded)