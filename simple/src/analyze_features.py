import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
from stable_baselines3 import PPO
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rl_system import PortfolioEnv
from core.data_eng_simple import create_features_no_leakage
from core.data_eng_expanded import create_expanded_features
from core.config import *

# MLP Specific Paths
MLP_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

def train_quick_model(features, prices, expanded=False):
    """
    Train a QUICK model (10k timesteps) just to get feature importance
    """
    print("="*60)
    print("TRAINING QUICK MODEL FOR FEATURE IMPORTANCE ANALYSIS")
    print(f"Using {'EXPANDED' if expanded else 'STANDARD'} features")
    print("="*60)
    
    # Create environment
    print("\nCreating environment...")
    env = PortfolioEnv(features, prices_df=prices, tickers=TICKERS)
    
    # Train QUICK model (10k timesteps)
    print("\nTraining QUICK MLP model (10k timesteps for speed)...")
    
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
    )
    
    model.learn(total_timesteps=10000, progress_bar=True)
    
    return model

def calculate_feature_importance(expanded_mode=False, force_train=False):
    mode_str = "EXPANDED" if expanded_mode else "STANDARD"
    print("="*60)
    print(f"ANALYZING FEATURE IMPORTANCE ({mode_str} SET)")
    print("="*60)
    
    # 1. Load Data
    # Get correct paths based on mode
    paths = get_data_paths(expanded_mode=expanded_mode)
    raw_train_file = paths['train']
    test_set_file = paths['test']
    
    if not os.path.exists(raw_train_file):
        script_name = "data_eng_expanded.py" if expanded_mode else "data_eng_simple.py"
        print(f"❌ Raw data not found at {raw_train_file}")
        print(f"Please run core/{script_name} first.")
        return
        
    # We use TRAIN data for feature importance to see what the model learned
    # (or we can use Test data, but usually we want to explain the model's behavior)
    # The original script used Test data. Let's stick to Test data if available, else Train.
    
    if os.path.exists(test_set_file):
        print(f"Loading test set from {test_set_file}")
        data = pd.read_csv(test_set_file, index_col=0, parse_dates=True)
    else:
        print(f"Test set not found. Using train set from {raw_train_file}")
        data = pd.read_csv(raw_train_file, index_col=0, parse_dates=True)
        
    prices = data[TICKERS + list(MACRO_SYMBOLS.keys())]
    
    # Create features
    print(f"\nCreating {mode_str.lower()} features...")
    if expanded_mode:
        features = create_expanded_features(prices, None, TICKERS)
    else:
        features = create_features_no_leakage(prices, TICKERS)
        
    print(f"Features shape: {features.shape}")
    
    # 2. Load or Train Model
    model = None
    
    if not force_train and not expanded_mode:
        # Try to load existing best model for standard features
        model_path = os.path.join(MLP_MODELS_DIR, 'best_overall_model.zip')
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            model = PPO.load(model_path)
    
    if model is None:
        print("\nNo suitable model found (or forced training). Training quick model...")
        # We need to train on the same data we analyze, or ideally train on train set and analyze on test set.
        # But for quick analysis, we can just train on the data we have.
        # To be rigorous, let's train on the data provided.
        model = train_quick_model(features, prices[TICKERS], expanded=expanded_mode)
    
    # 3. Baseline Performance
    print("\nCalculating baseline performance...")
    env = PortfolioEnv(features, prices[TICKERS], tickers=TICKERS)
    baseline_reward = evaluate_agent(model, env)
    print(f"Baseline Avg Reward: {baseline_reward:.4f}")
    
    # 4. Permutation Importance
    feature_names = features.columns.tolist()
    importances = {}
    
    print(f"\nAnalyzing {len(feature_names)} features...")
    print("(This may take a while)")
    
    for i, feature in enumerate(feature_names):
        # Create perturbed features
        perturbed_df = features.copy()
        # Shuffle the column
        perturbed_df[feature] = np.random.permutation(perturbed_df[feature].values)
        
        # Evaluate
        env_perturbed = PortfolioEnv(perturbed_df, prices[TICKERS], tickers=TICKERS)
        perturbed_reward = evaluate_agent(model, env_perturbed)
        
        # Drop in performance = Importance
        importance = baseline_reward - perturbed_reward
        importances[feature] = importance
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(feature_names)} features...")
            
    # 5. Visualize Results
    importance_df = pd.DataFrame({
        'Feature': list(importances.keys()),
        'Importance': list(importances.values())
    })
    
    # Sort
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Output filenames
    suffix = "_expanded" if expanded_mode else ""
    csv_filename = f'feature_importance{suffix}.csv'
    plot_filename = f'feature_importance{suffix}.png'
    
    # Save full list
    csv_path = os.path.join(OUTPUTS_DIR, csv_filename)
    importance_df.to_csv(csv_path, index=False)
    print(f"\n✅ Feature importance saved to {csv_path}")
    
    # Plot Top 20/30
    n_top = 30 if expanded_mode else 20
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df.head(n_top), palette='viridis', legend=False)
    plt.title(f'Top {n_top} Most Important Features ({mode_str})', fontsize=14)
    plt.xlabel('Drop in Reward (Importance)', fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUTS_DIR, plot_filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✅ Feature importance plot saved to {plot_path}")
    
    # Recommendations
    print("\n" + "-"*60)
    print("RECOMMENDATIONS:")
    print("-" * 60)
    print("Top 5 Drivers:")
    for i in range(min(5, len(importance_df))):
        row = importance_df.iloc[i]
        print(f"{i+1}. {row['Feature']} ({row['Importance']:.4f})")
        
    print("\nBottom 5 (Potential Noise):")
    for i in range(min(5, len(importance_df))):
        row = importance_df.iloc[-(i+1)]
        print(f"{i+1}. {row['Feature']} ({row['Importance']:.4f})")

def evaluate_agent(model, env, n_episodes=3):
    """Run N episodes and return average total reward"""
    total_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Feature Importance")
    parser.add_argument("--expanded", action="store_true", help="Use expanded feature set (175 features)")
    parser.add_argument("--train", action="store_true", help="Force training a new quick model")
    args = parser.parse_args()
    
    calculate_feature_importance(expanded_mode=args.expanded, force_train=args.train)
