import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from stable_baselines3 import PPO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rl_system import PortfolioEnv
from core.data_loading_preprocessing import create_features
from core.config import *

# MLP Specific Paths (use project root, not core/)
MLP_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

def train_quick_model(features, prices):
    """Train a QUICK model (10k timesteps) for feature importance."""
    print("="*60)
    print("TRAINING QUICK MODEL FOR FEATURE IMPORTANCE ANALYSIS")
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

def calculate_feature_importance(force_train=False):
    print("="*60)
    print("ANALYZING FEATURE IMPORTANCE")
    print("="*60)
    
    # 1. Load Data
    paths = get_data_paths()
    raw_train_file = paths['train']
    
    if not os.path.exists(raw_train_file):
        script_name = "data_loading_preprocessing.py"
        print(f"❌ Raw data not found at {raw_train_file}")
        print(f"Please run core/{script_name} first.")
        return
        
    # We use TRAIN data for feature importance to see what the model learned
    # (or we can use Test data, but usually we want to explain the model's behavior)
    # The original script used Test data. Let's stick to Test data if available, else Train.
    
    # CRITICAL FIX: Use TRAIN data for feature importance to avoid selection bias on Test set
    # We want to know which features were important during training
    print(f"Loading train set from {raw_train_file}")
    # Read MultiIndex CSV (header=[0, 1] because row 0 is 'prices'/'volumes' and row 1 is 'AAPL'/'AMD' etc)
    data = pd.read_csv(raw_train_file, index_col=0, parse_dates=True, header=[0, 1])
    
    # Extract prices (level 0 is 'prices', level 1 is ticker)
    if 'prices' in data.columns.levels[0]:
        prices = data['prices'].copy()
    else:
        # Fallback if flat
        prices = data.copy()
        
    # Ensure we have the required columns
    missing_cols = [c for c in TICKERS if c not in prices.columns]
    if missing_cols:
        print(f"Warning: Missing tickers in data: {missing_cols}")
    
    # Create features
    print("\nCreating features...")
    features = create_features(prices, None, verbose=False)
        
    print(f"Features shape: {features.shape}")
    
    # 2. Load or Train Model
    model = None
    
    if not force_train:
        # Try to load existing best model
        model_path = os.path.join(MLP_MODELS_DIR, 'best_overall_model.zip')
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            model = PPO.load(model_path)
            
            # Check if model's observation space matches current features
            expected_obs_dim = model.observation_space.shape[0]
            actual_obs_dim = features.shape[1]
            
            if expected_obs_dim != actual_obs_dim:
                print(f"⚠️  Dimension mismatch detected!")
                print(f"   Model expects {expected_obs_dim} features, but data has {actual_obs_dim} features")
                print(f"   Will train a new quick model with correct dimensions...")
                model = None  # Force training
    
    if model is None:
        print("\nNo suitable model found (or forced training). Training quick model...")
        model = train_quick_model(features, prices[TICKERS])
    
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
        
        if (i+1) % 30 == 0:
            print(f"Processed {i+1}/{len(feature_names)} features...")
            
    # 5. Visualize Results
    importance_df = pd.DataFrame({
        'Feature': list(importances.keys()),
        'Importance': list(importances.values())
    })
    
    # Sort
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # Save full list
    csv_path = os.path.join(OUTPUTS_DIR, 'feature_importance.csv')
    importance_df.to_csv(csv_path, index=False)
    print(f"\n✅ Feature importance saved to {csv_path}")
    
    # Filter features by importance threshold (0.05)
    importance_threshold = 0.05
    selected_features = importance_df[importance_df['Importance'] > importance_threshold]
    
    # Save selected features list
    selected_csv_path = os.path.join(OUTPUTS_DIR, 'selected_features.csv')
    selected_json_path = os.path.join(OUTPUTS_DIR, 'selected_features.json')

    selected_features.to_csv(selected_csv_path, index=False)

    # Also save as JSON list for easy import
    import json
    with open(selected_json_path, 'w') as f:
        json.dump(selected_features['Feature'].tolist(), f, indent=2)

    print(f"\n📊 FEATURE SELECTION RESULTS:")
    print(f"   Original features: {len(importance_df)}")
    print(f"   Selected features (importance > {importance_threshold}): {len(selected_features)}")
    print(f"   Reduction: {len(importance_df) - len(selected_features)} features removed")
    print(f"   ✅ Selected features saved to {selected_csv_path}")
    print(f"   ✅ Selected features list saved to {selected_json_path}")
    
    # Plot Top 20
    n_top = 20
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df.head(n_top), palette='viridis', legend=False)
    plt.title(f'Top {n_top} Most Important Features', fontsize=14)
    plt.xlabel('Drop in Reward (Importance)', fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUTS_DIR, 'feature_importance.png')
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
    calculate_feature_importance(force_train=False)
