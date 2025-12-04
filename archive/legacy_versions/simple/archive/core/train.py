"""
Training Script for Portfolio RL

Simple walk-forward validation with:
- VecNormalize stats properly shared between train/val
- Early stopping on validation performance
- Best model selection across folds
"""

import numpy as np
import pandas as pd
import os
import sys
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import torch.nn as nn

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.rl_system import VecNormalize, PortfolioEnv
from core.data_loading_preprocessing import create_features
from core.config import *

# --- PPO Hyperparameters ---
MLP_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 192,
    "batch_size": 64,
    "ent_coef": 0.02,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
}

# --- Utility to avoid lambda closure bug ---
def make_env(env):
    def _init():
        return env
    return _init


def warmup_normalization(env_normalized, env_vec, n_steps=500):
    """
    Run random actions to populate normalization statistics.
    """
    obs = env_vec.reset()
    for _ in range(n_steps):
        action = [env_vec.action_space.sample()]
        obs, _, done, _ = env_vec.step(action)
        if done[0]:
            obs = env_vec.reset()


def train_model():
    """
    Main training function with walk-forward validation.
    """
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    # Setup directories
    models_dir = os.path.join(BASE_DIR, 'models')
    logs_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Load data
    print("\nLoading price data...")
    if not os.path.exists(RAW_DATA_TRAIN_FILE):
        print(f"❌ Data not found: {RAW_DATA_TRAIN_FILE}")
        print("Please run data_loading_preprocessing.py first.")
        sys.exit(1)

    raw_data = pd.read_csv(RAW_DATA_TRAIN_FILE, index_col=0, parse_dates=True, header=[0, 1])
    raw_prices = raw_data['prices']

    # Walk-forward setup
    n_samples = len(raw_prices)
    fold_size = n_samples // (N_FOLDS + 1)

    print(f"Total samples: {n_samples}")
    print(f"Fold size: {fold_size}")
    print(f"Number of folds: {N_FOLDS}")

    best_fold_score = -np.inf
    best_fold_model_path = None
    best_fold_stats_path = None

    for fold in range(N_FOLDS):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold + 1}/{N_FOLDS}")
        print(f"{'=' * 60}")

        # Define splits
        train_end_idx = (fold + 1) * fold_size
        val_start_idx = train_end_idx + GAP
        val_end_idx = min(val_start_idx + fold_size, n_samples)

        if val_end_idx - val_start_idx < 20:
            print(f"⚠️ Insufficient validation data, skipping fold {fold + 1}")
            continue

        # Get price slices
        train_prices = raw_prices.iloc[:train_end_idx]
        val_prices = raw_prices.iloc[val_start_idx:val_end_idx]

        print(f"Train: 0 to {train_end_idx} ({len(train_prices)} samples)")
        print(f"Val: {val_start_idx} to {val_end_idx} ({len(val_prices)} samples)")

        # Create features
        print("\nCreating features...")
        train_features = create_features(train_prices, None, verbose=False)

        # For validation: create features from combined data, then slice
        combined_prices = raw_prices.iloc[:val_end_idx]
        combined_features = create_features(combined_prices, None, verbose=False)

        # Extract validation portion
        val_start_in_features = len(train_features) + GAP
        val_features = combined_features.iloc[val_start_in_features:]

        print(f"Train features: {train_features.shape}")
        print(f"Val features: {val_features.shape}")

        # --- Create environments ---
        base_env_train = PortfolioEnv(
            train_features,
            prices_df=train_prices,
            tickers=TICKERS,
            use_correlation=False
        )

        base_env_val = PortfolioEnv(
            val_features,
            prices_df=val_prices,
            tickers=TICKERS,
            use_correlation=False
        )

        # Training env with normalization
        env_train_norm = VecNormalize(base_env_train, training=True)
        env_train = DummyVecEnv([make_env(env_train_norm)])
        env_train = VecMonitor(env_train, filename=os.path.join(logs_dir, f'train_fold_{fold}'))

        # Warmup normalization stats
        print("Warming up normalization statistics...")
        warmup_normalization(env_train_norm, env_train, n_steps=500)

        # Save normalization stats
        stats_path = os.path.join(models_dir, f'fold_{fold}_vecnormalize.pkl')
        env_train_norm.save(stats_path)

        # Create validation env with loaded stats
        env_val_norm = VecNormalize.load(stats_path, base_env_val, training=False)
        env_val = DummyVecEnv([make_env(env_val_norm)])
        env_val = VecMonitor(env_val, filename=os.path.join(logs_dir, f'val_fold_{fold}'))

        # --- Initialize PPO ---
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
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

        # Training schedule
        n_train_samples = len(train_features)
        timesteps_per_epoch = n_train_samples
        total_timesteps = EPOCHS_PER_FOLD * timesteps_per_epoch
        eval_freq = EVAL_FREQ_EPOCHS * timesteps_per_epoch

        print(f"\n📊 Training Schedule:")
        print(f"   Epochs: {EPOCHS_PER_FOLD}")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Eval frequency: every {EVAL_FREQ_EPOCHS} epochs")

        # Callbacks
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=15,
            min_evals=5,
            verbose=1
        )

        eval_callback = EvalCallback(
            env_val,
            best_model_save_path=os.path.join(models_dir, f'fold_{fold}'),
            log_path=os.path.join(logs_dir, f'fold_{fold}'),
            eval_freq=eval_freq,
            deterministic=True,
            n_eval_episodes=1,
            verbose=1,
            callback_after_eval=stop_callback
        )

        # Train
        print(f"\n🚀 Training fold {fold + 1}...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )

        # Save final model and updated stats
        final_model_path = os.path.join(models_dir, f'final_model_fold_{fold}.zip')
        final_stats_path = os.path.join(models_dir, f'final_model_fold_{fold}_vecnormalize.pkl')
        model.save(final_model_path)
        env_train_norm.save(final_stats_path)

        # Also save stats for best model
        best_model_dir = os.path.join(models_dir, f'fold_{fold}')
        best_stats_path = os.path.join(best_model_dir, 'best_model_vecnormalize.pkl')
        if os.path.exists(best_model_dir):
            env_train_norm.save(best_stats_path)

        print(f"✅ Fold {fold + 1} complete")

        # Print diagnostics
        _print_fold_diagnostics(logs_dir, fold)

        # Track best fold
        eval_path = os.path.join(logs_dir, f'fold_{fold}', 'evaluations.npz')
        if os.path.exists(eval_path):
            data = np.load(eval_path)
            fold_best = np.max(data['results'].mean(axis=1))
            if fold_best > best_fold_score:
                best_fold_score = fold_best
                best_fold_model_path = os.path.join(models_dir, f'fold_{fold}', 'best_model.zip')
                best_fold_stats_path = best_stats_path

    # Copy best overall model
    if best_fold_model_path and os.path.exists(best_fold_model_path):
        best_overall_model = os.path.join(models_dir, 'best_overall_model.zip')
        best_overall_stats = os.path.join(models_dir, 'best_overall_model_vecnormalize.pkl')

        shutil.copy(best_fold_model_path, best_overall_model)
        if best_fold_stats_path and os.path.exists(best_fold_stats_path):
            shutil.copy(best_fold_stats_path, best_overall_stats)

        print(f"\n🏆 Best overall model saved to {best_overall_model}")
        print(f"   Best validation score: {best_fold_score:.2f}")


def _print_fold_diagnostics(logs_dir, fold):
    """Print training diagnostics for a fold."""
    print(f"\n📊 Fold {fold + 1} Diagnostics:")

    # Training rewards
    train_monitor = os.path.join(logs_dir, f'train_fold_{fold}.monitor.csv')
    if os.path.exists(train_monitor):
        try:
            df = pd.read_csv(train_monitor, skiprows=1)
            rewards = df['r'].values
            print(f"   Train episodes: {len(rewards)}")
            print(f"   Train reward: {rewards.mean():.2f} ± {rewards.std():.2f}")
            print(f"   Train range: [{rewards.min():.2f}, {rewards.max():.2f}]")
            pct_positive = 100 * (rewards > 0).sum() / len(rewards)
            print(f"   Positive episodes: {pct_positive:.1f}%")
        except Exception as e:
            print(f"   ⚠️ Could not load train monitor: {e}")

    # Validation rewards
    val_path = os.path.join(logs_dir, f'fold_{fold}', 'evaluations.npz')
    if os.path.exists(val_path):
        try:
            data = np.load(val_path)
            val_rewards = data['results'].mean(axis=1)
            print(f"   Val reward: {val_rewards.mean():.2f} (best: {val_rewards.max():.2f})")
        except Exception as e:
            print(f"   ⚠️ Could not load val evaluations: {e}")


def train_single_model(df_features, df_prices, tickers, total_timesteps=20000):
    """
    Train a single model (for inference/app usage).
    Returns model and normalization wrapper.
    """
    print(f"Training single model for {total_timesteps} timesteps...")

    # Create environment
    base_env = PortfolioEnv(df_features, prices_df=df_prices, tickers=tickers)
    env_norm = VecNormalize(base_env, training=True)
    env = DummyVecEnv([make_env(env_norm)])

    # Warmup
    warmup_normalization(env_norm, env, n_steps=300)

    # Model
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        activation_fn=nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        **MLP_PARAMS
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    return model, env_norm


if __name__ == "__main__":
    train_model()