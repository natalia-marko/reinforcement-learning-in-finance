import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Matplotlib backend setup
try:
    get_ipython()
    from IPython import display
    matplotlib.use('module://matplotlib_inline.backend_inline')
except NameError:
    matplotlib.use('Agg')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import *


def is_jupyter():
    """Check if running in Jupyter environment"""
    try:
        get_ipython()
        return True
    except NameError:
        return False


def get_project_root():
    """Get project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_training_data(fold_idx, logs_dir):
    """Load training data for a specific fold"""
    train_log_path = os.path.join(logs_dir, f'train_fold_{fold_idx}.monitor.csv')
    
    if not os.path.exists(train_log_path):
        print(f"⚠️ Training log not found for Fold {fold_idx + 1}")
        return None
    
    try:
        df = pd.read_csv(train_log_path, skiprows=1)
        df['timesteps'] = df['l'].cumsum()
        df['reward_raw'] = df['r']
        df['reward_smooth_10'] = df['r'].rolling(window=10, min_periods=1).mean()
        df['reward_smooth_50'] = df['r'].rolling(window=50, min_periods=1).mean()
        print(f"Loaded {len(df)} training episodes for Fold {fold_idx + 1}")
        return df
    except Exception as e:
        print(f"❌ Error loading training log: {e}")
        return None


def load_validation_data(fold_idx, logs_dir):
    """Load validation data for a specific fold"""
    eval_log_path = os.path.join(logs_dir, f'fold_{fold_idx}', 'evaluations.npz')
    
    if not os.path.exists(eval_log_path):
        return None, None, None
    
    try:
        data = np.load(eval_log_path)
        timesteps = data['timesteps']
        results = data['results']
        rewards_mean = results.mean(axis=1)
        rewards_std = results.std(axis=1)
        print(f"Loaded {len(timesteps)} validation evaluations for Fold {fold_idx + 1}")
        return timesteps, rewards_mean, rewards_std
    except Exception as e:
        print(f"❌ Error loading validation log: {e}")
        return None, None, None


def plot_fold(ax, df_train, val_timesteps, val_rewards_mean, val_rewards_std, fold_idx):
    """Plot learning curve for a single fold"""
    # Training curves
    ax.plot(df_train['timesteps'], df_train['reward_raw'], 
            alpha=0.2, color='blue', linewidth=0.5, label='Train (raw)')
    ax.plot(df_train['timesteps'], df_train['reward_smooth_10'], 
            alpha=0.5, color='blue', linewidth=1, label='Train (10-ep avg)')
    ax.plot(df_train['timesteps'], df_train['reward_smooth_50'], 
            alpha=1.0, color='blue', linewidth=2, label='Train (50-ep avg)')
    
    # Validation curves
    if val_timesteps is not None and len(val_timesteps) > 0:
        ax.plot(val_timesteps, val_rewards_mean, 
               color='red', linewidth=2, marker='o', label='Validation')
        ax.fill_between(val_timesteps, 
                       val_rewards_mean - val_rewards_std, 
                       val_rewards_mean + val_rewards_std, 
                       color='red', alpha=0.2)
        
        # Mark best validation point
        best_idx = np.argmax(val_rewards_mean)
        best_timestep = val_timesteps[best_idx]
        best_reward = val_rewards_mean[best_idx]
        ax.plot(best_timestep, best_reward, 
               marker='*', markersize=15, color='gold', 
               label=f'Best Val: {best_reward:.2f}')
        ax.axvline(x=best_timestep, color='gold', linestyle='--', alpha=0.5)
    
    # Styling
    ax.set_title(f'Fold {fold_idx + 1} Learning Curve', fontsize=14)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Episode Reward')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Stats box
    stats_text = f"Final Train: {df_train['reward_smooth_50'].iloc[-1]:.2f}"
    if val_rewards_mean is not None and len(val_rewards_mean) > 0:
        stats_text += f"\nBest Val: {best_reward:.2f}"
        stats_text += f"\nFinal Val: {val_rewards_mean[-1]:.2f}"
        
        train_final = df_train['reward_smooth_50'].iloc[-1]
        val_final = val_rewards_mean[-1]
        if train_final - val_final > 20:
            stats_text += "\n⚠️ OVERFITTING!"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5), fontsize=10)


def save_plot(filename):
    """Save plot and show in Jupyter if applicable"""
    plot_path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if is_jupyter():
        plt.show()
    plt.close()
    return plot_path


def plot_learning_curves():
    """Generate comprehensive learning curves for the RL agent"""
    logs_dir = os.path.join(get_project_root(), 'logs')
    
    print("=" * 60)
    print("GENERATING LEARNING CURVES")
    print("=" * 60)
    
    if not os.path.exists(logs_dir):
        print(f"❌ Logs directory not found at {logs_dir}")
        print("Please run 'python core/train.py' first.")
        return
    
    # Create figure with subplots for each fold
    fig = plt.figure(figsize=(15, 5 * N_FOLDS))
    
    for i in range(N_FOLDS):
        print(f"\nProcessing Fold {i + 1}...")
        
        # Load data
        df_train = load_training_data(i, logs_dir)
        if df_train is None:
            continue
        
        val_timesteps, val_rewards_mean, val_rewards_std = load_validation_data(i, logs_dir)
        
        # Plot
        ax = plt.subplot(N_FOLDS, 1, i + 1)
        plot_fold(ax, df_train, val_timesteps, val_rewards_mean, val_rewards_std, i)
    
    plt.tight_layout()
    plot_path = save_plot('learning_curves.png')
    print(f"\n✅ Learning curves saved to {plot_path}")
    
    # Additional analysis
    plot_fold_comparison(logs_dir)
    analyze_stability(logs_dir)


def plot_fold_comparison(logs_dir):
    """Plot validation performance comparison across folds"""
    print("\nGenerating fold comparison plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = ['blue', 'green', 'red']
    all_val_scores = []
    
    for i in range(N_FOLDS):
        eval_log_path = os.path.join(logs_dir, f'fold_{i}', 'evaluations.npz')
        if not os.path.exists(eval_log_path):
            continue
        
        data = np.load(eval_log_path)
        val_timesteps = data['timesteps']
        val_rewards_mean = data['results'].mean(axis=1)
        
        ax.plot(val_timesteps, val_rewards_mean,
               color=colors[i % len(colors)], linewidth=2, marker='o',
               label=f'Fold {i + 1}')
        all_val_scores.extend(val_rewards_mean.tolist())
    
    if all_val_scores:
        avg_score = np.mean(all_val_scores)
        ax.axhline(y=avg_score, color='black', linestyle='--', 
                  label=f'Avg: {avg_score:.2f}')
    
    ax.set_title('Validation Performance Across Folds', fontsize=14)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Validation Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_plot('fold_comparison.png')
    print(f"✅ Fold comparison saved to {plot_path}")


def analyze_stability(logs_dir):
    """Analyze training stability across folds"""
    print("\nAnalyzing training stability...")
    results = []
    
    for i in range(N_FOLDS):
        train_log_path = os.path.join(logs_dir, f'train_fold_{i}.monitor.csv')
        eval_log_path = os.path.join(logs_dir, f'fold_{i}', 'evaluations.npz')
        
        if not os.path.exists(train_log_path) or not os.path.exists(eval_log_path):
            continue
        
        df_train = pd.read_csv(train_log_path, skiprows=1)
        data = np.load(eval_log_path)
        
        train_rewards = df_train['r'].values
        val_rewards = data['results'].mean(axis=1)
        train_final = df_train['r'].rolling(50).mean().iloc[-1]
        
        fold_stats = {
            'fold': i + 1,
            'train_mean': np.mean(train_rewards),
            'train_std': np.std(train_rewards),
            'train_final': train_final,
            'val_best': np.max(val_rewards),
            'val_final': val_rewards[-1],
            'val_std': np.std(val_rewards),
            'val_episode_std': np.mean(data['results'].std(axis=1)),
            'overfit_gap': train_final - val_rewards[-1]
        }
        results.append(fold_stats)
    
    if not results:
        return
    
    # Print analysis
    print("\n" + "=" * 60)
    print("TRAINING STABILITY ANALYSIS")
    print("=" * 60)
    
    df_results = pd.DataFrame(results).round(2)
    print(df_results.to_string(index=False))
    
    print("\n" + "-" * 60)
    print("OVERALL STATISTICS:")
    print(f"Avg Train Final:  {df_results['train_final'].mean():.2f} ± {df_results['train_final'].std():.2f}")
    print(f"Avg Val Best:     {df_results['val_best'].mean():.2f} ± {df_results['val_best'].std():.2f}")
    print(f"Avg Val Final:    {df_results['val_final'].mean():.2f} ± {df_results['val_final'].std():.2f}")
    print(f"Avg Overfit Gap:  {df_results['overfit_gap'].mean():.2f}")
    
    analysis_path = os.path.join(OUTPUTS_DIR, 'training_analysis.csv')
    df_results.to_csv(analysis_path, index=False)
    print(f"\n✅ Analysis saved to {analysis_path}")


if __name__ == "__main__":
    plot_learning_curves()
