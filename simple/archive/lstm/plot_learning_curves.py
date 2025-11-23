import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
# Add parent dir to path to find 'core' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import *

def get_logs_dir(lean_mode=False, expanded_mode=False):
    if expanded_mode and lean_mode:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs_expanded_lean')
    elif expanded_mode:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs_expanded')
    elif lean_mode:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs_lean')
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

def plot_learning_curves(lean_mode=False, expanded_mode=False):
    """
    Generate comprehensive learning curves for all folds
    Shows train/validation performance and identifies overfitting
    """
    logs_dir = get_logs_dir(lean_mode, expanded_mode)
    if expanded_mode and lean_mode:
        mode_str = "EXPANDED-LEAN"
    elif expanded_mode:
        mode_str = "EXPANDED-FULL"
    elif lean_mode:
        mode_str = "LEAN"
    else:
        mode_str = "FULL"
        
    print("="*60)
    print(f"GENERATING LEARNING CURVES ({mode_str})")
    print("="*60)
    print(f"Using logs directory: {logs_dir}")
    
    if not os.path.exists(logs_dir):
        print(f"❌ Logs directory not found: {logs_dir}")
        print("Please run training first.")
        return
    
    # Create subplots for each fold
    fig = plt.figure(figsize=(15, 5 * N_FOLDS))
    
    for i in range(N_FOLDS):
        print(f"\nProcessing Fold {i+1}...")
        
        # 1. Load Training Data
        train_log_path = os.path.join(logs_dir, f'train_fold_{i}.monitor.csv')
        if not os.path.exists(train_log_path):
            print(f"⚠️ Training log not found for Fold {i+1}: {train_log_path}")
            continue
        
        # Read monitor file (skip metadata line)
        try:
            df_train = pd.read_csv(train_log_path, skiprows=1)
            print(f"Loaded {len(df_train)} training episodes for Fold {i+1}")
        except Exception as e:
            print(f"❌ Error loading training log: {e}")
            continue
        
        # Calculate cumulative timesteps and smoothed rewards
        df_train['timesteps'] = df_train['l'].cumsum()
        df_train['reward_raw'] = df_train['r']
        df_train['reward_smooth_10'] = df_train['r'].rolling(window=10, min_periods=1).mean()
        df_train['reward_smooth_50'] = df_train['r'].rolling(window=50, min_periods=1).mean()
        
        # 2. Load Validation Data
        eval_log_path = os.path.join(logs_dir, f'fold_{i}', 'evaluations.npz')
        if not os.path.exists(eval_log_path):
            print(f"⚠️ Validation log not found for Fold {i+1}")
            val_timesteps = []
            val_rewards_mean = []
            val_rewards_std = []
        else:
            try:
                data = np.load(eval_log_path)
                val_timesteps = data['timesteps']
                val_results = data['results']
                val_rewards_mean = val_results.mean(axis=1)
                val_rewards_std = val_results.std(axis=1)
                print(f"Loaded {len(val_timesteps)} validation evaluations for Fold {i+1}")
            except Exception as e:
                print(f"❌ Error loading validation log: {e}")
                val_timesteps = []
                val_rewards_mean = []
                val_rewards_std = []
        
        # 3. Create subplot for this fold
        ax = plt.subplot(N_FOLDS, 1, i + 1)
        
        # Plot training rewards
        ax.plot(df_train['timesteps'], df_train['reward_raw'], 
                alpha=0.2, color='blue', linewidth=0.5, label='Train (raw)')
        ax.plot(df_train['timesteps'], df_train['reward_smooth_10'], 
                alpha=0.5, color='blue', linewidth=1, label='Train (10-ep avg)')
        ax.plot(df_train['timesteps'], df_train['reward_smooth_50'], 
                alpha=1.0, color='blue', linewidth=2, label='Train (50-ep avg)')
        
        # Plot validation rewards
        if len(val_timesteps) > 0:
            ax.plot(val_timesteps, val_rewards_mean, 
                   color='red', linewidth=2, marker='o', label='Validation')
            ax.fill_between(val_timesteps, 
                           val_rewards_mean - val_rewards_std, 
                           val_rewards_mean + val_rewards_std, 
                           color='red', alpha=0.2)
            
            # Mark best validation point
            best_val_idx = np.argmax(val_rewards_mean)
            best_val_timestep = val_timesteps[best_val_idx]
            best_val_reward = val_rewards_mean[best_val_idx]
            ax.plot(best_val_timestep, best_val_reward, 
                   marker='*', markersize=15, color='gold', 
                   label=f'Best Val: {best_val_reward:.2f}')
            
            # Add vertical line at best validation
            ax.axvline(x=best_val_timestep, color='gold', linestyle='--', alpha=0.5)
        
        # Labels and formatting
        ax.set_title(f'Fold {i+1} Learning Curve', fontsize=14)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Reward')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"Final Train: {df_train['reward_smooth_50'].iloc[-1]:.2f}"
        if len(val_rewards_mean) > 0:
            stats_text += f"\nBest Val: {best_val_reward:.2f}"
            stats_text += f"\nFinal Val: {val_rewards_mean[-1]:.2f}"
            
            # Check for overfitting
            train_final = df_train['reward_smooth_50'].iloc[-1]
            val_final = val_rewards_mean[-1]
            if train_final - val_final > 20:
                stats_text += "\n⚠️ OVERFITTING!"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUTS_DIR, f'lstm_learning_curves_{mode_str.lower().replace("-", "_")}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Learning curves saved to {plot_path}")
    
    # Additional analysis plot - comparing folds
    plot_fold_comparison(logs_dir)
    analyze_training_stability(logs_dir)


def plot_fold_comparison(logs_dir):
    """
    Create a single plot comparing validation performance across folds
    """
    print("\nGenerating fold comparison plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    all_val_scores = []
    
    for i in range(N_FOLDS):
        eval_log_path = os.path.join(logs_dir, f'fold_{i}', 'evaluations.npz')
        if not os.path.exists(eval_log_path):
            continue
            
        data = np.load(eval_log_path)
        val_timesteps = data['timesteps']
        val_rewards_mean = data['results'].mean(axis=1)
        
        ax.plot(val_timesteps, val_rewards_mean, 
               color=colors[i % len(colors)], 
               linewidth=2, 
               marker='o',
               label=f'Fold {i+1}')
        
        all_val_scores.extend(val_rewards_mean.tolist())
    
    if all_val_scores:
        # Add average line
        avg_score = np.mean(all_val_scores)
        ax.axhline(y=avg_score, color='black', linestyle='--', 
                  label=f'Avg: {avg_score:.2f}')
    
    ax.set_title('Validation Performance Across Folds', fontsize=14)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Validation Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(OUTPUTS_DIR, 'lstm_fold_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Fold comparison saved to {plot_path}")


def analyze_training_stability(logs_dir):
    """
    Analyze training stability and convergence
    """
    print("\nAnalyzing training stability...")
    
    results = []
    
    for i in range(N_FOLDS):
        train_log_path = os.path.join(logs_dir, f'train_fold_{i}.monitor.csv')
        eval_log_path = os.path.join(logs_dir, f'fold_{i}', 'evaluations.npz')
        
        if not os.path.exists(train_log_path) or not os.path.exists(eval_log_path):
            continue
        
        # Load data
        df_train = pd.read_csv(train_log_path, skiprows=1)
        data = np.load(eval_log_path)
        
        # Calculate metrics
        train_rewards = df_train['r'].values
        val_rewards = data['results'].mean(axis=1)
        
        fold_stats = {
            'fold': i + 1,
            'train_mean': np.mean(train_rewards),
            'train_std': np.std(train_rewards),
            'train_final': df_train['r'].rolling(50).mean().iloc[-1],
            'val_best': np.max(val_rewards),
            'val_final': val_rewards[-1],
            'val_std': np.mean(data['results'].std(axis=1)),
            'overfit_gap': df_train['r'].rolling(50).mean().iloc[-1] - val_rewards[-1]
        }
        
        results.append(fold_stats)
    
    if results:
        # Print summary table
        print("\n" + "="*60)
        print("TRAINING STABILITY ANALYSIS (Reward Units)")
        print("="*60)
        
        df_results = pd.DataFrame(results)
        
        # Round all numeric columns to 2 decimal places
        df_results = df_results.round(2)
        
        print(df_results.to_string(index=False))
        
        # Overall statistics
        print("\n" + "-"*60)
        print("OVERALL STATISTICS:")
        print(f"Avg Train Final:  {df_results['train_final'].mean():.2f} ± {df_results['train_final'].std():.2f}")
        print(f"Avg Val Best:     {df_results['val_best'].mean():.2f} ± {df_results['val_best'].std():.2f}")
        print(f"Avg Val Final:    {df_results['val_final'].mean():.2f} ± {df_results['val_final'].std():.2f}")
        print(f"Avg Overfit Gap:  {df_results['overfit_gap'].mean():.2f}")
        
        # Warnings
        if df_results['overfit_gap'].mean() > 15:
            print("\n⚠️ WARNING: Significant overfitting detected!")
            print("Consider: Reducing model complexity, adding regularization, or more data")
        
        if df_results['val_std'].mean() > 50:
            print("\n⚠️ WARNING: High validation variance!")
            print("Consider: Increasing n_eval_episodes or stabilizing reward function")
        
        # Save analysis
        analysis_path = os.path.join(OUTPUTS_DIR, 'lstm_training_analysis.csv')
        df_results.to_csv(analysis_path, index=False)
        print(f"\n✅ Analysis saved to {analysis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot LSTM Learning Curves")
    parser.add_argument("--lean", action="store_true", help="Use lean feature set")
    parser.add_argument("--expanded", action="store_true", help="Use expanded feature set")
    args = parser.parse_args()
    
    plot_learning_curves(lean_mode=args.lean, expanded_mode=args.expanded)