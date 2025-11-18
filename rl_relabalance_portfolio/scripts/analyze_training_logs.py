"""
Analyze and Visualize Training Logs

Usage:
    python analyze_training_logs.py --fold 0
    python analyze_training_logs.py --all  # Analyze all folds
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style('whitegrid')


def analyze_fold(fold_idx: int, base_dir: Path):
    """Analyze and visualize logs for a single fold."""

    fold_dir = base_dir / f"fold_{fold_idx}"
    logs_dir = fold_dir / "logs"

    if not logs_dir.exists():
        print(f"Warning: Logs directory not found for fold {fold_idx}")
        return None

    # Load logs
    train_log_path = logs_dir / "train_episodes.csv"
    val_log_path = logs_dir / "val_evaluations.csv"

    train_log = None
    val_log = None

    if train_log_path.exists():
        train_log = pd.read_csv(train_log_path)
    else:
        print(f"Warning: Train log not found for fold {fold_idx}")

    if val_log_path.exists():
        val_log = pd.read_csv(val_log_path)
    else:
        print(f"Warning: Validation log not found for fold {fold_idx}")

    if train_log is None and val_log is None:
        return None

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Analysis - Fold {fold_idx}', fontsize=16, fontweight='bold')

    # 1. Sharpe Ratio
    ax = axes[0, 0]
    if train_log is not None:
        ax.plot(train_log['step'], train_log['sharpe_ratio'],
                alpha=0.3, label='Train', color='blue')
        # Add smoothed line
        window = min(20, len(train_log) // 4)
        if window > 1:
            smooth = train_log['sharpe_ratio'].rolling(window=window).mean()
            ax.plot(train_log['step'], smooth,
                    label=f'Train (MA-{window})', color='blue', linewidth=2)

    if val_log is not None:
        ax.plot(val_log['step'], val_log['sharpe_ratio'],
                marker='o', label='Validation', color='red', linewidth=2)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Total Return
    ax = axes[0, 1]
    if train_log is not None:
        ax.plot(train_log['step'], train_log['total_return'] * 100,
                alpha=0.3, label='Train', color='green')
        window = min(20, len(train_log) // 4)
        if window > 1:
            smooth = train_log['total_return'].rolling(window=window).mean()
            ax.plot(train_log['step'], smooth * 100,
                    label=f'Train (MA-{window})', color='green', linewidth=2)

    if val_log is not None:
        ax.plot(val_log['step'], val_log['total_return'] * 100,
                marker='o', label='Validation', color='darkgreen', linewidth=2)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Total Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Volatility
    ax = axes[0, 2]
    if train_log is not None:
        ax.plot(train_log['step'], train_log['volatility'] * 100,
                alpha=0.3, label='Train', color='purple')
        window = min(20, len(train_log) // 4)
        if window > 1:
            smooth = train_log['volatility'].rolling(window=window).mean()
            ax.plot(train_log['step'], smooth * 100,
                    label=f'Train (MA-{window})', color='purple', linewidth=2)

    if val_log is not None:
        ax.plot(val_log['step'], val_log['volatility'] * 100,
                marker='o', label='Validation', color='darkviolet', linewidth=2)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Volatility (%)')
    ax.set_title('Volatility (Annualized)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Max Drawdown
    ax = axes[1, 0]
    if train_log is not None:
        ax.plot(train_log['step'], train_log['max_drawdown'] * 100,
                alpha=0.3, label='Train', color='coral')
        window = min(20, len(train_log) // 4)
        if window > 1:
            smooth = train_log['max_drawdown'].rolling(window=window).mean()
            ax.plot(train_log['step'], smooth * 100,
                    label=f'Train (MA-{window})', color='coral', linewidth=2)

    if val_log is not None:
        ax.plot(val_log['step'], val_log['max_drawdown'] * 100,
                marker='o', label='Validation', color='red', linewidth=2)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Max Drawdown (%)')
    ax.set_title('Maximum Drawdown')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Portfolio Value
    ax = axes[1, 1]
    if train_log is not None:
        ax.plot(train_log['step'], train_log['portfolio_value'] / 1000,
                alpha=0.3, label='Train', color='teal')
        window = min(20, len(train_log) // 4)
        if window > 1:
            smooth = train_log['portfolio_value'].rolling(window=window).mean()
            ax.plot(train_log['step'], smooth / 1000,
                    label=f'Train (MA-{window})', color='teal', linewidth=2)

    if val_log is not None:
        ax.plot(val_log['step'], val_log['portfolio_value'] / 1000,
                marker='o', label='Validation', color='darkcyan', linewidth=2)

    ax.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Portfolio Value ($1000s)')
    ax.set_title('Portfolio Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Episode Reward
    ax = axes[1, 2]
    if train_log is not None:
        ax.plot(train_log['step'], train_log['reward'],
                alpha=0.3, label='Train', color='orange')
        window = min(20, len(train_log) // 4)
        if window > 1:
            smooth = train_log['reward'].rolling(window=window).mean()
            ax.plot(train_log['step'], smooth,
                    label=f'Train (MA-{window})', color='orange', linewidth=2)

    if val_log is not None:
        ax.plot(val_log['step'], val_log['reward'],
                marker='o', label='Validation', color='darkorange', linewidth=2)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = fold_dir / "training_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return {
        'train_log': train_log,
        'val_log': val_log,
        'fold_idx': fold_idx
    }


def compare_all_folds(base_dir: Path, n_folds: int = 5):
    """Compare validation performance across all folds."""

    all_val_logs = []

    for fold_idx in range(n_folds):
        val_log_path = base_dir / f"fold_{fold_idx}" / "logs" / "val_evaluations.csv"
        if val_log_path.exists():
            val_log = pd.read_csv(val_log_path)
            val_log['fold'] = fold_idx
            all_val_logs.append(val_log)

    if not all_val_logs:
        print("No validation logs found!")
        return

    combined = pd.concat(all_val_logs, ignore_index=True)

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Validation Performance Across Folds', fontsize=16, fontweight='bold')

    # 1. Sharpe Ratio
    ax = axes[0, 0]
    for fold_idx in range(n_folds):
        fold_data = combined[combined['fold'] == fold_idx]
        if len(fold_data) > 0:
            ax.plot(fold_data['step'], fold_data['sharpe_ratio'],
                    marker='o', label=f'Fold {fold_idx}', alpha=0.7)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Validation Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Total Return
    ax = axes[0, 1]
    for fold_idx in range(n_folds):
        fold_data = combined[combined['fold'] == fold_idx]
        if len(fold_data) > 0:
            ax.plot(fold_data['step'], fold_data['total_return'] * 100,
                    marker='o', label=f'Fold {fold_idx}', alpha=0.7)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Validation Total Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Final Sharpe by Fold
    ax = axes[1, 0]
    final_sharpes = []
    for fold_idx in range(n_folds):
        fold_data = combined[combined['fold'] == fold_idx]
        if len(fold_data) > 0:
            final_sharpes.append(fold_data.iloc[-1]['sharpe_ratio'])
        else:
            final_sharpes.append(0)

    ax.bar(range(n_folds), final_sharpes, color='steelblue', alpha=0.7)
    ax.axhline(y=np.mean(final_sharpes), color='red', linestyle='--',
               label=f'Mean: {np.mean(final_sharpes):.3f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Final Sharpe Ratio')
    ax.set_title('Final Validation Sharpe by Fold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Best Sharpe by Fold
    ax = axes[1, 1]
    best_sharpes = []
    for fold_idx in range(n_folds):
        fold_data = combined[combined['fold'] == fold_idx]
        if len(fold_data) > 0:
            best_sharpes.append(fold_data['sharpe_ratio'].max())
        else:
            best_sharpes.append(0)

    ax.bar(range(n_folds), best_sharpes, color='green', alpha=0.7)
    ax.axhline(y=np.mean(best_sharpes), color='red', linestyle='--',
               label=f'Mean: {np.mean(best_sharpes):.3f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Best Sharpe Ratio')
    ax.set_title('Best Validation Sharpe by Fold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = base_dir / "all_folds_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY ACROSS FOLDS")
    print("=" * 80)
    print(f"\nFinal Sharpe Ratios:")
    for i, sharpe in enumerate(final_sharpes):
        print(f"  Fold {i}: {sharpe:.3f}")
    print(f"\nAverage: {np.mean(final_sharpes):.3f} ± {np.std(final_sharpes):.3f}")
    print(f"Best: {np.max(final_sharpes):.3f}")
    print(f"Worst: {np.min(final_sharpes):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze training logs')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index to analyze (default: analyze all)')
    parser.add_argument('--base-dir', type=str, default='models/baseline_v2',
                        help='Base directory for models')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Total number of folds')

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return

    if args.fold is not None:
        # Analyze single fold
        print(f"Analyzing fold {args.fold}...")
        analyze_fold(args.fold, base_dir)
    else:
        # Analyze all folds
        print(f"Analyzing all {args.n_folds} folds...")
        for fold_idx in range(args.n_folds):
            print(f"\nFold {fold_idx}:")
            analyze_fold(fold_idx, base_dir)

        # Compare across folds
        print(f"\nComparing all folds...")
        compare_all_folds(base_dir, args.n_folds)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
