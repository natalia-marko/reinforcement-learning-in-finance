"""
Visualization Functions for Data Preparation

Simple, focused visualization functions for the minimal feature data preparation pipeline.
Each function creates one specific plot type.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional


def plot_price_trends(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tickers: List[str],
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Plot normalized price trends for all tickers (train + test)

    Args:
        train_df: Training data with 'date', 'ticker', 'close' columns
        test_df: Test data with same columns
        tickers: List of ticker symbols to plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Combine train and test
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # Plot each ticker
    for ticker in tickers:
        ticker_data = full_df[full_df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date')

        # Normalize to 100 at start
        base_price = ticker_data['close'].iloc[0]
        ticker_data['norm_price'] = 100 * ticker_data['close'] / base_price

        ax.plot(ticker_data['date'], ticker_data['norm_price'],
                label=ticker, linewidth=2, alpha=0.8)

    # Mark train/test split
    if len(test_df) > 0:
        split_date = test_df['date'].min()
        ax.axvline(split_date, color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label='Train/Test Split')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Price (Base=100)', fontsize=12)
    ax.set_title('Price Trends: All Tickers (Normalized)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_distributions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    n_features: int = 12,
    figsize: tuple = (16, 10)
) -> plt.Figure:
    """
    Plot distributions of top features (train vs test)

    Args:
        train_df: Training data
        test_df: Test data
        feature_cols: List of feature column names
        n_features: Number of features to plot
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Select features to plot
    features_to_plot = feature_cols[:n_features]
    n_cols = 4
    n_rows = (len(features_to_plot) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]

        # Get data
        train_data = train_df[feature].dropna()
        test_data = test_df[feature].dropna()

        # Plot histograms
        ax.hist(train_data, bins=30, alpha=0.6, label='Train', color='blue', density=True)
        ax.hist(test_data, bins=30, alpha=0.6, label='Test', color='orange', density=True)

        # Clean feature name for title
        clean_name = feature.replace('_norm', '').replace('_', ' ').title()
        ax.set_title(clean_name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(features_to_plot), len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Feature Distributions: Train vs Test',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


def plot_feature_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    n_features: int = 20,
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Plot mean and std comparison between train and test features

    Args:
        train_df: Training data
        test_df: Test data
        feature_cols: List of feature columns
        n_features: Number of features to show
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Calculate statistics
    stats = []
    for col in feature_cols[:n_features]:
        train_mean = train_df[col].mean()
        train_std = train_df[col].std()
        test_mean = test_df[col].mean()
        test_std = test_df[col].std()

        stats.append({
            'feature': col.replace('_norm', ''),
            'train_mean': train_mean,
            'train_std': train_std,
            'test_mean': test_mean,
            'test_std': test_std,
            'mean_diff': abs(test_mean - train_mean),
            'std_diff': abs(test_std - train_std)
        })

    stats_df = pd.DataFrame(stats)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Mean comparison
    x = np.arange(len(stats_df))
    width = 0.35

    ax1.bar(x - width/2, stats_df['train_mean'], width,
            label='Train Mean', alpha=0.8, color='blue')
    ax1.bar(x + width/2, stats_df['test_mean'], width,
            label='Test Mean', alpha=0.8, color='orange')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Mean Value', fontsize=11)
    ax1.set_title('Feature Means: Train vs Test', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df['feature'], rotation=90, fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Std comparison
    ax2.bar(x - width/2, stats_df['train_std'], width,
            label='Train Std', alpha=0.8, color='blue')
    ax2.bar(x + width/2, stats_df['test_std'], width,
            label='Test Std', alpha=0.8, color='orange')
    ax2.set_ylabel('Standard Deviation', fontsize=11)
    ax2.set_title('Feature Std Dev: Train vs Test', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats_df['feature'], rotation=90, fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    figsize: tuple = (14, 12),
    top_n: int = 20
) -> plt.Figure:
    """
    Plot correlation heatmap for top features

    Args:
        train_df: Training data
        feature_cols: List of feature columns
        figsize: Figure size
        top_n: Number of features to include

    Returns:
        matplotlib Figure object
    """
    # Select features
    features_to_plot = feature_cols[:top_n]

    # Calculate correlation matrix
    corr_matrix = train_df[features_to_plot].corr()

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Clean feature names for labels
    clean_names = [f.replace('_norm', '').replace('_', ' ').title()
                   for f in features_to_plot]

    sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, ax=ax,
                xticklabels=clean_names, yticklabels=clean_names,
                cbar_kws={'label': 'Correlation'})

    ax.set_title(f'Feature Correlation Matrix (Top {top_n} Features)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    return fig


def plot_temporal_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot temporal statistics (samples per date)

    Args:
        train_df: Training data
        test_df: Test data
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Combine data
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()
    train_df_copy['split'] = 'Train'
    test_df_copy['split'] = 'Test'
    full_df = pd.concat([train_df_copy, test_df_copy])

    # Plot 1: Samples per date
    samples_per_date = full_df.groupby(['date', 'split']).size().reset_index(name='count')

    for split in ['Train', 'Test']:
        split_data = samples_per_date[samples_per_date['split'] == split]
        ax1.plot(split_data['date'], split_data['count'],
                label=split, linewidth=2, alpha=0.8)

    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.set_title('Samples Per Date', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative samples
    train_dates = train_df.groupby('date').size().reset_index(name='count')
    test_dates = test_df.groupby('date').size().reset_index(name='count')

    train_dates['cumsum'] = train_dates['count'].cumsum()
    test_dates['cumsum'] = test_dates['count'].cumsum() + train_dates['cumsum'].iloc[-1]

    ax2.plot(train_dates['date'], train_dates['cumsum'],
            label='Train', linewidth=2, alpha=0.8, color='blue')
    ax2.plot(test_dates['date'], test_dates['cumsum'],
            label='Test', linewidth=2, alpha=0.8, color='orange')

    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Cumulative Samples', fontsize=11)
    ax2.set_title('Cumulative Sample Count', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_all_plots(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tickers: List[str],
    feature_cols: List[str],
    output_dir: Optional[Path] = None
) -> dict:
    """
    Create all visualization plots and optionally save them

    Args:
        train_df: Training data
        test_df: Test data
        tickers: List of ticker symbols
        feature_cols: List of feature columns
        output_dir: Optional directory to save plots

    Returns:
        Dictionary of figure objects
    """
    print("\nCreating visualizations...")

    figures = {}

    # Price trends
    print("  1/5 Price trends...")
    figures['price_trends'] = plot_price_trends(train_df, test_df, tickers)

    # Feature distributions
    print("  2/5 Feature distributions...")
    figures['feature_distributions'] = plot_feature_distributions(
        train_df, test_df, feature_cols, n_features=12
    )

    # Feature statistics
    print("  3/5 Feature statistics...")
    figures['feature_stats'] = plot_feature_stats(
        train_df, test_df, feature_cols, n_features=20
    )

    # Correlation heatmap
    print("  4/5 Correlation heatmap...")
    figures['correlation'] = plot_correlation_heatmap(
        train_df, feature_cols, top_n=20
    )

    # Temporal stats
    print("  5/5 Temporal statistics...")
    figures['temporal_stats'] = plot_temporal_stats(train_df, test_df)

    # Save if output directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving plots to {output_dir}...")
        for name, fig in figures.items():
            output_path = output_dir / f'{name}.png'
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ {name}.png")

    print("✓ All visualizations created\n")
    return figures


if __name__ == '__main__':
    """Example usage"""
    print("Visualization module for data preparation pipeline")
    print("\nAvailable functions:")
    print("  - plot_price_trends(): Normalized price trends")
    print("  - plot_feature_distributions(): Feature histograms (train vs test)")
    print("  - plot_feature_stats(): Mean/std comparison")
    print("  - plot_correlation_heatmap(): Feature correlation matrix")
    print("  - plot_temporal_stats(): Temporal sample distribution")
    print("  - create_all_plots(): Generate all plots at once")
