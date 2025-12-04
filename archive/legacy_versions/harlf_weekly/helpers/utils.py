"""
Utility functions for Multi-Hierarchical RL Portfolio System.

This module provides utility functions for:
- Data loading and preprocessing
- Performance metrics calculation
- Result saving and loading
- Visualization helpers
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

from .config import DATA_DIR, TICKERS, N_ASSETS


# ============================================================================
# DATA LOADING
# ============================================================================

def load_features(
    agent_type: str,
    split: str = 'train',
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load feature data for specific agent and split.

    Args:
        agent_type: Type of agent ('technical', 'sentiment', 'macro')
        split: Data split ('train', 'val', 'test')
        data_dir: Data directory (default: from config)

    Returns:
        DataFrame with features
    """
    if data_dir is None:
        data_dir = DATA_DIR

    file_path = data_dir / agent_type / f'{split}.csv'

    if not file_path.exists():
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    df = pd.read_csv(file_path)
    return df


def load_returns(
    split: str = 'train',
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load returns data for specific split.

    Args:
        split: Data split ('train', 'val', 'test')
        data_dir: Data directory (default: from config)

    Returns:
        DataFrame with returns
    """
    if data_dir is None:
        data_dir = DATA_DIR

    file_path = data_dir / f'returns_{split}.csv'

    if not file_path.exists():
        raise FileNotFoundError(f"Returns file not found: {file_path}")

    df = pd.read_csv(file_path)
    return df


def load_metadata(data_dir: Optional[Path] = None) -> Dict:
    """
    Load metadata JSON file.

    Args:
        data_dir: Data directory (default: from config)

    Returns:
        Metadata dictionary
    """
    if data_dir is None:
        data_dir = DATA_DIR

    file_path = data_dir / 'metadata.json'

    if not file_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    with open(file_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def prepare_env_data(
    agent_type: str,
    split: str = 'train',
    data_dir: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for environment (features + returns).

    Args:
        agent_type: Type of agent ('technical', 'sentiment', 'macro')
        split: Data split ('train', 'val', 'test')
        data_dir: Data directory (default: from config)

    Returns:
        Tuple of (features_array, returns_array)
        - features_array: Shape (n_weeks, n_assets, n_features)
        - returns_array: Shape (n_weeks, n_assets)

    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If data shapes are incompatible
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Validate agent type
    valid_agent_types = ['technical', 'sentiment', 'macro']
    if agent_type not in valid_agent_types:
        raise ValueError(
            f"Invalid agent_type '{agent_type}'. Must be one of {valid_agent_types}"
        )

    # Validate split
    valid_splits = ['train', 'val', 'test']
    if split not in valid_splits:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of {valid_splits}"
        )

    # Check if data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please run data preparation scripts first."
        )

    # Load features
    try:
        features_df = load_features(agent_type, split, data_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Feature file missing for agent_type='{agent_type}', split='{split}'.\n"
            f"Expected: {data_dir / agent_type / f'{split}.csv'}\n"
            f"Please run data preparation scripts first."
        ) from e

    # Load returns
    try:
        returns_df = load_returns(split, data_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Returns file missing for split='{split}'.\n"
            f"Expected: {data_dir / f'returns_{split}.csv'}\n"
            f"Please run data preparation scripts first."
        ) from e

    # Exclude non-feature columns
    feature_cols = [col for col in features_df.columns if col not in ['date', 'ticker']]

    # Get unique dates
    dates = sorted(features_df['date'].unique())
    n_weeks = len(dates)
    n_features = len(feature_cols)

    # Initialize arrays
    features_array = np.zeros((n_weeks, N_ASSETS, n_features))
    returns_array = np.zeros((n_weeks, N_ASSETS))

    # Fill features array
    for i, date in enumerate(dates):
        date_data = features_df[features_df['date'] == date]
        for j, ticker in enumerate(TICKERS):
            ticker_data = date_data[date_data['ticker'] == ticker]
            if not ticker_data.empty:
                features_array[i, j, :] = ticker_data[feature_cols].values[0]

    # Fill returns array (returns_df already has columns for each ticker)
    for i, ticker in enumerate(TICKERS):
        if ticker in returns_df.columns:
            returns_array[:, i] = returns_df[ticker].fillna(0).values
        else:
            raise ValueError(
                f"Ticker '{ticker}' not found in returns data.\n"
                f"Available columns: {list(returns_df.columns)}"
            )

    # Validate data alignment
    if features_array.shape[0] != returns_array.shape[0]:
        raise ValueError(
            f"Data length mismatch: features has {features_array.shape[0]} periods "
            f"but returns has {returns_array.shape[0]} periods"
        )

    # Check for NaN values
    if np.isnan(features_array).any():
        n_nans = np.isnan(features_array).sum()
        raise ValueError(
            f"Features contain {n_nans} NaN values. Please clean data first."
        )

    if np.isnan(returns_array).any():
        n_nans = np.isnan(returns_array).sum()
        raise ValueError(
            f"Returns contain {n_nans} NaN values. Please clean data first."
        )

    return features_array, returns_array


def load_features_and_returns(
    agent_type: str,
    split: str = 'train',
    data_dir: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features and returns for a specific agent and split.
    Alias for prepare_env_data for compatibility.

    Args:
        agent_type: Type of agent ('technical', 'sentiment', 'macro')
        split: Data split ('train', 'val', 'test')
        data_dir: Data directory (default: from config)

    Returns:
        Tuple of (features_array, returns_array)
    """
    return prepare_env_data(agent_type, split, data_dir)


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    annualization_factor: float = 52.0
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Factor to annualize (52 for weekly)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    rf_period = risk_free_rate / annualization_factor
    excess_returns = returns - rf_period

    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0:
        return 0.0

    sharpe = mean_excess / std_excess * np.sqrt(annualization_factor)
    return sharpe


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    annualization_factor: float = 52.0
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Factor to annualize

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    rf_period = risk_free_rate / annualization_factor
    excess_returns = returns - rf_period

    mean_excess = excess_returns.mean()
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf if mean_excess > 0 else 0.0

    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0.0

    sortino = mean_excess / downside_std * np.sqrt(annualization_factor)
    return sortino


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Array of returns

    Returns:
        Maximum drawdown (negative value)
    """
    if len(returns) == 0:
        return 0.0

    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max

    max_dd = drawdown.min()
    return max_dd


def calculate_calmar_ratio(
    returns: np.ndarray,
    annualization_factor: float = 52.0
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Args:
        returns: Array of returns
        annualization_factor: Factor to annualize

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    annual_return = returns.mean() * annualization_factor
    max_dd = abs(calculate_max_drawdown(returns))

    if max_dd == 0:
        return np.inf if annual_return > 0 else 0.0

    calmar = annual_return / max_dd
    return calmar


def calculate_win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (% of positive periods).

    Args:
        returns: Array of returns

    Returns:
        Win rate (0 to 1)
    """
    if len(returns) == 0:
        return 0.0

    win_rate = (returns > 0).mean()
    return win_rate


def calculate_all_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.04,
    annualization_factor: float = 52.0
) -> Dict[str, float]:
    """
    Calculate all performance metrics.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Factor to annualize

    Returns:
        Dictionary of metrics
    """
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'max_drawdown': 0.0,
            'calmar': 0.0,
            'win_rate': 0.0,
            'n_periods': 0
        }

    total_return = (1 + returns).prod() - 1
    annual_return = returns.mean() * annualization_factor
    annual_vol = returns.std() * np.sqrt(annualization_factor)

    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe': calculate_sharpe_ratio(returns, risk_free_rate, annualization_factor),
        'sortino': calculate_sortino_ratio(returns, risk_free_rate, annualization_factor),
        'max_drawdown': calculate_max_drawdown(returns),
        'calmar': calculate_calmar_ratio(returns, annualization_factor),
        'win_rate': calculate_win_rate(returns),
        'n_periods': len(returns)
    }

    return metrics


# ============================================================================
# RESULT SAVING/LOADING
# ============================================================================

def save_agent_results(
    agent_name: str,
    results: Dict,
    results_dir: Optional[Path] = None
):
    """
    Save agent results to JSON file.

    Args:
        agent_name: Name of agent (e.g., 'technical_ema_sharpe')
        results: Dictionary of results to save
        models_dir: Models directory (default: from config)
    """
    if results_dir is None:
        from .config import MODELS_DIR
        results_dir = MODELS_DIR

    # Add timestamp
    results['saved_at'] = datetime.now().isoformat()

    # Save to JSON
    file_path = results_dir / f'{agent_name}_results.json'
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to {file_path}")


def compile_base_agent_results(
    agent_type: str,
    agent_name: str,
    reward_type: str,
    training_history: Dict,
    train_results: Dict,
    val_results: Dict,
) -> Dict:
    """
    Compile base agent results into standardized format for saving.

    Args:
        agent_type: Type of agent ('technical' or 'sentiment')
        agent_name: Name of agent (e.g., 'technical_ema_sharpe')
        reward_type: Reward type used (e.g., 'ema_sharpe')
        training_history: Training history dictionary
        train_results: Training evaluation results from evaluate_agent()
        val_results: Validation evaluation results from evaluate_agent()
        test_results: Test evaluation results from evaluate_agent()

    Returns:
        Complete results dictionary ready for saving
    """
    return {
        'agent_type': agent_type,
        'agent_name': agent_name,
        'reward_type': reward_type,
        'training_history': training_history,
        'performance': {
            'train': {**train_results['metrics'], 'returns': train_results['returns']},
            'val': {**val_results['metrics'], 'returns': val_results['returns']},
        }
    }


def load_agent_results(
    agent_name: str,
    results_dir: Optional[Path] = None
) -> Dict:
    """
    Load agent results from JSON file.

    Args:
        agent_name: Name of agent
        models_dir: Models directory (default: from config)

    Returns:
        Dictionary of results
    """
    if results_dir is None:
        from .config import MODELS_DIR
        results_dir = MODELS_DIR

    file_path = results_dir / f'{agent_name}_results.json'

    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    with open(file_path, 'r') as f:
        results = json.load(f)

    return results


def print_metrics(metrics: Dict[str, float], title: str = "Performance Metrics"):
    """
    Pretty print performance metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title for printout
    """
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)

    print(f"Total Return:       {metrics.get('total_return', 0)*100:>8.2f}%")
    print(f"Annual Return:      {metrics.get('annual_return', 0)*100:>8.2f}%")
    print(f"Annual Volatility:  {metrics.get('annual_volatility', 0)*100:>8.2f}%")
    print(f"Sharpe Ratio:       {metrics.get('sharpe', 0):>8.2f}")
    print(f"Sortino Ratio:      {metrics.get('sortino', 0):>8.2f}")
    print(f"Max Drawdown:       {metrics.get('max_drawdown', 0)*100:>8.2f}%")
    print(f"Calmar Ratio:       {metrics.get('calmar', 0):>8.2f}")
    print(f"Win Rate:           {metrics.get('win_rate', 0)*100:>8.2f}%")
    print(f"N Periods:          {metrics.get('n_periods', 0):>8d}")
    print("="*70 + "\n")


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def plot_equity_curve(
    returns: np.ndarray,
    dates: Optional[List] = None,
    benchmark_returns: Optional[np.ndarray] = None,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot equity curve.

    Args:
        returns: Array of returns
        dates: List of dates (optional)
        benchmark_returns: Benchmark returns (optional)
        title: Plot title
        figsize: Figure size
    """
    cumulative = (1 + returns).cumprod()

    plt.figure(figsize=figsize)

    if dates is None:
        dates = range(len(returns))

    plt.plot(dates, cumulative, label='Strategy', linewidth=2)

    if benchmark_returns is not None:
        bench_cumulative = (1 + benchmark_returns).cumprod()
        plt.plot(dates, bench_cumulative, label='Benchmark',
                 linewidth=2, linestyle='--', alpha=0.7)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_drawdown(
    returns: np.ndarray,
    dates: Optional[List] = None,
    title: str = "Drawdown",
    figsize: Tuple[int, int] = (14, 4)
):
    """
    Plot drawdown over time.

    Args:
        returns: Array of returns
        dates: List of dates (optional)
        title: Plot title
        figsize: Figure size
    """
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max

    plt.figure(figsize=figsize)

    if dates is None:
        dates = range(len(returns))

    plt.fill_between(dates, drawdown * 100, 0, color='red', alpha=0.3)
    plt.plot(dates, drawdown * 100, color='darkred', linewidth=1.5)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_returns_distribution(
    returns: np.ndarray,
    title: str = "Returns Distribution",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot returns distribution with histogram and KDE.

    Args:
        returns: Array of returns
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    plt.hist(returns * 100, bins=50, alpha=0.6, color='steelblue',
             edgecolor='black', density=True, label='Histogram')

    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(returns * 100)
    x_range = np.linspace(returns.min() * 100, returns.max() * 100, 200)
    plt.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    plt.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Return (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_allocation_heatmap(
    allocations: np.ndarray,
    tickers: List[str],
    dates: Optional[List] = None,
    title: str = "Portfolio Allocation",
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot portfolio allocation heatmap over time.

    Args:
        allocations: Array of allocations (n_periods, n_assets)
        tickers: List of ticker symbols
        dates: List of dates (optional)
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Transpose for plotting (tickers on y-axis, time on x-axis)
    data = allocations.T * 100  # Convert to percentage

    if dates is not None:
        # Subsample dates for readability if too many
        n_dates = len(dates)
        if n_dates > 20:
            step = n_dates // 20
            xticks_idx = range(0, n_dates, step)
            xticks_labels = [dates[i] for i in xticks_idx]
        else:
            xticks_idx = range(n_dates)
            xticks_labels = dates
    else:
        xticks_idx = None
        xticks_labels = None

    sns.heatmap(data, yticklabels=tickers, cmap='RdYlGn',
                center=100/len(tickers), vmin=0, vmax=100,
                cbar_kws={'label': 'Allocation (%)'})

    if xticks_labels is not None:
        plt.xticks(xticks_idx, xticks_labels, rotation=45, ha='right')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Assets')
    plt.tight_layout()
    plt.show()


def plot_hierarchical_agent_training(
    training_history: Dict,
    agent_name: str = 'Agent',
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Plot training history for Super Agent or Meta Agent.

    This function is designed for hierarchical agents (Super/Meta) that only
    evaluate on validation set during training (no train set evaluation).

    Args:
        training_history: Dict with keys:
            - 'val_rewards': List of validation rewards per evaluation
            - 'val_sharpe': List of validation Sharpe ratios per evaluation
            - 'val_returns': List of validation annual returns per evaluation
            - 'val_drawdown': List of validation max drawdowns per evaluation
        agent_name: Name for plot title (e.g., 'Super Agent', 'Meta Agent')
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract data
    val_rewards = training_history['val_rewards']
    val_sharpe = training_history['val_sharpe']
    val_returns = training_history['val_returns']
    val_drawdown = training_history['val_drawdown']

    evaluations = list(range(1, len(val_rewards) + 1))

    # Plot 1: Rewards
    axes[0, 0].plot(evaluations, val_rewards, label='Validation',
                    marker='o', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Evaluation #', fontsize=11)
    axes[0, 0].set_ylabel('Mean Episode Reward', fontsize=11)
    axes[0, 0].set_title(f'{agent_name} - Validation Rewards',
                         fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Sharpe Ratios
    axes[0, 1].plot(evaluations, val_sharpe, label='Validation',
                    marker='o', alpha=0.7, linewidth=2, color='green')
    axes[0, 1].axhline(y=2.0, color='r', linestyle='--',
                       label='Target (2.0)', alpha=0.5)
    axes[0, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 1].set_xlabel('Evaluation #', fontsize=11)
    axes[0, 1].set_ylabel('Sharpe Ratio', fontsize=11)
    axes[0, 1].set_title(f'{agent_name} - Sharpe Ratio Progress',
                         fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Annual Returns
    axes[1, 0].plot(evaluations, [r * 100 for r in val_returns],
                    label='Validation', marker='o', alpha=0.7,
                    linewidth=2, color='purple')
    axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('Evaluation #', fontsize=11)
    axes[1, 0].set_ylabel('Annualized Return (%)', fontsize=11)
    axes[1, 0].set_title(f'{agent_name} - Annualized Returns',
                         fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Max Drawdown
    axes[1, 1].plot(evaluations, [d * 100 for d in val_drawdown],
                    label='Validation', marker='o', alpha=0.7,
                    linewidth=2, color='red')
    axes[1, 1].axhline(y=-20, color='orange', linestyle='--',
                       label='Warning (-20%)', alpha=0.5)
    axes[1, 1].set_xlabel('Evaluation #', fontsize=11)
    axes[1, 1].set_ylabel('Max Drawdown (%)', fontsize=11)
    axes[1, 1].set_title(f'{agent_name} - Maximum Drawdown',
                         fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\n{agent_name} Training Summary:")
    print("=" * 70)
    print(f"Best Validation Reward: {max(val_rewards):.4f}")
    print(f"Best Validation Sharpe: {max(val_sharpe):.3f}")
    print(f"Best Validation Return: {max(val_returns)*100:.2f}%")
    print(f"Best Max Drawdown: {max(val_drawdown)*100:.2f}%")
    print(f"\nFinal Validation Metrics:")
    print(f"  Reward: {val_rewards[-1]:.4f}")
    print(f"  Sharpe: {val_sharpe[-1]:.3f}")
    print(f"  Return: {val_returns[-1]*100:.2f}%")
    print(f"  Max DD: {val_drawdown[-1]*100:.2f}%")


# ============================================================================
# NOTEBOOK UTILITIES
# ============================================================================

def verify_required_files(files_dict: Dict[str, Path]) -> bool:
    """
    Verify existence of required files for notebook execution.

    Args:
        files_dict: Dictionary mapping file description to file path
                   e.g., {'Technical Agent': Path('models/tech.zip')}

    Returns:
        True if all files exist, False otherwise

    Example:
        >>> files = {
        ...     'Technical Agent': MODELS_DIR / 'technical.zip',
        ...     'Returns Data': DATA_DIR / 'returns_train.csv'
        ... }
        >>> if not verify_required_files(files):
        ...     print("Missing files! Run previous notebooks first.")
    """
    print("Checking required files:")
    all_exist = True

    for name, path in files_dict.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n⚠️  Missing files! Please run previous notebooks first.")
    else:
        print("\n✓ All required files found!")

    return all_exist


def print_metadata_summary(
    metadata: Dict,
    agent_config=None,
    agent_name: str = "Agent"
):
    """
    Print formatted summary of metadata and configuration.

    Args:
        metadata: Metadata dictionary loaded from metadata.json
        agent_config: Agent configuration class (e.g., SuperAgentConfig)
        agent_name: Name of agent for display (e.g., "Super Agent")

    Example:
        >>> with open('data/metadata.json') as f:
        ...     metadata = json.load(f)
        >>> print_metadata_summary(metadata, SuperAgentConfig, "Super Agent")
    """
    print("Portfolio Configuration:")
    print(f"  Tickers: {', '.join(metadata['tickers'])}")
    print(f"  Benchmark: {metadata['benchmark']}")

    print(f"\nData Splits:")
    for split in ['train', 'val', 'test']:
        s = metadata['splits'][split]
        print(f"  {split.capitalize():5s}: {s['start']} to {s['end']} ({s['weeks']} weeks)")

    if agent_config is not None:
        print(f"\n{agent_name} Configuration:")
        print(f"  Learning Rate: {agent_config.LEARNING_RATE}")
        print(f"  Total Timesteps: {agent_config.TOTAL_TIMESTEPS:,}")
        print(f"  Network Architecture: {agent_config.NET_ARCH}")
        print(f"  Batch Size: {agent_config.BATCH_SIZE}")
        print(f"  Eval Frequency: {agent_config.EVAL_FREQ:,} steps")
        print(f"  Early Stopping Patience: {agent_config.PATIENCE} evaluations")


def compare_agents_across_splits(
    agent_results: Dict[str, Dict],
    splits: List[str] = ['train', 'val', 'test']
) -> pd.DataFrame:
    """
    Create comparison dataframe for multiple agents across data splits.

    Args:
        agent_results: Dictionary mapping agent names to their results
                      Each result dict should have keys for each split containing metrics
        splits: List of data splits to compare (default: train/val/test)

    Returns:
        DataFrame with comparison across agents and splits

    Example:
        >>> results = {
        ...     'Technical': tech_results['performance'],
        ...     'Sentiment': sent_results['performance'],
        ...     'Super': super_results
        ... }
        >>> df = compare_agents_across_splits(results)
        >>> print(df.to_string(index=False))
    """
    comparison_data = []

    for split in splits:
        for agent_name, results in agent_results.items():
            split_results = results.get(split, {})

            comparison_data.append({
                'Split': split.capitalize(),
                'Agent': agent_name,
                'Sharpe': split_results.get('sharpe', np.nan),
                'Return': split_results.get('annual_return', np.nan),
                'Volatility': split_results.get('annual_volatility', np.nan),
                'Max DD': split_results.get('max_drawdown', np.nan),
                'Calmar': split_results.get('calmar', np.nan),
                'Win Rate': split_results.get('win_rate', np.nan)
            })

    return pd.DataFrame(comparison_data)


def analyze_blending_weights(
    model,
    env,
    agent_names: List[str] = ['Technical', 'Sentiment'],
    plot: bool = True,
    figsize: Tuple[int, int] = (14, 8)
) -> np.ndarray:
    """
    Analyze and visualize agent blending weights from Super Agent.

    Args:
        model: Trained Super Agent model
        env: Super Agent environment
        agent_names: Names of base agents being blended
        plot: Whether to create visualization
        figsize: Figure size for plot

    Returns:
        Array of blending weights (n_steps, n_agents)

    Example:
        >>> weights = analyze_blending_weights(
        ...     super_model, test_env,
        ...     agent_names=['Technical', 'Sentiment']
        ... )
        >>> print(f"Mean Technical weight: {weights[:, 0].mean():.3f}")
    """
    print("Analyzing blending strategy...\n")

    # Collect weights
    obs, _ = env.reset()
    blending_weights = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # Normalize to sum to 1
        action_sum = action.sum()
        if action_sum > 1e-10:
            normalized_weights = action / action.sum()
        else:
            normalized_weights = np.ones(len(agent_names)) / len(agent_names)

        blending_weights.append(normalized_weights)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated

    blending_weights = np.array(blending_weights)

    # Print statistics
    print("Blending Weight Statistics:")
    for i, name in enumerate(agent_names):
        print(f"\n{name}:")
        print(f"  Mean: {blending_weights[:, i].mean():.3f}")
        print(f"  Std:  {blending_weights[:, i].std():.3f}")
        print(f"  Min:  {blending_weights[:, i].min():.3f}")
        print(f"  Max:  {blending_weights[:, i].max():.3f}")

    # Interpretation
    if len(agent_names) == 2:
        mean_0 = blending_weights[:, 0].mean()
        if mean_0 > 0.6:
            print(f"\n📊 Interpretation: Strongly favors {agent_names[0]} signals")
        elif mean_0 < 0.4:
            print(f"\n📊 Interpretation: Strongly favors {agent_names[1]} signals")
        else:
            print("\n📊 Interpretation: Uses balanced blending")

        if blending_weights.std() > 0.2:
            print("   Dynamic strategy - weights adapt significantly over time")
        else:
            print("   Static strategy - weights remain relatively constant")

    # Plot if requested
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # Weights over time
        for i, name in enumerate(agent_names):
            axes[0].plot(blending_weights[:, i], label=f'{name} Weight', alpha=0.7)
        axes[0].axhline(y=1/len(agent_names), color='gray', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Week')
        axes[0].set_ylabel('Weight')
        axes[0].set_title('Blending Weights Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Weight distribution
        for i, name in enumerate(agent_names):
            axes[1].hist(blending_weights[:, i], bins=20, alpha=0.7, label=f'{name} Weight')
        axes[1].axvline(x=1/len(agent_names), color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Weight')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Blending Weights')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return blending_weights


def save_hierarchical_results(
    results: Dict[str, Dict],
    filepath: Path,
    additional_data: Optional[Dict] = None
):
    """
    Save hierarchical agent results to JSON with proper float conversion.

    Args:
        results: Dictionary with 'train', 'val', 'test' keys containing metrics
        filepath: Path to save JSON file
        additional_data: Optional additional data to include (e.g., training_history)

    Example:
        >>> save_hierarchical_results(
        ...     results={'train': {...}, 'val': {...}, 'test': {...}},
        ...     filepath=Path('models/super_agent_results.json'),
        ...     additional_data={
        ...         'training_history': history,
        ...         'blending_weights_mean': {'technical': 0.5, 'sentiment': 0.5}
        ...     }
        ... )
    """
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    results_to_save = {}

    # Convert results for each split
    for split in ['train', 'val', 'test']:
        if split in results:
            split_data = results[split]
            results_to_save[split] = {
                'sharpe': float(split_data.get('sharpe', 0)),
                'annual_return': float(split_data.get('annual_return', 0)),
                'annual_volatility': float(split_data.get('annual_volatility', 0)),
                'max_drawdown': float(split_data.get('max_drawdown', 0)),
                'calmar': float(split_data.get('calmar', 0)),
                'win_rate': float(split_data.get('win_rate', 0)),
                'returns': split_data.get('returns', [])
            }

    # Add additional data with conversion
    if additional_data:
        converted_additional = convert_to_json_serializable(additional_data)
        results_to_save.update(converted_additional)

    # Add timestamp
    from datetime import datetime
    results_to_save['saved_at'] = datetime.now().isoformat()

    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"✓ Results saved to {filepath}")


def plot_agent_comparison(
    agent_results: Dict[str, Dict],
    split: str = 'val',
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Create comprehensive visual comparison of multiple agents.

    Args:
        agent_results: Dictionary mapping agent names to their results
                      Each result dict should have 'returns', 'sharpe', metrics
        split: Which data split to visualize ('train', 'val', 'test')
               Default 'val' - use validation during development, 'test' only for final backtest
        figsize: Figure size

    Best Practice:
        - During development (notebooks 02-04): Use split='val' to avoid test set leakage
        - Final evaluation (notebook 05): Use split='test' for out-of-sample results

    Example:
        >>> # During development - use validation
        >>> results = {
        ...     'Technical': tech_results['performance']['val'],
        ...     'Sentiment': sent_results['performance']['val'],
        ...     'Super': super_results['val']
        ... }
        >>> plot_agent_comparison(results, split='val')
        >>>
        >>> # Final backtest only - use test
        >>> plot_agent_comparison(results, split='test')
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 1. Cumulative Returns
    for agent_name, result in agent_results.items():
        returns = result.get('returns', [])
        cumulative = (1 + pd.Series(returns)).cumprod()
        axes[0, 0].plot(cumulative, label=agent_name, alpha=0.7, linewidth=2)

    axes[0, 0].set_xlabel('Week')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].set_title(f'{split.capitalize()} Set: Cumulative Returns', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Sharpe Ratio Comparison
    agent_names = list(agent_results.keys())
    sharpes = [agent_results[name].get('sharpe', 0) for name in agent_names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(agent_names)))

    bars = axes[0, 1].bar(agent_names, sharpes, color=colors, alpha=0.7)
    axes[0, 1].axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Target (2.0)')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].set_title('Sharpe Ratio Comparison', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, sharpe in zip(bars, sharpes):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{sharpe:.2f}',
                       ha='center', va='bottom', fontsize=10)

    # 3. Drawdown Comparison
    for agent_name, result in agent_results.items():
        returns = result.get('returns', [])
        cumulative = pd.Series(returns).cumsum()
        drawdown = cumulative - cumulative.cummax()
        axes[0, 2].fill_between(range(len(drawdown)), drawdown, 0,
                               alpha=0.3, label=agent_name)

    axes[0, 2].set_xlabel('Week')
    axes[0, 2].set_ylabel('Drawdown')
    axes[0, 2].set_title('Drawdown Comparison', fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Return Distribution
    for agent_name, result in agent_results.items():
        returns = result.get('returns', [])
        axes[1, 0].hist(returns, bins=20, alpha=0.5, label=agent_name, density=True)

    axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Weekly Return')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Return Distribution', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Risk-Return Space
    for agent_name, result in agent_results.items():
        ret = result.get('annual_return', 0)
        vol = result.get('annual_volatility', 0)
        axes[1, 1].scatter(vol, ret, s=200, alpha=0.7, label=agent_name)
        axes[1, 1].annotate(agent_name, (vol, ret), fontsize=9, ha='center')

    axes[1, 1].set_xlabel('Annual Volatility')
    axes[1, 1].set_ylabel('Annual Return')
    axes[1, 1].set_title('Risk-Return Space', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Performance Metrics Comparison
    metrics = ['Sharpe', 'Return', 'Calmar', 'Win Rate']
    x = np.arange(len(metrics))
    width = 0.8 / len(agent_names)

    for i, agent_name in enumerate(agent_names):
        result = agent_results[agent_name]
        # Normalize metrics to 0-1 scale for visualization
        values = [
            result.get('sharpe', 0) / 5.0,  # Sharpe (normalized to max 5)
            result.get('annual_return', 0),  # Return (already 0-1 range typically)
            min(result.get('calmar', 0) / 10.0, 1.0),  # Calmar (normalized to max 10)
            result.get('win_rate', 0)  # Win rate (already 0-1)
        ]
        offset = (i - len(agent_names)/2) * width + width/2
        axes[1, 2].bar(x + offset, values, width, label=agent_name, alpha=0.7)

    axes[1, 2].set_ylabel('Normalized Value')
    axes[1, 2].set_title('Performance Metrics', fontweight='bold')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\n{split.upper()} SET SUMMARY:")
    print("=" * 80)
    for agent_name, result in agent_results.items():
        print(f"\n{agent_name}:")
        print(f"  Sharpe:     {result.get('sharpe', 0):.3f}")
        print(f"  Return:     {result.get('annual_return', 0):.2%}")
        print(f"  Volatility: {result.get('annual_volatility', 0):.2%}")
        print(f"  Max DD:     {result.get('max_drawdown', 0):.2%}")
        print(f"  Calmar:     {result.get('calmar', 0):.3f}")
        print(f"  Win Rate:   {result.get('win_rate', 0):.2%}")


def analyze_regime_performance(
    returns: pd.Series,
    macro_df: pd.DataFrame,
    plot: bool = True
) -> Dict[str, Dict]:
    """
    Analyze agent performance across different macroeconomic regimes.

    Args:
        returns: Series of returns to analyze
        macro_df: DataFrame with macro features (must include 'vix_regime', 'yield_curve_2_10')
        plot: Whether to create visualization

    Returns:
        Dictionary with regime analysis results

    Example:
        >>> regime_stats = analyze_regime_performance(
        ...     returns=pd.Series(meta_results['test']['returns']),
        ...     macro_df=macro_test_df
        ... )
        >>> print(f"High VIX Sharpe: {regime_stats['high_vix']['sharpe']:.3f}")
    """
    # Ensure lengths match
    n_returns = len(returns)
    macro_df = macro_df.iloc[:n_returns]

    # VIX regime analysis
    low_vix_mask = macro_df['vix_regime'] == 0
    high_vix_mask = macro_df['vix_regime'] == 1

    low_vix_returns = returns[low_vix_mask]
    high_vix_returns = returns[high_vix_mask]

    # Yield curve analysis
    normal_curve_mask = macro_df['yield_curve_2_10'] > 0
    inverted_curve_mask = macro_df['yield_curve_2_10'] <= 0

    normal_curve_returns = returns[normal_curve_mask]
    inverted_curve_returns = returns[inverted_curve_mask]

    # Calculate statistics
    def calc_regime_stats(ret_series):
        if len(ret_series) == 0:
            return {
                'sharpe': 0.0,
                'mean_return': 0.0,
                'std': 0.0,
                'win_rate': 0.0,
                'count': 0
            }

        ret_std = ret_series.std()
        if ret_std == 0:
            return {
                'sharpe': 0.0,
                'mean_return': float(ret_series.mean()),
                'std': 0.0,
                'win_rate': float((ret_series > 0).mean()),
                'count': int(len(ret_series))
            }

        return {
            'sharpe': float(ret_series.mean() / ret_std * np.sqrt(52)),
            'mean_return': float(ret_series.mean()),
            'std': float(ret_std),
            'win_rate': float((ret_series > 0).mean()),
            'count': int(len(ret_series))
        }

    regime_stats = {
        'low_vix': calc_regime_stats(low_vix_returns),
        'high_vix': calc_regime_stats(high_vix_returns),
        'normal_curve': calc_regime_stats(normal_curve_returns),
        'inverted_curve': calc_regime_stats(inverted_curve_returns)
    }

    # Print results
    print("Performance by VIX Regime:")
    print("=" * 60)
    print(f"\nLow VIX Regime:")
    print(f"  Mean Return: {regime_stats['low_vix']['mean_return']:.4f}")
    print(f"  Std Dev: {regime_stats['low_vix']['std']:.4f}")
    print(f"  Sharpe: {regime_stats['low_vix']['sharpe']:.3f}")
    print(f"  Win Rate: {regime_stats['low_vix']['win_rate']:.2%}")
    print(f"  Count: {regime_stats['low_vix']['count']} weeks")

    print(f"\nHigh VIX Regime:")
    print(f"  Mean Return: {regime_stats['high_vix']['mean_return']:.4f}")
    print(f"  Std Dev: {regime_stats['high_vix']['std']:.4f}")
    print(f"  Sharpe: {regime_stats['high_vix']['sharpe']:.3f}")
    print(f"  Win Rate: {regime_stats['high_vix']['win_rate']:.2%}")
    print(f"  Count: {regime_stats['high_vix']['count']} weeks")

    print("\n\nPerformance by Yield Curve:")
    print("=" * 60)
    print(f"\nNormal Yield Curve (2Y < 10Y):")
    print(f"  Mean Return: {regime_stats['normal_curve']['mean_return']:.4f}")
    print(f"  Sharpe: {regime_stats['normal_curve']['sharpe']:.3f}")
    print(f"  Win Rate: {regime_stats['normal_curve']['win_rate']:.2%}")
    print(f"  Count: {regime_stats['normal_curve']['count']} weeks")

    print(f"\nInverted Yield Curve (2Y >= 10Y):")
    print(f"  Mean Return: {regime_stats['inverted_curve']['mean_return']:.4f}")
    print(f"  Sharpe: {regime_stats['inverted_curve']['sharpe']:.3f}")
    print(f"  Win Rate: {regime_stats['inverted_curve']['win_rate']:.2%}")
    print(f"  Count: {regime_stats['inverted_curve']['count']} weeks")

    # Plot if requested
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # VIX regime comparison
        axes[0].bar(['Low VIX', 'High VIX'],
                    [regime_stats['low_vix']['mean_return'], regime_stats['high_vix']['mean_return']],
                    color=['green', 'red'], alpha=0.7)
        axes[0].set_ylabel('Mean Weekly Return')
        axes[0].set_title('Returns by VIX Regime')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Yield curve comparison
        axes[1].bar(['Normal', 'Inverted'],
                    [regime_stats['normal_curve']['mean_return'], regime_stats['inverted_curve']['mean_return']],
                    color=['blue', 'orange'], alpha=0.7)
        axes[1].set_ylabel('Mean Weekly Return')
        axes[1].set_title('Returns by Yield Curve')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    return regime_stats


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    """Test utility functions."""

    print("Testing Utility Functions")
    print("="*70)

    # Test metrics with synthetic returns
    np.random.seed(42)
    returns = 0.01 + 0.02 * np.random.randn(100)

    print("\n1. Calculate All Metrics:")
    metrics = calculate_all_metrics(returns)
    print_metrics(metrics)

    print("\n2. Individual Metrics:")
    print(f"   Sharpe: {calculate_sharpe_ratio(returns):.2f}")
    print(f"   Sortino: {calculate_sortino_ratio(returns):.2f}")
    print(f"   Max DD: {calculate_max_drawdown(returns)*100:.2f}%")
    print(f"   Calmar: {calculate_calmar_ratio(returns):.2f}")
    print(f"   Win Rate: {calculate_win_rate(returns)*100:.2f}%")

    print("\n" + "="*70)
    print("✅ All utility functions tested successfully")
