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
    """
    # Load features
    features_df = load_features(agent_type, split, data_dir)
    returns_df = load_returns(split, data_dir)

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
    test_results: Dict
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
            'test': {**test_results['metrics'], 'returns': test_results['returns']}
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
