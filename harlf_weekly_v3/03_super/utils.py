"""
Utility Functions
=================
Helper functions for loading data, computing metrics, etc.

CHANGES IN THIS VERSION:
- Added compute_risk_metrics() for additional performance measures
- Enhanced print_results() to show more metrics
- Added plot_training_history() for visualization
"""

import json
import numpy as np
import pandas as pd
import math
from pathlib import Path


def load_data(data_dir, agent_type, split):
    """
    Load data for training.
    
    Parameters
    ----------
    data_dir : str
        Path to data directory
    agent_type : str
        'technical' or 'sentiment'
    split : str
        'train', 'val', or 'test'
    
    Returns
    -------
    features_df : DataFrame
        Features with 'ticker' column
    returns_df : DataFrame
        Returns (columns = tickers)
    tickers : list
        List of tickers
    feature_cols : list
        List of feature columns
    
    Examples
    --------
    >>> features, returns, tickers, cols = load_data('data_hierarchical', 'technical', 'train')
    """
    
    data_dir = Path(data_dir)
    
    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    tickers = metadata['tickers']
    
    # Load features based on agent type
    if agent_type == 'technical':
        features_path = data_dir / 'technical' / f'{split}.csv'
        feature_cols = metadata.get('technical_indicator_features',
                                     [c for c in metadata['technical_features'] 
                                      if c not in ['open', 'high', 'low', 'close', 'volume', 'return']])
    else:  # sentiment
        features_path = data_dir / 'sentiment' / f'{split}.csv'
        feature_cols = metadata.get('sentiment_indicator_features',
                                     [c for c in metadata['sentiment_features']
                                      if c not in ['open', 'high', 'low', 'close', 'volume', 'return']])
    
    features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    
    # Load returns
    returns_path = data_dir / f'returns_{split}.csv'
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    return features_df, returns_df, tickers, feature_cols


def compute_sharpe(returns):
    """
    Compute annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : array-like
        Array of returns
    
    Returns
    -------
    float : Sharpe ratio
    
    Examples
    --------
    >>> sharpe = compute_sharpe([0.01, 0.02, -0.01, 0.03])
    """
    
    returns = np.array(returns)
    if len(returns) < 2:
        return 0.0
    
    mean = returns.mean()
    std = returns.std()
    
    if std < 1e-12:
        return 0.0
    
    return float((mean / std) * math.sqrt(52.0))


def compute_risk_metrics(returns):
    """
    NEW: Compute comprehensive risk metrics.
    
    Parameters
    ----------
    returns : array-like
        Array of returns
    
    Returns
    -------
    dict : Dictionary of risk metrics
    
    Examples
    --------
    >>> metrics = compute_risk_metrics([0.01, 0.02, -0.01, 0.03])
    """
    
    returns = np.array(returns)
    if len(returns) < 2:
        return {
            'sharpe': 0.0,
            'sortino': 0.0,
            'max_drawdown': 0.0,
            'calmar': 0.0,
            'total_return': 0.0,
            'volatility': 0.0
        }
    
    # Basic metrics
    mean_return = returns.mean()
    std_return = returns.std()
    total_return = returns.sum()
    
    # Sharpe ratio
    sharpe = (mean_return / std_return) * math.sqrt(52) if std_return > 0 else 0.0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 1 else std_return
    sortino = (mean_return / downside_std) * math.sqrt(52) if downside_std > 0 else 0.0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    # Calmar ratio (return / max drawdown)
    annual_return = mean_return * 52
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0
    
    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_drawdown),
        'calmar': float(calmar),
        'total_return': float(total_return),
        'volatility': float(std_return * math.sqrt(52))
    }


def save_results(results, filepath):
    """
    Save results to JSON file.
    
    Parameters
    ----------
    results : dict
        Results dictionary
    filepath : str
        Path to save file
    
    Examples
    --------
    >>> save_results(results, 'results/my_experiment.json')
    """
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Results saved to: {filepath}")


def print_results(results):
    """
    Print training results nicely.
    
    Parameters
    ----------
    results : dict
        Results from training
    
    Examples
    --------
    >>> print_results(result)
    """
    
    print(f"\n{'='*60}")
    print(f"Results: {results['agent_type']} - {results['algorithm']} - {results['reward_type']}")
    print(f"{'='*60}")
    print(f"Train Sharpe:  {results['train_sharpe']:.3f}")
    print(f"Val Sharpe:    {results['val_sharpe']:.3f}")
    print(f"Test Sharpe:   {results['test_sharpe']:.3f}")
    
    # NEW: Print additional parameters if available
    if 'gamma' in results:
        print(f"\nParameters:")
        print(f"  Gamma: {results['gamma']:.3f}")
    if 'softmax_temperature' in results:
        print(f"  Softmax temp: {results['softmax_temperature']:.2f}")
    if 'transaction_cost' in results and results['transaction_cost'] > 0:
        print(f"  Transaction cost: {results['transaction_cost']*10000:.1f} bps")
    
    print(f"\nModel saved:   {results['model_path']}")
    print(f"{'='*60}\n")


def compare_results(results_list):
    """
    Compare multiple results.
    
    Parameters
    ----------
    results_list : list of dict
        List of result dictionaries
    
    Returns
    -------
    DataFrame : Comparison table sorted by test Sharpe ratio
    
    Examples
    --------
    >>> df = compare_results([result1, result2, result3])
    """
    
    df = pd.DataFrame(results_list)
    
    # Select columns to display
    base_cols = ['agent_type', 'algorithm', 'reward_type', 'train_sharpe', 'val_sharpe', 'test_sharpe']
    extra_cols = []
    
    # Add extra columns if they exist
    if 'gamma' in df.columns:
        extra_cols.append('gamma')
    if 'softmax_temperature' in df.columns:
        extra_cols.append('softmax_temperature')
    if 'transaction_cost' in df.columns:
        extra_cols.append('transaction_cost')
    
    display_cols = base_cols + extra_cols
    display_cols = [c for c in display_cols if c in df.columns]
    
    df_display = df[display_cols].sort_values('test_sharpe', ascending=False)
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(df_display.to_string(index=False))
    print("="*80 + "\n")
    
    return df_display


def plot_training_history(callback, save_path=None):
    """
    NEW: Plot training history from callback.
    
    Parameters
    ----------
    callback : ValidationCallback
        Callback with eval_history
    save_path : str, optional
        Path to save plot
    
    Examples
    --------
    >>> plot_training_history(callback, 'training_history.png')
    """
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not available, skipping plot")
        return
    
    if not hasattr(callback, 'eval_history') or not callback.eval_history:
        print("⚠️  No evaluation history available")
        return
    
    history = pd.DataFrame(callback.eval_history)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['step'], history['val_sharpe'], label='Validation Sharpe', marker='o')
    plt.plot(history['step'], history['best_sharpe'], label='Best Sharpe', linestyle='--')
    plt.xlabel('Training Steps')
    plt.ylabel('Sharpe Ratio')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_comparison_table(results_dict, metric='test_sharpe'):
    """
    NEW: Create a formatted comparison table.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of results by approach
    metric : str
        Metric to compare (default: 'test_sharpe')
    
    Returns
    -------
    DataFrame : Formatted comparison table
    
    Examples
    --------
    >>> table = create_comparison_table({
    ...     'ema_sharpe': {'technical': result1, 'sentiment': result2},
    ...     'multi_objective': {'technical': result3, 'sentiment': result4}
    ... })
    """
    
    data = []
    for approach, agents in results_dict.items():
        row = {'approach': approach}
        if 'technical' in agents:
            row['technical'] = agents['technical'].get(metric, 0.0)
        if 'sentiment' in agents:
            row['sentiment'] = agents['sentiment'].get(metric, 0.0)
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    print("Utility functions loaded!")
    print("\n⚠️  REFACTORED VERSION - Key changes:")
    print("  - Added compute_risk_metrics() for comprehensive metrics")
    print("  - Enhanced print_results() with more info")
    print("  - Added plot_training_history() for visualization")
    print("\nAvailable functions:")
    print("  - load_data(): Load training data")
    print("  - compute_sharpe(): Calculate Sharpe ratio")
    print("  - compute_risk_metrics(): Calculate multiple risk metrics (NEW)")
    print("  - save_results(): Save results to JSON")
    print("  - print_results(): Print results nicely")
    print("  - compare_results(): Compare multiple results")
    print("  - plot_training_history(): Plot training curves (NEW)")
    print("  - create_comparison_table(): Format comparison table (NEW)")