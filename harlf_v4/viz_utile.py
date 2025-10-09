"""
Visualization Utilities for Hierarchical RL Agents

Clean, modular plotting functions for analyzing agent performance.
All functions follow KISS principle: simple, clear, well-documented.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def plot_training_curves(super_callback, meta_callback, save_path='plots/training_curves.png'):
    """
    Plot validation Sharpe ratios during training for super and meta agents.
    
    Args:
        super_callback: EarlyStoppingCallback from super agent training
        meta_callback: EarlyStoppingCallback from meta agent training
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot super agent
    if hasattr(super_callback, 'val_sharpes') and len(super_callback.val_sharpes) > 0:
        ax.plot(super_callback.eval_steps, super_callback.val_sharpes, 
                marker='o', label='Super Agent', linewidth=2.5, markersize=7, color='#2E86AB')
        best_super = max(super_callback.val_sharpes)
        ax.axhline(y=best_super, color='#2E86AB', linestyle='--', alpha=0.5, 
                   label=f'Best Super: {best_super:.3f}')
    
    # Plot meta agent
    if hasattr(meta_callback, 'val_sharpes') and len(meta_callback.val_sharpes) > 0:
        ax.plot(meta_callback.eval_steps, meta_callback.val_sharpes, 
                marker='s', label='Meta Agent', linewidth=2.5, markersize=7, color='#A23B72')
        best_meta = max(meta_callback.val_sharpes)
        ax.axhline(y=best_meta, color='#A23B72', linestyle='--', alpha=0.5, 
                   label=f'Best Meta: {best_meta:.3f}')
    
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Training Progress: Validation Sharpe Ratio', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(super_metrics: Dict[str, float], 
                            meta_metrics: Dict[str, float],
                            save_path='plots/metrics_comparison.png'):
    """
    Bar chart comparing key metrics between super and meta agents.
    
    Args:
        super_metrics: Dictionary of metrics from super agent
        meta_metrics: Dictionary of metrics from meta agent
        save_path: Path to save the plot
    """
    metrics_to_plot = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
    metric_labels = ['Sharpe Ratio', 'Total Return', 'Max Drawdown', 'Win Rate']
    
    super_values = [super_metrics[m] for m in metrics_to_plot]
    meta_values = [meta_metrics[m] for m in metrics_to_plot]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, super_values, width, label='Super Agent', 
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, meta_values, width, label='Meta Agent', 
                   color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Test Set Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_portfolio_evolution(super_model, meta_model, 
                             super_test_env, meta_test_env,
                             save_path='plots/portfolio_evolution.png'):
    """
    Plot portfolio value evolution over test period for both agents.
    
    Args:
        super_model: Trained super agent model
        meta_model: Trained meta agent model
        super_test_env: Super agent test environment
        meta_test_env: Meta agent test environment
        save_path: Path to save the plot
    """
    # Run super agent through test set
    obs, _ = super_test_env.reset()
    done = False
    while not done:
        action, _ = super_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = super_test_env.step(action)
    
    # Run meta agent through test set
    obs, _ = meta_test_env.reset()
    done = False
    while not done:
        action, _ = meta_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = meta_test_env.step(action)
    
    # Get portfolio histories
    super_portfolio = pd.Series(super_test_env.portfolio_history)
    meta_portfolio = pd.Series(meta_test_env.portfolio_history)
    
    # Normalize to percentage returns
    super_returns = (super_portfolio / super_portfolio.iloc[0] - 1) * 100
    meta_returns = (meta_portfolio / meta_portfolio.iloc[0] - 1) * 100
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(super_returns.values, label='Super Agent', linewidth=2.5, color='#2E86AB')
    ax.plot(meta_returns.values, label='Meta Agent', linewidth=2.5, color='#A23B72')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.fill_between(range(len(super_returns)), super_returns, alpha=0.2, color='#2E86AB')
    ax.fill_between(range(len(meta_returns)), meta_returns, alpha=0.2, color='#A23B72')
    
    ax.set_xlabel('Time Steps (Months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Portfolio Performance on Test Set', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_drawdown_analysis(super_model, meta_model,
                           super_test_env, meta_test_env,
                           save_path='plots/drawdown_analysis.png'):
    """
    Plot drawdown evolution for both agents on test set.
    
    Args:
        super_model: Trained super agent model
        meta_model: Trained meta agent model
        super_test_env: Super agent test environment
        meta_test_env: Meta agent test environment
        save_path: Path to save the plot
    """
    # Run agents through test set
    obs, _ = super_test_env.reset()
    done = False
    while not done:
        action, _ = super_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = super_test_env.step(action)
    
    obs, _ = meta_test_env.reset()
    done = False
    while not done:
        action, _ = meta_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = meta_test_env.step(action)
    
    # Calculate drawdowns
    super_portfolio = pd.Series(super_test_env.portfolio_history)
    meta_portfolio = pd.Series(meta_test_env.portfolio_history)
    
    super_peak = super_portfolio.expanding().max()
    super_dd = (super_portfolio - super_peak) / super_peak * 100
    
    meta_peak = meta_portfolio.expanding().max()
    meta_dd = (meta_portfolio - meta_peak) / meta_peak * 100
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.fill_between(range(len(super_dd)), super_dd, alpha=0.6, color='#2E86AB', 
                     label=f'Super Agent (Max: {super_dd.min():.2f}%)')
    ax.fill_between(range(len(meta_dd)), meta_dd, alpha=0.6, color='#A23B72', 
                     label=f'Meta Agent (Max: {meta_dd.min():.2f}%)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax.set_xlabel('Time Steps (Months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_title('Drawdown Evolution on Test Set', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_agent_allocations(sentiment_agent, technical_agent,
                           super_test_env, meta_test_env,
                           asset_names: List[str],
                           save_path='plots/agent_allocations.png'):
    """
    Plot final portfolio allocations for all agents in the hierarchy.
    
    Args:
        sentiment_agent: Sentiment agent wrapper
        technical_agent: Technical agent wrapper
        super_test_env: Super agent test environment
        meta_test_env: Meta agent test environment
        asset_names: List of asset names
        save_path: Path to save the plot
    """
    n_assets = len(asset_names)
    equal_weight = 1 / n_assets
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Agent Allocation Strategy Analysis', fontsize=16, fontweight='bold')
    
    agents = [
        (sentiment_agent.weights, 'Sentiment Agent', '#F18F01', axes[0, 0]),
        (technical_agent.weights, 'Technical Agent', '#006494', axes[0, 1]),
        (super_test_env.weights, 'Super Agent', '#2E86AB', axes[1, 0]),
        (meta_test_env.weights, 'Meta Agent', '#A23B72', axes[1, 1])
    ]
    
    for weights, title, color, ax in agents:
        ax.bar(range(n_assets), weights, color=color, alpha=0.8, edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Assets', fontsize=11, fontweight='bold')
        ax.set_ylabel('Weight', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}: Final Allocation', fontsize=12, fontweight='bold')
        ax.set_xticks(range(n_assets))
        ax.set_xticklabels(asset_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=equal_weight, color='red', linestyle='--', alpha=0.5, 
                   linewidth=1.5, label=f'Equal Weight ({equal_weight:.3f})')
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(weights) * 1.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_risk_return_scatter(super_metrics: Dict[str, float],
                             meta_metrics: Dict[str, float],
                             save_path='plots/risk_return_scatter.png'):
    """
    Scatter plot showing risk-return profile for both agents.
    
    Args:
        super_metrics: Dictionary of metrics from super agent
        meta_metrics: Dictionary of metrics from meta agent
        save_path: Path to save the plot
    """
    agents = ['Super Agent', 'Meta Agent']
    returns = [super_metrics['total_return'] * 100, meta_metrics['total_return'] * 100]
    volatilities = [super_metrics['volatility'] * 100, meta_metrics['volatility'] * 100]
    sharpe_values = [super_metrics['sharpe_ratio'], meta_metrics['sharpe_ratio']]
    colors = ['#2E86AB', '#A23B72']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, agent in enumerate(agents):
        ax.scatter(volatilities[i], returns[i], s=500, alpha=0.7, color=colors[i], 
                  edgecolor='black', linewidth=2.5, label=agent, zorder=3)
        ax.annotate(f"{agent}\nSharpe: {sharpe_values[i]:.3f}", 
                   xy=(volatilities[i], returns[i]), 
                   xytext=(20, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.7', facecolor=colors[i], alpha=0.3),
                   fontsize=11, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Add efficient frontier line
    ax.plot(volatilities, returns, 'k--', alpha=0.3, linewidth=2, zorder=1, 
            label='Hierarchy Path')
    
    ax.set_xlabel('Annualized Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Return Profile: Hierarchical Agents', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def create_performance_table(super_metrics: Dict[str, float],
                            meta_metrics: Dict[str, float],
                            test_periods: int) -> pd.DataFrame:
    """
    Create comprehensive performance comparison table.
    
    Args:
        super_metrics: Dictionary of metrics from super agent
        meta_metrics: Dictionary of metrics from meta agent
        test_periods: Number of test periods (months)
        
    Returns:
        DataFrame with formatted comparison
    """
    comparison_data = {
        'Metric': [
            'Sharpe Ratio', 
            'Total Return (%)', 
            'Annualized Return (%)', 
            'Max Drawdown (%)', 
            'Volatility (%)', 
            'Win Rate (%)', 
            'Final Value ($)'
        ],
        'Super Agent': [
            f"{super_metrics['sharpe_ratio']:.3f}",
            f"{super_metrics['total_return']*100:.2f}",
            f"{(np.power(1 + super_metrics['total_return'], 12/test_periods) - 1)*100:.2f}",
            f"{super_metrics['max_drawdown']*100:.2f}",
            f"{super_metrics['volatility']*100:.2f}",
            f"{super_metrics['win_rate']*100:.2f}",
            f"{super_metrics['final_value']:,.2f}"
        ],
        'Meta Agent': [
            f"{meta_metrics['sharpe_ratio']:.3f}",
            f"{meta_metrics['total_return']*100:.2f}",
            f"{(np.power(1 + meta_metrics['total_return'], 12/test_periods) - 1)*100:.2f}",
            f"{meta_metrics['max_drawdown']*100:.2f}",
            f"{meta_metrics['volatility']*100:.2f}",
            f"{meta_metrics['win_rate']*100:.2f}",
            f"{meta_metrics['final_value']:,.2f}"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Calculate improvement
    improvement = []
    for i, metric in enumerate(df['Metric']):
        super_val = float(df.iloc[i, 1].replace(',', '').replace('%', '').replace('$', ''))
        meta_val = float(df.iloc[i, 2].replace(',', '').replace('%', '').replace('$', ''))
        
        if 'Drawdown' in metric:  # Lower is better
            imp = ((super_val - meta_val) / super_val * 100) if super_val != 0 else 0
        else:  # Higher is better
            imp = ((meta_val - super_val) / super_val * 100) if super_val != 0 else 0
        
        improvement.append(f"{imp:+.1f}%")
    
    df['Improvement'] = improvement
    
    return df


def plot_meta_allocation_evolution(meta_model, meta_test_env, asset_names,
                                    save_path='plots/meta_allocation_evolution.png'):
    """
    Plot how meta agent's portfolio allocations evolve over time during test period.
    
    Args:
        meta_model: Trained meta agent model
        meta_test_env: Meta agent test environment
        asset_names: List of asset names
        save_path: Path to save the plot
    """
    obs, _ = meta_test_env.reset()
    done = False
    
    allocation_history = []
    
    while not done:
        action, _ = meta_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = meta_test_env.step(action)
        allocation_history.append(info['weights'].copy())
    
    allocation_df = pd.DataFrame(allocation_history, columns=asset_names)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('Meta Agent Portfolio Allocation Evolution Over Time', 
                 fontsize=16, fontweight='bold')
    
    # Stacked area chart
    ax1.stackplot(range(len(allocation_df)), 
                  *[allocation_df[col].values for col in asset_names],
                  labels=asset_names, alpha=0.8)
    ax1.set_xlabel('Time Steps (Months)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Weight', fontsize=12, fontweight='bold')
    ax1.set_title('Stacked Portfolio Weights', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    # Line chart
    for col in asset_names:
        ax2.plot(allocation_df[col].values, label=col, linewidth=2.5, alpha=0.8)
    
    ax2.set_xlabel('Time Steps (Months)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Portfolio Weight', fontsize=12, fontweight='bold')
    ax2.set_title('Individual Asset Weight Evolution', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")
    
    return allocation_df


def plot_all_visualizations(super_model, meta_model,
                            super_test_env, meta_test_env,
                            super_callback, meta_callback,
                            super_metrics, meta_metrics,
                            sentiment_agent, technical_agent,
                            asset_names, test_periods,
                            save_dir='plots'):
    """
    Generate all visualizations in one call.
    
    Args:
        super_model: Trained super agent model
        meta_model: Trained meta agent model
        super_test_env: Super agent test environment
        meta_test_env: Meta agent test environment
        super_callback: Super agent training callback
        meta_callback: Meta agent training callback
        super_metrics: Super agent metrics dictionary
        meta_metrics: Meta agent metrics dictionary
        sentiment_agent: Sentiment agent wrapper
        technical_agent: Technical agent wrapper
        asset_names: List of asset names
        test_periods: Number of test periods
        save_dir: Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    # 1. Training curves
    print("1/6 Plotting training curves...")
    plot_training_curves(super_callback, meta_callback, 
                         f"{save_dir}/training_curves.png")
    
    # 2. Metrics comparison
    print("2/6 Plotting metrics comparison...")
    plot_metrics_comparison(super_metrics, meta_metrics, 
                           f"{save_dir}/metrics_comparison.png")
    
    # 3. Portfolio evolution
    print("3/6 Plotting portfolio evolution...")
    plot_portfolio_evolution(super_model, meta_model, 
                            super_test_env, meta_test_env,
                            f"{save_dir}/portfolio_evolution.png")
    
    # 4. Drawdown analysis
    print("4/6 Plotting drawdown analysis...")
    plot_drawdown_analysis(super_model, meta_model,
                          super_test_env, meta_test_env,
                          f"{save_dir}/drawdown_analysis.png")
    
    # 5. Agent allocations
    print("5/6 Plotting agent allocations...")
    plot_agent_allocations(sentiment_agent, technical_agent,
                          super_test_env, meta_test_env,
                          asset_names,
                          f"{save_dir}/agent_allocations.png")
    
    # 6. Risk-return scatter
    print("6/6 Plotting risk-return scatter...")
    plot_risk_return_scatter(super_metrics, meta_metrics,
                            f"{save_dir}/risk_return_scatter.png")
    
    # 7. Print performance table
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*70)
    comparison_df = create_performance_table(super_metrics, meta_metrics, test_periods)
    print(comparison_df.to_string(index=False))
    print("="*70)
    print("\nAll visualizations completed!")

