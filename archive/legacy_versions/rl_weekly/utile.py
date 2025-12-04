"""
Utility Functions for Portfolio Analysis and Optimization
==========================================================
Consolidated utility functions for portfolio analysis, entropy analysis, and optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
from portfolio_system import SystemConfig
from portfolio_training_system import PPOAgent, PortfolioEnvironment


def analyze_portfolio_allocation(agent: PPOAgent, test_env: PortfolioEnvironment, 
                                 config: SystemConfig, plot: bool = True) -> Dict:
    """
    Analyze portfolio allocation strategy.
    
    Args:
        agent: Trained PPOAgent
        test_env: Test environment
        config: SystemConfig
        plot: Whether to create plots
    
    Returns:
        dict with analysis results
    """
    
    print("=" * 70)
    print("PORTFOLIO ALLOCATION ANALYSIS")
    print("=" * 70)
    
    # Run test episode to get weights
    obs = test_env.reset()
    all_weights = []
    portfolio_values = []
    dates_used = []
    
    # Track dates during evaluation
    current_step = test_env.current_step
    while True:
        action, _, _ = agent.select_action(obs, training=False, store_trajectory=False)
        all_weights.append(action.copy())
        
        # Get current date from environment
        if current_step < len(test_env.dates):
            dates_used.append(test_env.dates[current_step])
        else:
            # Fallback: create date from step if dates not available
            dates_used.append(None)
        
        next_obs, reward, done, truncated, info = test_env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        
        obs = next_obs
        current_step += 1
        
        if done or truncated:
            break
    
    # Convert dates to pandas datetime series
    if dates_used and dates_used[0] is not None:
        dates_series = pd.to_datetime(dates_used)
        # Add initial date if needed
        if len(dates_series) == len(portfolio_values) - 1:
            dates_series = pd.to_datetime([dates_series[0] - pd.Timedelta(days=7)] + dates_series.tolist())
    else:
        # Fallback: create date range from test period
        test_start = pd.Timestamp('2020-01-01')  # Default fallback
        dates_series = pd.date_range(start=test_start, periods=len(portfolio_values), freq='W-FRI')
    
    weights_array = np.array(all_weights)
    
    # Calculate statistics
    mean_weights = np.mean(weights_array, axis=0)
    std_weights = np.std(weights_array, axis=0)
    min_weights = np.min(weights_array, axis=0)
    max_weights = np.max(weights_array, axis=0)
    
    # Weight diversity metrics
    weight_entropy = -np.sum(mean_weights * np.log(mean_weights + 1e-10))
    max_entropy = np.log(len(config.tickers))
    weight_concentration = np.max(mean_weights)  # Max weight
    weight_gini = 1 - np.sum(mean_weights ** 2)  # Gini coefficient (1 = perfect diversity)
    
    print(f"\n📊 Allocation Statistics:")
    print(f"  Number of weeks: {len(weights_array)}")
    print(f"  Weight entropy: {weight_entropy:.4f} (max possible: {max_entropy:.4f})")
    print(f"  Entropy ratio: {weight_entropy/max_entropy:.2%}")
    print(f"  Weight concentration: {weight_concentration:.2%} (max single asset)")
    print(f"  Weight diversity (Gini): {weight_gini:.4f} (1.0 = perfect diversity)")
    
    print(f"\n📈 Average Portfolio Weights:")
    print(f"  {'Ticker':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("  " + "-" * 50)
    for ticker, weight, std, min_w, max_w in zip(config.tickers, mean_weights, std_weights, min_weights, max_weights):
        print(f"  {ticker:<8} {weight:>8.2%} {std:>8.2%} {min_w:>8.2%} {max_w:>8.2%}")
    
    # Check for issues
    issues = []
    recommendations = []
    
    if weight_entropy < max_entropy * 0.5:
        issues.append(f"Low weight entropy ({weight_entropy:.3f}) - portfolio too concentrated")
        recommendations.append("Increase entropy_coef to encourage more diverse allocations")
    
    if weight_concentration > 0.5:
        issues.append(f"High concentration ({weight_concentration:.1%}) - over-reliance on single asset")
        recommendations.append("Consider reducing max_position_size or increasing entropy")
    
    if np.std(mean_weights) < 0.05:
        issues.append("Low weight diversity - weights too uniform (possible policy collapse)")
        recommendations.append("Increase exploration_noise and entropy_coef")
    
    if weight_gini < 0.3:
        issues.append(f"Low diversity (Gini={weight_gini:.3f}) - weights too similar")
        recommendations.append("Agent may be converging to uniform allocation")
    
    # Weight stability over time
    weight_stability = np.mean([np.std(weights_array[i:i+5], axis=0).mean() 
                                for i in range(len(weights_array)-4)])
    
    print(f"\n📉 Weight Stability:")
    print(f"  Average weekly weight change: {weight_stability:.4f}")
    
    if weight_stability < 0.01:
        issues.append("Very stable weights - agent may not be adapting to market conditions")
        recommendations.append("Check if agent is learning or just using static allocation")
    
    if issues:
        print(f"\n⚠️  ALLOCATION ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  - {rec}")
    else:
        print(f"\n✅ Allocation looks reasonable")
    
    # Create plots if requested
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Average weights bar chart
        axes[0, 0].bar(config.tickers, mean_weights, alpha=0.7, edgecolor='black')
        axes[0, 0].set_ylabel('Average Weight')
        axes[0, 0].set_title('Average Portfolio Allocation')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].axhline(y=1/len(config.tickers), color='r', linestyle='--', 
                           label='Equal Weight', alpha=0.5)
        axes[0, 0].legend()
        
        # 2. Weight evolution over time (with dates)
        # Use dates_series excluding first value (initial state) if lengths match
        weight_dates = dates_series[1:] if len(dates_series) == len(weights_array) + 1 else dates_series[:len(weights_array)]
        for i, ticker in enumerate(config.tickers):
            axes[0, 1].plot(weight_dates, weights_array[:, i], label=ticker, alpha=0.7)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].set_title('Portfolio Weights Over Time')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Weight distribution (box plot)
        axes[1, 0].boxplot([weights_array[:, i] for i in range(len(config.tickers))], 
                          labels=config.tickers)
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].set_title('Weight Distribution')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Portfolio value over time (with dates)
        axes[1, 1].plot(dates_series, portfolio_values, linewidth=2)
        axes[1, 1].axhline(y=config.initial_capital, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Portfolio Value ($)')
        axes[1, 1].set_title('Portfolio Value Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'mean_weights': mean_weights,
        'std_weights': std_weights,
        'entropy': weight_entropy,
        'max_entropy': max_entropy,
        'entropy_ratio': weight_entropy / max_entropy,
        'concentration': weight_concentration,
        'gini': weight_gini,
        'stability': weight_stability,
        'issues': issues,
        'recommendations': recommendations,
        'weights_over_time': weights_array
    }


def analyze_entropy(agent: PPOAgent) -> Optional[Dict]:
    """Analyze entropy during training."""
    
    print("\n" + "=" * 70)
    print("ENTROPY ANALYSIS")
    print("=" * 70)
    
    if not agent.entropies:
        print("  ⚠️  No entropy history available")
        return None
    
    entropies = np.array(agent.entropies)
    
    print(f"\n📊 Entropy Statistics:")
    print(f"  Total training steps: {len(entropies)}")
    print(f"  Mean entropy: {np.mean(entropies):.6f}")
    print(f"  Std entropy: {np.std(entropies):.6f}")
    print(f"  Min entropy: {np.min(entropies):.6f}")
    print(f"  Max entropy: {np.max(entropies):.6f}")
    
    # Recent entropy
    if len(entropies) >= 10:
        recent_entropy = np.mean(entropies[-10:])
        print(f"  Recent (last 10) entropy: {recent_entropy:.6f}")
    
    # Check for low entropy
    issues = []
    if len(entropies) >= 10:
        if np.mean(entropies[-10:]) < 0.01:
            issues.append("Very low recent entropy (< 0.01) - policy may have collapsed")
        elif np.mean(entropies[-10:]) < 0.05:
            issues.append("Low recent entropy (< 0.05) - policy becoming deterministic")
    
    if np.mean(entropies) < 0.01:
        issues.append("Overall low entropy - agent not exploring enough")
    
    if issues:
        print(f"\n⚠️  ENTROPY ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  - Increase entropy_coef from {agent.config.entropy_coef} to 0.02-0.05")
        print(f"  - Increase exploration_noise from {agent.config.exploration_noise} to 0.4-0.5")
        print(f"  - Consider increasing min_exploration from {agent.config.min_exploration} to 0.05")
    else:
        print(f"\n✅ Entropy levels look reasonable")
    
    return {
        'mean_entropy': np.mean(entropies),
        'recent_entropy': np.mean(entropies[-10:]) if len(entropies) >= 10 else None,
        'issues': issues
    }


def get_optimized_config(base_config: SystemConfig) -> SystemConfig:
    """Generate optimized configuration to increase returns."""
    
    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    
    # Create optimized config
    opt_config = SystemConfig()
    
    # Copy base config
    for key, value in base_config.__dict__.items():
        if not key.startswith('_'):
            setattr(opt_config, key, value)
    
    print("\n🔧 OPTIMIZATIONS FOR INCREASED RETURNS:")
    
    # 1. Increase risk tolerance
    print("\n1. RISK TOLERANCE:")
    print(f"   max_position_size: {base_config.max_position_size} → 0.45")
    opt_config.max_position_size = 0.45  # Allow larger positions
    
    # 2. Increase entropy for exploration
    print(f"   entropy_coef: {base_config.entropy_coef} → 0.03")
    opt_config.entropy_coef = 0.03  # Increase from 0.01
    
    # 3. Increase exploration
    print(f"   exploration_noise: {base_config.exploration_noise} → 0.4")
    opt_config.exploration_noise = 0.4  # Increase from 0.3
    
    print(f"   min_exploration: {base_config.min_exploration} → 0.05")
    opt_config.min_exploration = 0.05  # Increase from 0.01
    
    # 4. Increase reward scaling
    print(f"\n2. REWARD SIGNAL:")
    print(f"   reward_scaling: {base_config.reward_scaling} → 2.0")
    opt_config.reward_scaling = 2.0  # Stronger learning signal
    
    # 5. Reduce drawdown penalty
    print(f"\n3. RISK MANAGEMENT:")
    print(f"   stop_loss_threshold: {base_config.stop_loss_threshold} → 0.30")
    opt_config.stop_loss_threshold = 0.30  # More lenient
    
    # 6. Adjust reward function
    print(f"   penalty_drawdown: {base_config.penalty_drawdown} → True (reduced coefficient)")
    opt_config.penalty_drawdown = True  # Keep but reduce in code
    
    # 7. Learning rate adjustments
    print(f"\n4. LEARNING:")
    print(f"   learning_rate: {base_config.learning_rate} → 5e-4")
    opt_config.learning_rate = 5e-4  # Slightly higher
    
    print(f"   batch_size: {base_config.batch_size} → 128")
    opt_config.batch_size = 128  # Larger batches for stability
    
    print(f"\n✅ Optimized configuration created!")
    print(f"\n⚠️  NOTE: You'll need to update the reward function to reduce")
    print(f"   drawdown penalty coefficient from 0.2 to 0.1 in the code.")
    
    return opt_config


def create_fixed_buy_hold_function() -> str:
    """Create corrected buy-and-hold calculation function."""
    
    code = '''
def calculate_buy_hold_performance(test_data, config):
    """
    Calculate buy-and-hold performance correctly.
    
    Args:
        test_data: DataFrame with test period data
        config: SystemConfig
    
    Returns:
        dict with performance metrics
    """
    import numpy as np
    
    # Get test dates
    test_dates = sorted(test_data['date'].unique())
    n_assets = len(config.tickers)
    
    # Initialize portfolio
    initial_value = config.initial_capital
    portfolio_value = initial_value
    equal_weight = 1.0 / n_assets
    
    # Track values and returns
    portfolio_values = [initial_value]
    returns = []
    
    # Get prices for each date
    for i in range(len(test_dates) - 1):
        current_date = test_dates[i]
        next_date = test_dates[i + 1]
        
        # Get current prices
        current_data = test_data[test_data['date'] == current_date]
        current_prices = {}
        for ticker in config.tickers:
            ticker_data = current_data[current_data['ticker'] == ticker]
            if not ticker_data.empty:
                current_prices[ticker] = ticker_data['close'].values[0]
        
        # Get next prices
        next_data = test_data[test_data['date'] == next_date]
        next_prices = {}
        for ticker in config.tickers:
            ticker_data = next_data[next_data['ticker'] == ticker]
            if not ticker_data.empty:
                next_prices[ticker] = ticker_data['close'].values[0]
        
        # Calculate portfolio return (equal weights, no rebalancing)
        # Portfolio return = sum(weight_i * return_i)
        portfolio_return = 0.0
        for ticker in config.tickers:
            if ticker in current_prices and ticker in next_prices:
                if current_prices[ticker] > 0:
                    asset_return = (next_prices[ticker] / current_prices[ticker]) - 1
                    portfolio_return += equal_weight * asset_return
        
        # Update portfolio value
        portfolio_value *= (1 + portfolio_return)
        portfolio_values.append(portfolio_value)
        returns.append(portfolio_return)
    
    # Calculate metrics
    returns_array = np.array(returns)
    total_return = (portfolio_value / initial_value) - 1
    annual_return = (1 + total_return) ** (52 / len(returns)) - 1
    
    # Sharpe ratio
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    sharpe = (mean_return - config.risk_free_rate/52) / (std_return + 1e-8) * np.sqrt(52)
    
    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (np.array(portfolio_values) - peak) / peak
    max_drawdown = abs(drawdown.min())
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'portfolio_values': portfolio_values,
        'returns': returns_array
    }
'''
    
    return code

