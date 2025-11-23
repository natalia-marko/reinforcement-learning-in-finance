import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
# Add parent dir to path to find 'core' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import RecurrentPPO
from core.rl_system import PortfolioEnv
from core.data_eng_simple import create_features_no_leakage, get_lean_features
from core.config import *

from core.data_eng_expanded import create_expanded_features

def backtest_model(lean_mode=False, expanded_mode=False):
    """
    Backtest on TRUE held-out test set
    CRITICAL: This uses data that was NEVER seen during training/validation
    """
    if expanded_mode and lean_mode:
        mode_str = "EXPANDED-LEAN"
    elif expanded_mode:
        mode_str = "EXPANDED-FULL"
    elif lean_mode:
        mode_str = "LEAN"
    else:
        mode_str = "FULL"
        
    print("="*60)
    print(f"BACKTESTING ON HELD-OUT TEST SET ({mode_str} MODE)")
    print("="*60)
    
    # 1. Load the TRUE test set (never touched during training)
    from core.config import get_data_paths
    data_paths = get_data_paths(expanded_mode=expanded_mode)
    test_set_file = data_paths['test']
    
    if not os.path.exists(test_set_file):
        print(f"❌ Test set not found at {test_set_file}")
        print("Please run core/data_eng_expanded.py (or simple) first.")
        return
    
    print(f"\nLoading test set from {test_set_file}")
    test_data = pd.read_csv(test_set_file, index_col=0, parse_dates=True)
    
    # Separate prices and benchmark
    test_prices = test_data[TICKERS + list(MACRO_SYMBOLS.keys())]
    test_qqq = test_data['QQQ'] if 'QQQ' in test_data.columns else None
    
    print(f"Test period: {test_prices.index[0]} to {test_prices.index[-1]}")
    print(f"Test duration: {len(test_prices)} weeks")
    
    
    # 2. Create features for test set (using same process as training)
    print("\nCreating features for test set...")
    
    if expanded_mode:
        print(f"   Using EXPANDED feature set (~175 features)")
        test_features_full = create_expanded_features(test_prices, None, TICKERS)
    else:
        test_features_full = create_features_no_leakage(test_prices, TICKERS)
    
    if lean_mode:
        print("Using LEAN feature set")
        test_features = get_lean_features(test_features_full)
    else:
        test_features = test_features_full
    
    print(f"Test features shape: {test_features.shape}")
    
    # 3. Find best model
    if expanded_mode and lean_mode:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_expanded_lean')
        plot_filename = 'backtest_results_expanded_lean.png'
        results_filename = 'backtest_results_detailed_expanded_lean.csv'
    elif expanded_mode:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_expanded')
        plot_filename = 'backtest_results_expanded.png'
        results_filename = 'backtest_results_detailed_expanded.csv'
    elif lean_mode:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_lean')
        plot_filename = 'backtest_results_lean.png'
        results_filename = 'backtest_results_detailed_lean.csv'
    else:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        plot_filename = 'backtest_results.png'
        results_filename = 'backtest_results_detailed.csv'
        
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    model_path = None
    
    # First try to find the overall best model
    overall_best = os.path.join(models_dir, 'best_overall_model.zip')
    if os.path.exists(overall_best):
        model_path = overall_best
        print(f"Found overall best model: {model_path}")
    else:
        # Find best model from individual folds
        for i in range(N_FOLDS - 1, -1, -1):
            path = os.path.join(models_dir, f'fold_{i}', 'best_model.zip')
            if os.path.exists(path):
                model_path = path
                print(f"Found model from fold {i+1}: {model_path}")
                break
    
    if model_path is None:
        print(f"❌ No trained model found in {models_dir}")
        return
    
    print(f"\nLoading model from {model_path}")
    model = RecurrentPPO.load(model_path)
    
    # 4. Create test environment
    test_prices_for_env = test_prices[TICKERS] if test_prices is not None else None
    env = PortfolioEnv(test_features, test_prices_for_env, tickers=TICKERS)
    
    # 5. Run backtest
    print("\nRunning backtest...")
    obs, _ = env.reset()
    
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    # Track detailed metrics
    portfolio_values = []
    weights_history = []
    returns_history = []
    dates = []
    
    step_count = 0
    while True:
        # Get action from model
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_starts = terminated
        
        # Record metrics
        portfolio_values.append(info['balance'])
        weights_history.append(info['weights'])
        returns_history.append(info['period_return'])
        
        # Track dates (every rebalance period)
        if step_count * REBALANCE_PERIOD < len(test_features):
            date_idx = min(step_count * REBALANCE_PERIOD, len(test_features) - 1)
            dates.append(test_features.index[date_idx])
        
        step_count += 1
        
        if terminated or truncated:
            break
    
    print(f"Backtest complete. Total steps: {step_count}")
    
    # 6. Calculate performance metrics
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Total return
    total_return = (portfolio_values[-1] / env.initial_balance) - 1
    
    # Annualized return
    n_years = len(test_features) / 52
    annual_return = (1 + total_return) ** (1/n_years) - 1
    
    # Sharpe ratio (annualized)
    if len(returns) > 1:
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(52 / REBALANCE_PERIOD)
    else:
        sharpe = 0
    
    # Max drawdown
    cumulative = portfolio_values / env.initial_balance
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Compare with benchmark
    if test_qqq is not None:
        # Get benchmark values at the specific rebalance dates
        # We use the dates list we populated during the backtest
        # Ensure dates are valid indices in test_qqq
        valid_dates = [d for d in dates if d in test_qqq.index]
        
        if len(valid_dates) > 1:
            qqq_values = test_qqq.loc[valid_dates].values
            qqq_return = (qqq_values[-1] / qqq_values[0]) - 1
            qqq_annual = (1 + qqq_return) ** (1/n_years) - 1
            alpha = annual_return - qqq_annual
        else:
            # Fallback if date alignment fails
            print("⚠️ Warning: Could not align benchmark dates. Using full period.")
            qqq_return = (test_qqq.iloc[-1] / test_qqq.iloc[0]) - 1
            qqq_annual = (1 + qqq_return) ** (1/n_years) - 1
            alpha = annual_return - qqq_annual
    else:
        qqq_return = 0
        qqq_annual = 0
        alpha = annual_return
    
    # 7. Print results
    print("\n" + "="*60)
    print(f"BACKTEST RESULTS ({mode_str})")
    print("="*60)
    print(f"Test Period: {test_features.index[0].date()} to {test_features.index[-1].date()}")
    print(f"Duration:    {n_years:.2f} years")
    print("-"*60)
    print("PORTFOLIO PERFORMANCE:")
    print(f"Total Return (Absolute):   {total_return*100:>8.2f}%")
    print(f"Annualized Return:         {annual_return*100:>8.2f}%")
    print(f"Sharpe Ratio:              {sharpe:>8.2f}")
    print(f"Max Drawdown:              {max_drawdown*100:>8.2f}%")
    print(f"Calmar Ratio:              {calmar:>8.2f}")
    print("-"*60)
    print("BENCHMARK COMPARISON (QQQ):")
    print(f"Total Return (Absolute):   {qqq_return*100:>8.2f}%")
    print(f"Annualized Return:         {qqq_annual*100:>8.2f}%")
    print(f"Alpha (Annualized):        {alpha*100:>8.2f}%")
    print("-"*60)
    print("TRANSACTION ANALYSIS:")
    print(f"Total Trades:     {step_count}")
    print(f"Avg Turnover:     {np.mean([np.sum(np.abs(w - weights_history[0])) for w in weights_history])*100:.2f}%")
    print(f"Transaction Cost: {TOTAL_COST_BPS*10000:.0f} bps per trade")
    print("="*60)
    
    # 8. Generate Performance Plot
    plt.figure(figsize=(12, 8))
    
    # Portfolio vs Benchmark
    plt.subplot(2, 1, 1)
    
    # Calculate percentage returns
    portfolio_pct = (portfolio_values / env.initial_balance - 1) * 100
    color = 'green' if lean_mode else 'blue'
    plt.plot(portfolio_pct, label=f'RL Portfolio (Total: {portfolio_pct[-1]:.1f}%)', color=color, linewidth=2)
    
    if test_qqq is not None:
        # Use the same valid dates logic as in calculation
        valid_dates = [d for d in dates if d in test_qqq.index]
        
        if len(valid_dates) > 1:
            # Normalize benchmark to start at 0% based on the FIRST rebalance date
            qqq_values = test_qqq.loc[valid_dates].values
            benchmark_pct = (qqq_values / qqq_values[0] - 1) * 100
            
            # Plot against the portfolio values
            min_len = min(len(portfolio_values), len(benchmark_pct))
            plt.plot(benchmark_pct[:min_len], label=f'Benchmark QQQ (Total: {benchmark_pct[min_len-1]:.1f}%)', color='gray', linestyle='--')
        else:
             # Fallback
             benchmark_norm = test_qqq.values / test_qqq.values[0]
             benchmark_pct = (benchmark_norm - 1) * 100
             plt.plot(np.linspace(0, len(portfolio_values), len(benchmark_pct)), benchmark_pct, label='Benchmark (QQQ)', color='gray', linestyle='--', alpha=0.7)

    plt.title(f'Out-of-Sample Performance ({mode_str})', fontsize=14)
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Drawdown
    plt.subplot(2, 1, 2)
    plt.plot(drawdown * 100, label='Drawdown', color='red', linewidth=1)
    plt.fill_between(range(len(drawdown)), drawdown * 100, 0, color='red', alpha=0.2)
    plt.title('Portfolio Drawdown', fontsize=12)
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Rebalance Steps (Weekly)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUTS_DIR, plot_filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()  # Close the figure to prevent display and free memory
    print(f"✅ Performance plot saved to {plot_path}")

    # 9. Save detailed results
    results_df = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'period_return': returns_history,
        'total_return': portfolio_values / env.initial_balance - 1
    })
    
    # Add weights as separate columns
    for i, ticker in enumerate(TICKERS):
        results_df[f'weight_{ticker}'] = [w[i] for w in weights_history]
    
    results_path = os.path.join(OUTPUTS_DIR, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"✅ Detailed results saved to {results_path}")
    
    # 10. Risk warning
    if total_return > 0.3 or sharpe > 1.5:
        print("\n⚠️  WARNING: Results seem too good to be true!")
        print("Please verify:")
        print("1. No data leakage in feature engineering")
        print("2. Transaction costs are realistic (currently {:.0f} bps)".format(TOTAL_COST_BPS*10000))
        print("3. Test set was truly held out during training")
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'alpha': alpha
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest LSTM Agent")
    parser.add_argument("--lean", action="store_true", help="Use lean feature set (optimized)")
    parser.add_argument("--expanded", action="store_true", help="Use expanded feature set (175 features)")
    args = parser.parse_args()
    
    backtest_model(lean_mode=args.lean, expanded_mode=args.expanded)