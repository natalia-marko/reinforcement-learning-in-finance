import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rl_system import PortfolioEnv
from core.data_loading_preprocessing import create_features
from core.config import *

# Helper function to check if running in Jupyter
def is_jupyter():
    try:
        get_ipython()
        return True
    except NameError:
        return False

def backtest_model():
    """
    Backtest RL agent on TRUE held-out test set
    """
    print("="*60)
    print("BACKTESTING MODEL")
    print("="*60)
    
    # Set directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    plot_filename = 'backtest_results.png'
    
    # 1. Load Test Set
    from core.config import get_data_paths
    data_paths = get_data_paths()
    test_set_file = data_paths['test']
    
    if not os.path.exists(test_set_file):
        print(f"❌ Test set not found at {test_set_file}")
        print("Please run core/data_eng_expanded.py (or simple) first.")
        return
    
    print(f"\nLoading test set from {test_set_file}")
    test_data = pd.read_csv(test_set_file, index_col=0, parse_dates=True, header=[0, 1])

    # Extract prices from multi-level columns
    test_prices = test_data['prices']

    # Create equal-weight benchmark - CORRECT METHOD
    # Normalize each stock to start at 1.0, then average
    stock_tickers = [t for t in TICKERS if t in test_prices.columns]
    normalized_prices = test_prices[stock_tickers].div(test_prices[stock_tickers].iloc[0])
    ew_benchmark = normalized_prices.mean(axis=1)  # Already normalized, no need to divide again

    print(f"Test period: {test_prices.index[0].date()} to {test_prices.index[-1].date()}")
    print(f"Test duration: {len(test_prices)} weeks")
    
    # 2. Create features
    print("\nCreating features for test set...")
    test_features = create_features(test_prices, None, verbose=False)
    print(f"Test features shape: {test_features.shape}")
    
    # 3. Find best model
    model_path = None
    overall_best = os.path.join(models_dir, 'best_overall_model.zip')
    
    if os.path.exists(overall_best):
        model_path = overall_best
        print(f"Found overall best model: {model_path}")
    else:
        for i in range(N_FOLDS - 1, -1, -1):
            path = os.path.join(models_dir, f'fold_{i}', 'best_model.zip')
            if os.path.exists(path):
                model_path = path
                print(f"Found model from fold {i+1}: {model_path}")
                break
    
    if model_path is None:
        print(f"❌ No trained model found in {models_dir}. Please run src/train.py first.")
        return

    print(f"\nLoading model from {model_path}")
    model = PPO.load(model_path)
    
    # Validate model dimensions match test data
    expected_obs_dim = model.observation_space.shape[0]
    actual_obs_dim = test_features.shape[1]
    
    if expected_obs_dim != actual_obs_dim:
        print(f"\n❌ DIMENSION MISMATCH ERROR:")
        print(f"   Model expects {expected_obs_dim} features")
        print(f"   Test data has {actual_obs_dim} features")
        print(f"\n   This means the model was trained with different features than the current test data.")
        print(f"   You need to either:")
        print(f"   1. Retrain the model with: python core/train.py")
        print(f"   2. Or use a model that matches the current {actual_obs_dim}-feature setup")
        return

    # Load VecNormalize stats (CRITICAL - model was trained with normalized observations!)
    stats_path = model_path.replace('.zip', '_vecnormalize.pkl')
    if not os.path.exists(stats_path):
        print(f"❌ VecNormalize stats not found at {stats_path}")
        print("   This model was trained with VecNormalize - normalization stats are required!")
        print("   Without them, predictions will be incorrect.")
        return

    print(f"Loading normalization stats from {stats_path}")

    # 4. Create test environment with VecNormalize
    # CRITICAL FIX: Set use_correlation=False to match training/evaluation code
    test_prices_for_env = test_prices[TICKERS] if test_prices is not None else None

    # Wrap in DummyVecEnv (required by VecNormalize)
    env = DummyVecEnv([
        lambda: PortfolioEnv(test_features, test_prices_for_env, tickers=TICKERS, use_correlation=False)
    ])

    # Load VecNormalize stats - handle dict format
    import pickle
    try:
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        
        # Check if it's a dict (old format) or VecNormalize object
        if isinstance(stats, dict):
            # Create VecNormalize wrapper manually and set stats
            env = VecNormalize(env, training=False, norm_obs=True, norm_reward=False)
            # Load stats from dict
            if 'obs_rms' in stats:
                env.obs_rms = stats['obs_rms']
            if 'ret_rms' in stats:
                env.ret_rms = stats['ret_rms']
            print("✅ Loaded normalization stats from dict format")
        else:
            # It's a VecNormalize object, use standard loading
            env = VecNormalize.load(stats_path, env)
            print("✅ Loaded VecNormalize object")
    except Exception as e:
        print(f"⚠️  Could not load VecNormalize stats: {e}")
        print("   Creating environment without normalization...")
        # Don't wrap in VecNormalize if we can't load stats
        pass

    # Set to inference mode (don't update normalization stats, don't normalize rewards)
    if isinstance(env, VecNormalize):
        env.training = False
        env.norm_reward = False

    print("✅ Environment created with VecNormalize (inference mode)")
    
    # 5. Run backtest
    print("\nRunning backtest...")
    obs = env.reset()  # VecEnv.reset() returns obs only, not tuple

    portfolio_values = []
    weights_history = []
    returns_history = []
    dates = []

    step_count = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # VecEnv returns list of info dicts (one per environment)
        # We have only 1 environment, so take first element
        info_dict = info[0]
        terminated = done[0]

        portfolio_values.append(info_dict['balance'])
        weights_history.append(info_dict['weights'])
        returns_history.append(info_dict['period_return'])
        
        if step_count * REBALANCE_PERIOD < len(test_features):
            date_idx = min(step_count * REBALANCE_PERIOD, len(test_features) - 1)
            dates.append(test_features.index[date_idx])

        step_count += 1
        if terminated:  # done[0] already extracted as 'terminated'
            break
    
    print(f"Backtest complete. Total steps: {step_count}")

    # 6. Calculate Metrics
    portfolio_values = np.array(portfolio_values)
    # Get initial_balance from the unwrapped environment (VecEnv method)
    initial_balance = env.get_attr('initial_balance')[0]
    total_return = (portfolio_values[-1] / initial_balance) - 1
    n_years = len(test_features) / 52
    annual_return = (1 + total_return) ** (1/n_years) - 1
    
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(returns) > 1:
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(52 / REBALANCE_PERIOD)
    else:
        sharpe = 0
        
    cumulative = portfolio_values / initial_balance
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Benchmark metrics - Equal-Weight
    if ew_benchmark is not None:
        ew_return = float(ew_benchmark.iloc[-1].item() - 1)
        ew_annual = (1 + ew_return) ** (1/n_years) - 1
        alpha_ew = annual_return - ew_annual
    else:
        ew_return = 0; ew_annual = 0; alpha_ew = annual_return

    # 7. Print Results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
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
    print("EQUAL-WEIGHT PORTFOLIO BENCHMARK:")
    print(f"Total Return (Absolute):   {ew_return*100:>8.2f}%")
    print(f"Annualized Return:         {ew_annual*100:>8.2f}%")
    print(f"Alpha vs EW:               {alpha_ew*100:>8.2f}%")
    print("="*60)
    
    # 8. Plot
    plt.figure(figsize=(12, 8))
    
    # Portfolio vs Benchmark
    plt.subplot(2, 1, 1)
    portfolio_pct = (portfolio_values / initial_balance - 1) * 100
    plt.plot(portfolio_pct, label=f'RL Portfolio ({portfolio_pct[-1]:.1f}%)', color='blue', linewidth=2.5)
    
    # Equal-weight benchmark
    if ew_benchmark is not None:
        ew_pct = (ew_benchmark.values - 1) * 100
        min_len = min(len(portfolio_pct), len(ew_pct))
        plt.plot(ew_pct[:min_len], label=f'Equal-Weight ({ew_pct[min_len-1].item():.1f}%)', 
                color='orange', linestyle='--', linewidth=2)
    
    plt.title('Out-of-Sample Performance', fontsize=14)
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
    if is_jupyter():
        plt.show()  # Display in Jupyter
    plt.close()  # Close the figure to prevent display and free memory
    print(f"✅ Performance plot saved to {plot_path}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'period_return': returns_history,
        'total_return': portfolio_values / initial_balance - 1
    })
    for i, ticker in enumerate(TICKERS):
        results_df[f'weight_{ticker}'] = [w[i] for w in weights_history]
        
    results_filename = 'backtest_detailed.csv'
    results_path = os.path.join(OUTPUTS_DIR, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"✅ Detailed results saved to {results_path}")

if __name__ == "__main__":
    backtest_model()
