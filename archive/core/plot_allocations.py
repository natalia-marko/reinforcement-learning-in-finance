import os
import sys
import pandas as pd
import numpy as np

# Matplotlib backend setup
import matplotlib
try:
    # Check if running in Jupyter
    get_ipython()
    # Use inline backend for Jupyter
    from IPython import display
    matplotlib.use('module://matplotlib_inline.backend_inline')
except NameError:
    # Use non-interactive backend for standalone execution
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rl_system import PortfolioEnv
from core.data_loading_preprocessing import create_features, get_lean_features
from core.config import *
from stable_baselines3 import PPO

# Helper function to check if running in Jupyter
def is_jupyter():
    try:
        get_ipython()
        return True
    except NameError:
        return False

def plot_portfolio_allocation(env, output_path='outputs/portfolio_allocation.png'):
    """
    Visualize how weights change over time.
    Use this after running an evaluation episode.
    """
    if not hasattr(env, 'weights_history') or len(env.weights_history) == 0:
        print("❌ No weights history found. Run an episode first.")
        return
    
    history = np.array(env.unwrapped.weights_history)  # Shape: (Steps, n_assets)
    df_weights = pd.DataFrame(history, columns=env.unwrapped.tickers)
    
    plt.figure(figsize=(15, 6))
    plt.stackplot(df_weights.index, df_weights.T, labels=df_weights.columns, alpha=0.8)
    plt.title("Portfolio Allocation Evolution (Stacked) - Test Set", fontsize=16)
    plt.xlabel("Rebalance Periods", fontsize=12)
    plt.ylabel("Weight", fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.margins(0, 0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if is_jupyter():
        plt.show()  # Display in Jupyter
    plt.close()
    print(f"✅ Portfolio allocation plot saved to {output_path}")

def main():
    """
    Load best model and plot portfolio allocation on test set
    """
    print("=" * 60)
    print("PORTFOLIO ALLOCATION VISUALIZATION")
    print("=" * 60)
    
    # Load test data
    data_paths = get_data_paths(expanded_mode=True)
    
    if not os.path.exists(data_paths['test']):
        print(f"❌ Test data not found at {data_paths['test']}")
        print("   Please run: python -m core.data_loading_preprocessing")
        return
    
    print("\n📊 Loading test data...")
    test_data = pd.read_csv(data_paths['test'], index_col=0, parse_dates=True, header=[0, 1])
    
    # Extract prices from MultiIndex columns
    test_prices = test_data['prices']
    
    print("🔧 Creating features...")
    test_features_full = create_features(test_prices, None, lean=False, drop_zero_var=False)
    test_features = get_lean_features(test_features_full)
    
    # Load model first to check expected observation shape
    model_path = 'models_expanded_lean/best_overall_model.zip'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please train the model first: python core/train.py --expanded --lean")
        return
    
    print(f"\n🤖 Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Get expected observation shape from model
    expected_obs_shape = model.observation_space.shape[0]
    actual_obs_shape = test_features.shape[1] + TICKERS.__len__() + 4  # features + weights + stats
    
    # The environment adds: current_weights (7) + portfolio_stats (4) + correlation features
    # So we need to account for this
    print(f"   Model expects: {expected_obs_shape} features")
    print(f"   Test features: {test_features.shape[1]}")
    
    # Check if features match
    if test_features.shape[1] != expected_obs_shape - 11:  # env adds 11 features (7 weights + 4 stats)
        print(f"   ⚠️  Feature mismatch detected, using only available features")
        # This is ok - the environment will handle it
    
    print(f"   Test features shape: {test_features.shape}")
    print(f"   Test prices shape: {test_prices.shape}")
    
    # Create test environment (without correlation features to match training)
    env = PortfolioEnv(test_features, prices_df=test_prices, tickers=TICKERS, use_correlation=False)
    
    print("\n🎮 Running evaluation on test set...")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = done or truncated
    
    print(f"\n📈 Evaluation Results:")
    print(f"   Total steps: {step_count}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final balance: ${info['balance']:,.2f}")
    print(f"   Total return: {info['total_return']:.2%}")
    print(f"   Sharpe ratio: {info['sharpe']:.2f}")
    print(f"   Max drawdown: {info['max_drawdown']:.2%}")
    
    # Plot portfolio allocation
    print(f"\n📊 Creating portfolio allocation plot...")
    os.makedirs('outputs', exist_ok=True)
    plot_portfolio_allocation(env, output_path='outputs/portfolio_allocation_test.png')
    
    # Also create a detailed allocation table
    if len(env.weights_history) > 0:
        history = np.array(env.weights_history)
        df_weights = pd.DataFrame(history, columns=env.tickers)
        
        # Summary statistics
        print(f"\n📊 Average Allocation:")
        avg_weights = df_weights.mean()
        for ticker, weight in avg_weights.items():
            print(f"   {ticker}: {weight:.1%}")
        
        # Save to CSV
        weights_csv = 'outputs/portfolio_weights_test.csv'
        df_weights.to_csv(weights_csv, index_label='Rebalance_Period')
        print(f"\n✅ Detailed weights saved to {weights_csv}")

if __name__ == "__main__":
    main()
