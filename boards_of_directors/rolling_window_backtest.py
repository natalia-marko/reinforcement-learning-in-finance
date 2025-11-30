import yfinance as yf
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import torch
import random
import os

# Import our custom modules
import config
from rl_system_setting import PortfolioRebalanceEnv, FinancialFeatureEngineer, AlphaHunterStrategy, load_real_data

# --- REPRODUCIBILITY ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs("./logs/", exist_ok=True)

# === SOTA UPGRADE: ROLLING WALK-FORWARD VALIDATION ===
def rolling_window_train_predict(full_prices, full_features, full_benchmark, 
                                 train_window_days=750,  # ~3 years training
                                 test_window_days=60,    # Retrain every 3 months
                                 model_params=None,
                                 ticker_names=None):
    
    if model_params is None:
        model_params = {'learning_rate': 3e-4, 'ent_coef': 0.01, 'n_steps': 2048}

    total_len = len(full_prices)
    current_start = 0
    
    history = []
    
    # Initialize the first model
    print(f"🚀 Starting Rolling Training (Train: {train_window_days}d, Slide: {test_window_days}d)")
    
    # We need to keep track of the model to fine-tune it
    model = None
    current_balance = 100_000.0
    
    while current_start + train_window_days + test_window_days < total_len:
        # 1. Slice Data
        train_end = current_start + train_window_days
        test_end = train_end + test_window_days
        
        # Train Sets
        p_train = full_prices[current_start:train_end]
        f_train = full_features[current_start:train_end]
        b_train = full_benchmark[current_start:train_end]
        
        # Test Sets (The "Live" period)
        # We need to include lookback window for the test set so the env has history
        test_start_idx = train_end - config.WINDOW_SIZE
        p_test = full_prices[test_start_idx : test_end]
        f_test = full_features[test_start_idx : test_end]
        b_test = full_benchmark[test_start_idx : test_end]
        
        # 2. Setup Environment
        # Note: We do NOT use VecNormalize here to ensure we test on raw, real data dynamics
        # The Feature Engineer already handles Z-scoring, which is safer than VecNormalize for rolling windows
        # Pass regime params from CONFIG
        env = DummyVecEnv([lambda: PortfolioRebalanceEnv(p_train, f_train, b_train, 
                                                         panic_vol=config.PANIC_VOL_RANGE[0], # Using lower bound as default or need specific value
                                                         strong_trend=0.005)]) # Hardcoded or from config if available
        
        # 3. Train Model (Fast Retraining)
        # Fine-tuning is SOTA for financial time series.
        if model is not None:
            model.set_env(env)
            # Fine-tune for fewer steps to adapt to recent regime
            model.learn(total_timesteps=10_000) 
        else:
            model = PPO("MlpPolicy", env, verbose=0, **model_params)
            model.learn(total_timesteps=30_000)
            
        # 4. Predict on Test Window (Simulate Live Trading)
        # We create a temporary env just for this test window
        test_env = PortfolioRebalanceEnv(p_test, f_test, b_test,
                                         initial_balance=current_balance,
                                         panic_vol=config.PANIC_VOL_RANGE[0], 
                                         strong_trend=0.005)
        
        # Run Strategy Wrapper logic manually for this window
        # Note: We use the strategy wrapper to handle the Regime Logic override
        # But the Env ALSO has regime logic now. 
        # The Env's regime logic is for TRAINING (Observation).
        # The Strategy's regime logic is for EXECUTION (Action Override).
        strat = AlphaHunterStrategy(
            model, ticker_names, 
            config.PANIC_VOL_RANGE[0], 0.005, 
            config.REBALANCE_THRESHOLD
        )
        
        obs, _ = test_env.reset()
        done = False
        step_idx = 0
        
        # We need to track the regime from the env now, not just calculate it externally
        # But for the wrapper, we still pass market_vol and trend_strength
        
        while not done and step_idx < len(p_test) - config.WINDOW_SIZE:
            # Recalculate signals locally for the wrapper logic
            # We need a slice of benchmark for volatility calculation
            # Be careful with indices. b_test is aligned with p_test.
            # step_idx corresponds to the current step in the test environment
            # The test env starts at lookback_window.
            
            # Actually, test_env.current_step is managed internally.
            # We just need to get the external signals for the Strategy Wrapper.
            
            # Calculate Market Volatility for Strategy Wrapper
            # We need the last 20 days relative to the current step
            # The current step in test_env is test_env.current_step
            curr_step_env = test_env.current_step
            bench_slice = b_test[curr_step_env-20 : curr_step_env]
            # Calculate returns from prices
            bench_ret = pd.Series(bench_slice).pct_change().dropna()
            market_vol = np.std(bench_ret) if len(bench_ret) > 1 else 0.01
            
            # Calculate Trend Strength for Strategy Wrapper
            # We can extract it from the observation
            # obs shape is (Lookback, Assets, Features) - Unbatched
            # Feature 3 is Relative Strength (Trend)
            # We take the mean across assets for the latest time step in the window.
            # The original code used: np.mean(raw_feats[:, 3]) where raw_feats was features[curr_step-1]
            # Here obs contains the lookback window.
            # Let's take the mean of the trend feature across all assets for the latest time step in the window.
            trend_strength = np.mean(obs[-1, :, 3])
            
            # Predict with Strategy Wrapper
            weights, regime = strat.predict(obs, market_vol, trend_strength)
            
            # Convert weights to action for the environment
            # The environment expects an action that produces these weights.
            # But we can't easily reverse softmax.
            # However, for the purpose of the backtest simulation, we just need to update the portfolio.
            # The test_env.step() method takes an action, applies softmax, and updates.
            # Since we already calculated the desired weights using the Strategy Wrapper (which includes Regime Override),
            # we should ideally bypass the env.step() action-to-weight conversion or reverse engineer the action.
            
            # Simpler approach:
            # We can manually update the test_env portfolio values using the calculated weights
            # instead of calling step(action).
            # But test_env.step() does a lot of bookkeeping.
            
            # Let's try to reverse engineer the action? No, that's hard.
            # Alternative: Just use the model's action directly for the step, 
            # BUT check if the Strategy Wrapper would have overridden it.
            # If Strategy says "CRASH" -> Weights = [0,0,...1].
            # We can construct a "fake" action that results in 100% cash.
            # Action = [-inf, -inf, ..., 100] -> Softmax -> [0, 0, ..., 1]
            
            if regime == "CRASH":
                # Force Cash
                action = np.full(test_env.n_assets + 1, -10.0, dtype=np.float32)
                action[-1] = 10.0
            elif regime == "RALLY":
                # Force Invested (No Cash)
                # We want to keep the relative weights of assets but zero out cash.
                # Get raw model action
                raw_action, _ = model.predict(obs, deterministic=True)
                if raw_action.ndim > 1: raw_action = raw_action[0]
                
                # Set cash logit to very low
                action = raw_action.copy()
                action[-1] = -10.0
            else:
                # Normal Regime
                action, _ = model.predict(obs, deterministic=True)
                if action.ndim > 1: action = action[0]

            # Step the environment
            obs, reward, done, _, _ = test_env.step(action)
            step_idx += 1
            
            # Store History
            # We need the global date index.
            # full_prices index corresponds to the dates? 
            # We assume full_prices is a numpy array, so we need the date index passed in or inferred.
            # Let's just store the index for now.
            history.append({
                'index': train_end + step_idx,
                'nav': test_env.portfolio_value,
                'regime': regime,
                'benchmark': b_test[curr_step_env]
            })
            
        # 5. Slide Window
        current_start += test_window_days
        current_balance = test_env.portfolio_value
        print(f"✓ Completed Window: {train_end} to {test_end} | Final NAV: {test_env.portfolio_value:.2f}")

    return pd.DataFrame(history)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # 1. Load Data
    df = load_real_data(tickers=config.TICKERS, benchmark=config.BENCHMARK, start=config.START_DATE, end=config.END_DATE)
    
    # 2. Feature Engineering
    engineer = FinancialFeatureEngineer()
    features, prices, benchmark, ticker_names = engineer.preprocess_data(df)
    
    # 3. Run Rolling Backtest
    # Using config values for panic_vol and strong_trend if available, else defaults
    # Note: config.PANIC_VOL_RANGE is a tuple, we pick a value or use a fixed best value if known.
    # The user provided BEST_FAST_VOL in config, let's use that if available, or the range start.
    
    print("Starting Rolling Window Backtest...")
    results_df = rolling_window_train_predict(
        prices, features, benchmark,
        train_window_days=750,
        test_window_days=60,
        ticker_names=ticker_names
    )
    
    # 4. Plot Results
    if not results_df.empty:
        # Map index back to dates
        results_df['date'] = df.index[results_df['index'].values]
        results_df = results_df.set_index('date')
        
        plt.figure(figsize=(12, 6))
        
        # Normalize Benchmark to match NAV start
        results_df['bench_norm'] = (results_df['benchmark'] / results_df['benchmark'].iloc[0]) * results_df['nav'].iloc[0]
        
        plt.plot(results_df.index, results_df['nav'], label='Hedge Fund AI', color='#00ff00')
        plt.plot(results_df.index, results_df['bench_norm'], label='Benchmark (QQQ)', color='gray', linestyle='--')
        
        # Highlight Regimes
        y_min, y_max = plt.gca().get_ylim()
        plt.fill_between(results_df.index, y_min, y_max, where=(results_df['regime']=='RALLY'), color='green', alpha=0.1, label='Rally')
        plt.fill_between(results_df.index, y_min, y_max, where=(results_df['regime']=='CRASH'), color='red', alpha=0.1, label='Crash')
        
        total_return = ((results_df['nav'].iloc[-1] / results_df['nav'].iloc[0]) - 1) * 100
        plt.title(f"Rolling Walk-Forward Backtest | Total Return: {total_return:.1f}%")
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        # Save plot
        plt.savefig("rolling_backtest_result.png")
        print("✓ Backtest Complete. Results saved to rolling_backtest_result.png")
    else:
        print("⚠️ No results generated.")
