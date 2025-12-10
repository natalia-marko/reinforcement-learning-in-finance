
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

# Import our system modules
import config
from rl_system_v0 import FinancialFeatureEngineer, BullEnv, BearEnv, SniperEnv, AdaptiveBoard

# 1. DATA LOADING & PREPARATION
print("⏳ Fetching Data (Tech + Gold)...")
tickers = config.TICKERS + [config.BENCHMARK]
# Fetch enough history for Train (2018-2021), Val (2022), Test (2023-2024)
raw_data = yf.download(tickers, start="2018-01-01", end="2024-12-30", progress=False, auto_adjust=True)['Close']
df = raw_data.ffill().dropna()

print("🛠️ Engineering Features...")
fe = FinancialFeatureEngineer()
features, prices, benchmark = fe.preprocess_data(df)
dates = df.index

# 2. WALK-FORWARD SPLIT
# We need to map dates to array indices
train_mask = (dates >= '2018-01-01') & (dates <= '2021-12-31')
val_mask   = (dates >= '2022-01-01') & (dates <= '2022-12-31')
test_mask  = (dates >= '2023-01-01')

# Indices
train_idx = np.where(train_mask)[0]
val_idx   = np.where(val_mask)[0]
test_idx  = np.where(test_mask)[0]

# Create Datasets
# Note: We slice the numpy arrays directly
train_features, train_prices, train_bench = features[train_idx], prices[train_idx], benchmark[train_idx]
val_features,   val_prices,   val_bench   = features[val_idx],   prices[val_idx],   benchmark[val_idx]
test_features,  test_prices,  test_bench  = features[test_idx],  prices[test_idx],  benchmark[test_idx]

print(f"\n📊 Data Split:")
print(f"Train: {len(train_idx)} days (2018-2021)")
print(f"Val:   {len(val_idx)} days (2022 - The Bear Market)")
print(f"Test:  {len(test_idx)} days (2023-2024)")

# 3. TRAINING AGENTS (ON TRAIN SET ONLY)
os.makedirs("models", exist_ok=True)
AGENTS = [
    ('bull', BullEnv, 60_000), 
    ('bear', BearEnv, 80_000), 
    ('sniper', SniperEnv, 60_000)
]

print("\n🏋️ Training Agents on 2018-2021 Data...")
for name, env_class, steps in AGENTS:
    if not os.path.exists(f"models/agent_{name}.zip"): # Avoid retraining if exists for speed
        print(f"   Training {name.upper()}...")
        env = env_class(train_prices, train_features, train_bench)
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, seed=42)
        model.learn(total_timesteps=steps)
        model.save(f"models/agent_{name}")
    else:
        print(f"   Loaded {name.upper()} from disk.")

# 4. OPTIMIZATION (GRID SEARCH ON VALIDATION SET - 2022)
print("\n🔍 Grid Search on VALIDATION Set (2022)...")
# 2022 was a crash year, so this is critical for tuning the 'Panic' threshold
board = AdaptiveBoard("models/agent_bull", "models/agent_bear", "models/agent_sniper")

# Calculate Volatility for Validation Period
val_bench_series = pd.Series(val_bench)
val_vol_series = val_bench_series.pct_change().rolling(20).std().fillna(0).values

panic_options = [0.012, 0.015, 0.018, 0.020, 0.025]
choppy_options = [0.006, 0.008, 0.010, 0.012]

best_score = -999
best_params = (config.PANIC_VOL_THRESHOLD, config.CHOPPY_VOL_THRESHOLD)

for p_vol in panic_options:
    for c_vol in choppy_options:
        if c_vol >= p_vol: continue
        
        # Configure Board
        board.panic_thresh = p_vol
        board.choppy_thresh = c_vol
        
        # Run Simulation on VAL data
        sim_env = BullEnv(val_prices, val_features, val_bench) # Env type doesn't matter for sim, just physics
        obs, _ = sim_env.reset()
        done = False
        nav_history = []
        
        while not done:
            idx = sim_env.curr_step - sim_env.lookback
            if idx >= len(val_vol_series): vol = 0
            else: vol = val_vol_series[idx]
            
            weights, _ = board.predict(obs, market_vol=vol)
            action = np.log(weights + 1e-9)
            obs, _, done, _, info = sim_env.step(action)
            nav_history.append(info['nav'])
            
        # Calculate Sharpe
        nav_s = pd.Series(nav_history)
        rets = nav_s.pct_change().dropna()
        if len(rets) > 0 and rets.std() > 0:
            sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252))
        else:
            sharpe = 0
            
        if sharpe > best_score:
            best_score = sharpe
            best_params = (p_vol, c_vol)
            
        # print(f"P={p_vol} C={c_vol} -> Val Sharpe: {sharpe:.2f}")

print(f"🏆 Best Validation Parameters: Panic={best_params[0]}, Choppy={best_params[1]}")
print(f"   (Optimized for the 2022 Bear Market)")

# 5. FINAL TEST (ON TEST SET - 2023-2024)
print("\n📈 Running FINAL Backtest on TEST Set (2023-2024)...")
board.panic_thresh = best_params[0]
board.choppy_thresh = best_params[1]

# Calc Volatility for Test
test_bench_series = pd.Series(test_bench)
test_vol_series = test_bench_series.pct_change().rolling(20).std().fillna(0).values

sim_env = BullEnv(test_prices, test_features, test_bench)
obs, _ = sim_env.reset()
done = False
history = []

while not done:
    idx = sim_env.curr_step - sim_env.lookback
    if idx >= len(test_vol_series): vol = 0
    else: vol = test_vol_series[idx]
    
    weights, regime = board.predict(obs, market_vol=vol)
    action = np.log(weights + 1e-9)
    obs, _, done, _, info = sim_env.step(action)
    
    # Store Date
    # Map step index back to date array
    # Note: sim_env.curr_step corresponds to the *end* of the lookback window
    step_date = dates[test_idx[0] + idx + sim_env.lookback] if (test_idx[0] + idx + sim_env.lookback) < len(dates) else None
    
    history.append({
        'date': step_date,
        'nav': info['nav'],
        'regime': regime,
        'weights': info['weights']
    })

results = pd.DataFrame(history).set_index('date')
results = results.dropna() # Drop any padding issues

# Performance Metrics
total_ret = (results['nav'].iloc[-1] / results['nav'].iloc[0]) - 1
print(f"\n📝 FINAL TEST REPORT (2023-2024):")
print(f"Total Return: {total_ret:.2%}")
print(f"Final NAV:    ${results['nav'].iloc[-1]:,.2f}")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['nav'], label='Walk-Forward AI Agent')
plt.title("Walk-Forward Test Results (2023-2024)")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
