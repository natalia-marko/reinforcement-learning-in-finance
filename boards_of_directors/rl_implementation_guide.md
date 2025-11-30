# Reinforcement Learning System Guide

A Pragmatic, Robust Framework for Financial Portfolio Optimization via Reinforcement Learning

---

## Table of Contents

1. [Introduction: The Challenge in Financial RL](#introduction)
2. [Core Philosophy and Wrappers](#core-philosophy)
3. [System Architecture Overview](#system-architecture)
4. [Feature Engineering for RL Stability](#feature-engineering)
5. [Observation Windows and Temporal Mismatches](#observation-windows)
6. [Data Pipeline Design](#data-pipeline)
7. [Monthly vs. Daily: Action Frequency](#action-frequency)
8. [Reward Function Engineering](#reward-function)
9. [Ensemble Agent Architecture](#ensemble-architecture)
10. [Training Protocols and Evaluation](#training-protocols)
11. [Implementation Blueprint: Phase 1 - Data Engineering](#phase1)
12. [Implementation Blueprint: Phase 2 - Custom Gym Environment](#phase2)
13. [Implementation Blueprint: Phase 3 - Training Pipeline](#phase3)
14. [Implementation Blueprint: Phase 4 - Smart Wrapper](#phase4)
15. [Implementation Blueprint: Phase 5 - Backtest Engine](#phase5)
16. [Early Stopping in RL: Why It Fails](#early-stopping)
17. [Hyperparameter Justification](#hyperparameters)
18. [Ensemble Voting: Board of Directors Architecture](#board-of-directors)
19. [Best Practices Summary](#best-practices)
20. [Conclusions and Next Steps](#conclusions)

---

## Introduction: The Challenge in Financial RL {#introduction}

**The Problem:** Financial RL systems often underperform simple equal-weighted buy-hold strategies. This document addresses the core issues:
- Signal-to-noise ratio in finance is incredibly low
- RL agents are notorious for overfitting to noise, memorizing specific price paths rather than learning genuine market dynamics
- Most implementations suffer from improper normalization, temporal aliasing, and regime blindness

**The Solution:** Build a modular, transparent system using Stable Baselines3 and a custom Gym environment.

**Core Constraint:** Only rebalance if the future reward improvement is 5-10%, avoiding unnecessary churn and transaction costs.

---

## Core Philosophy and Wrappers {#core-philosophy}

### Separation of Concerns

**Crucial Design Choice:** Do not force the RL agent to learn discrete thresholds (like the 5-10 rule) inside the neural network. Sparse rewards make training unstable.

**The RL Job:** Output ideal portfolio weights for the next month based on risk-reward optimization.

**The Wrapper Job:** Compare the RL's ideal weights vs. current weights. If the difference doesn't justify the cost/effort (the 5-10 rule), do nothing.

### Why This Works

- **Decoupling:** Prediction (RL) is separate from execution logic (wrapper), making the system easier to debug and audit.
- **Stability:** Removing hard constraints from the neural network prevents the agent from gaming the system or getting stuck on reward plateaus.
- **Interpretability:** Each layer (agent, wrapper, executor) has a clear responsibility.

---

## System Architecture Overview {#system-architecture}

### A. The Environment

We need a custom Gym environment. **Standardizing the input is where 90% of the stability comes from.**

**Key Components:**
- **Assets:** Your specific portfolio list (fixed universe)
- **Data Frequency:** Daily data, but the step function skips 20 days (approx. 1 month) to simulate monthly decision-making
- **Transaction Costs:** Hard-coded into the environment (e.g., 0.1% = 10 basis points). If the agent changes weights, it must pay a fee. This forces it to be lazy and stable.

### B. The State Space

**Observation:** Avoid raw pricesâ€”they are non-stationary and unbounded. Use normalized features.

For a portfolio of N assets, the input at step t should be a window of past features:
1. **Log Returns:** \(R_t = \log(P_t / P_{t-1})\) â€” Stationary
2. **Volatility:** Rolling standard deviation
3. **Correlation Matrix:** Optional, but helps diversification logic
4. **Current Weights:** The agent needs to know what it currently holds to calculate the cost of moving

### C. The Action Space

- **Type:** Continuous Box space
- **Output:** A vector of size N+1 (Assets + Cash)
- **Activation:** Softmax
- **Constraint:** Ensures weights always sum to 1.0 (100% of portfolio)

### D. The Reward Function

**The most critical part.** Don't just use Profitâ€”that encourages reckless gambling.

**Proposed Reward:**
\[R_t = \frac{\text{Portfolio Return}_t - \text{Volatility}_t - \text{TransactionCost}_t}{\text{Risk}}\]

- If the agent churns the portfolio too much, TransactionCost kills the reward
- If the agent picks volatile assets, the volatility penalty lowers the reward
- This creates a natural bias toward stability and cost-consciousness

---

## Feature Engineering for RL Stability {#feature-engineering}

### The Selection

Don't throw 50 indicators at it. We need features that are:
- Stationary
- Bounded range
- Normalized

### The Golden 4 Categories for RL

#### 1. Momentum: Bounded RSI (Relative Strength Index)

**Why?** It's naturally bounded between 0 and 100. It tells the agent if a stock is stretched.

**Modification:** Scale it to [0, 1] or [-1, 1] for the neural network.

```
RSI_14 = standard RSI with period 14
RSI_normalized = (RSI_14 - 50) / 50  # Center around 0
```

#### 2. Trend Strength: PPO (Percentage Price Oscillator)

**Why?** Do not use MACD. MACD values depend on the priceâ€”a $200 stock has a higher MACD than a $10 stock. PPO is the percentage version of MACD. It is price-agnostic.

**Formula:**
\[\text{PPO} = \frac{\text{EMA}_{12} - \text{EMA}_{26}}{\text{EMA}_{26}}\]

#### 3. Volatility: Normalized ATR (Average True Range)

**Why?** Raw ATR is in dollars (bad). We need volatility as a percentage of price.

**Formula:**
\[\text{NATR} = \frac{\text{ATR}_{14}}{\text{Close Price}}\]

**Insight:** This helps the agent shrink position sizes in high-volatility environments.

#### 4. Volume: RVOL (Relative Volume)

**Why?** Raw volume is useless across different stocks. We need relative context.

**Formula:**
\[\text{RVOL} = \frac{\text{Current Volume}}{\text{SMA(Volume, 20)}}\]

**Signal:** High RVOL often precedes a trend change.

---

## Observation Windows and Temporal Mismatches {#observation-windows}

### The Monthly Problem

**The Engineering Challenge:** You want to trade monthly, but technicals are usually daily. If you feed the agent only the technicals of the last day of the month, it's like driving a car while blinking your eyes once every minute. You miss the contextâ€”the path.

### The Solution: Windowed Observation

Even though the agent makes a decision once a month (step size = 20 trading days), the observation must look back.

**Observation Shape:** \([\text{NAssets}, \text{NFeatures}, \text{WindowSize}]\)

**Recommended Window Size:** 60 days (approx. 3 months)

**Why 3 months?**
- For a monthly rebalancing decision, the agent needs to see the medium-term trend
- Allows agent to determine if an asset is in a structural uptrend or just having a lucky day
- Captures one full earnings cycle (fiscal quarter â‰ˆ 63 trading days)

### Signal Processing Perspective (Aliasing)

If you only feed the agent technical indicators at the end of the month, you are **under-sampling the signal**. You lose the path-dependencyâ€”volatility, max drawdown that happened between samples.

In DSP (Digital Signal Processing) terms, this is **aliasing**: high-frequency noise (volatility) masquerades as a low-frequency smooth trend.

**The Windowing Solution:** Use a lookback window of daily data (e.g., 60 days) even if you only make a decision once a month. The agent sees crashes inside its memory window, even if it acts at the end.

---

## Data Pipeline Design {#data-pipeline}

### The Preprocessing Flow

1. **Load Data:** OHLCV (Open, High, Low, Close, Volume) for your stocks
2. **Calculate Indicators:** RSI, PPO, NATR, RVOL using pandas-ta (vectorized, fast)
3. **Z-Score Normalization:** Critical step to prevent look-ahead bias
4. **Clip Outliers:** RL hates outliers; clip values > 3 std devs

### The Normalization Strategy

**Critical Problem:** You cannot use MinMaxScaler on the whole datasetâ€”that's look-ahead bias.

**Solution:** Rolling Z-Score

\[Z_t = \frac{X_t - \mu_{t-252:t}}{\sigma_{t-252:t}}\]

Normalize today's indicator value against its own history over the last trading year (252 days). This keeps values mostly between -3 and 3.

**Why Rolling?** Each day, the mean and standard deviation update. If you normalize using the global mean of the whole dataset, you're cheatingâ€”the agent would know future ranges.

### Output Structure

A 3D NumPy array:
\[[\text{TimeSteps}, \text{LookbackWindow}, \text{Features per Asset Ã— NAssets}]\]

For a KISS implementation with PPO, we flatten the features per time-step so the MLP (Multi-Layer Perceptron) can digest it easily.

---

## Monthly vs. Daily: Action Frequency {#action-frequency}

### Approach A: Daily Rebalancing (Active Agent)

**Logic:** The agent steps every single day (step size = 1)

**Pros:**
- Can react instantly to a crash

**Cons (The Killers):**
1. SNR (Signal-to-Noise Ratio) is near zero: Daily price movements are ~90% noise (Brownian motion) and ~10% signal
2. RL agents are noise chasersâ€”they find patterns where none exist
3. Transaction costs compound exponentially: A daily agent that churns 1% of portfolio daily destroys 250% turnover per year
4. Wash sales and taxes (if applicable)

### Approach B: Monthly Rebalancing (Strategic Agent)

**Logic:** The agent steps every 20-22 days

**Pros:**
1. High SNR: Monthly trends are more robust than daily ticks
2. Forced Stability: The agent cannot panic sell on a random Tuesday

**Cons:**
1. Locked in during mid-month crashes (the blind period)

### The Research Verdict

Academic and industry research generally favors **lower frequency** for portfolio rebalancing. 

Research (e.g., Advisor Perspectives, Kitces) suggests that Quarterly or Monthly rebalancing often beats Daily because it:
- Captures momentum
- Lets winners run rather than cutting them off too early
- Produces higher Sharpe ratios (daily rebalancing often produces lower Sharpe due to the cost of volatility harvesting exceeding the premium gained)

### The Solution: The Hybrid Monitor

**Goal:** Stability of monthly actions + safety of daily monitoring

**Design Recommendation:** Do not train a daily agent. Train a monthly agent with a daily circuit breaker.

#### How It Works

**The Brain (RL Agent):**
- Input: Rolling 60-day window of daily technicals (solves aliasing)
- Action: Outputs target weights once per month
- Goal: Optimize long-term Sharpe ratio

**The Bodyguard (Wrapper):**
- This is a simple Python script (not RL) that runs daily
- Checks risk control logic:
  - Is portfolio drawdown > 5% since last rebalance?
  - Is an individual asset down > 15%?
- Trigger: If and only if a circuit breaker is hit, it calls the RL agent early to ask for new weights

#### Why This Works

2. **Solves the Noise:** RL is not distracted by daily noise; it focuses on monthly trends.
3. **Solves the Blindness:** The Bodyguard prevents holding a crashing asset for 29 days.

**Decision:** Stick to monthly decisions for the RL agent. It is much easier to train and much more stable. Handle the context by feeding it a history window.

---

## Reward Function Engineering {#reward-function}

### Information vs. Noise Trade-off

You might worry that daily data is noisy. You're rightâ€”price is noisy. But **volatility is not noise; it is data.**

**Scenario:**
- Stock A is flat all month
- Stock B goes up 50% and down 50% to end flat

**Weekly/Monthly Input:** Sees both as flat (information loss!)

**Daily Input:** Sees stock A as stable and stock B as dangerous

Since your reward function is the Sharpe Ratio (risk-adjusted return), you need that daily noise to accurately measure the risk part. If you hide volatility from the agent (using weekly bars), it will inadvertently learn to take hidden risks.

### Best Practice Design

- **Input:** Daily bars (high granularity for risk awareness)
- **Action:** Monthly rebalancing (low frequency for cost control)
- **Concept:** Low-frequency trader with high-frequency eyes

The agent uses daily volatility inside the month to estimate the risk of the path, even if it only executes a trade at the end.

### The Intelligent Reward Function

Standard Sharpe Ratio is problematic because it penalizes **upside volatility** (sudden profits) just as much as **downside volatility** (crashes).

**For Max Return goal, use Dynamic Sortino Ratio:**

\[R_t = \text{Log Return}_t - \lambda_t \cdot \text{Downside Deviation}_t\]

Where \(\lambda_t\) (risk aversion) is **not static**. It should be dynamic based on market regime:
- **Calm Market:** Low Î» â†’ agent is allowed to be aggressive
- **Volatile Market:** High Î» â†’ agent is forced to be defensive

### Python Implementation: Dynamic Sortino in Gym Environment

```python
import numpy as np

def calculate_reward(self, current_portfolio_value, prev_portfolio_value, price_history_window):
    """
    Intelligent Reward: Sortino Ratio with Dynamic Risk Scaling
    """
    # 1. Calculate simple return for the step (month)
    step_return = (current_portfolio_value / prev_portfolio_value) - 1
    
    # 2. Calculate downside deviation
    daily_returns = price_history_window.pct_change().dropna()
    negative_returns = daily_returns[daily_returns < 0]
    
    if len(negative_returns) == 0:
        downside_deviation = 0.001  # Avoid division by zero
    else:
        downside_deviation = np.std(negative_returns)
    
    # 3. Dynamic risk aversion based on market regime
    market_vol = price_history_window['SPY'].pct_change().std()
    
    # If recent market volatility (e.g., last 20 days) is high, punish risk more
    dynamic_lambda = 2.0  # Base
    if market_vol > 0.015:  # 1.5% daily volatility = panic mode
        dynamic_lambda = 5.0 + (market_vol * 100)
    
    # 4. Calculate reward
    reward = step_return - (dynamic_lambda * downside_deviation)
    return reward
```

**Why This Is Better:**
- Ignores "good volatility" (upside pumps)
- Dynamically tightens the leash when markets get scary
- Penalizes only downside, not unexpected gains

---

## Ensemble Agent Architecture {#ensemble-architecture}

### The Concept: Ensemble Voting

Since you want to ensure robustness, do not rely on a single agent. In data science, **ensembles almost always beat single models.**

### The Proposal: Train 3 Specialist Agents

1. **The Bull (Aggressive):**
   - Trained only on bull-market data (e.g., 2013-2014, 2017, 2019, 2021)
   - Reward: Pure profit maximization
   - Personality: Buy the dip, leverage up

2. **The Bear (Paranoid):**
   - Trained on crash data (e.g., 2008, 2018 Q4, 2020 Q1, 2022)
   - Reward: Sortino + max drawdown penalty
   - Personality: Short or go to cash rapidly

3. **The Sideways (Sniper):**
   - Trained on choppy/sideways data
   - Reward: Mean reversion
   - Personality: Fade the move, trade the range

### Inference: Live Trading

Every month, the 3 agents vote on the weights:

\[\text{Final Weights} = \frac{W_{\text{Bull}} + W_{\text{Bear}} + W_{\text{Sideways}}}{3}\]

**Gated Logic:** If volatility > threshold, listen only to The Bear.

This creates a **committee decision-making process** that naturally becomes more defensive in crisis and more aggressive during calm periods.

---

## Training Protocols and Evaluation {#training-protocols}

### Walk-Forward Validation

Do NOT do a random 80/20 split. Use temporal validation:

1. **Training Period:** 2015-2019 (good and bad years)
2. **Validation Period:** 2020 (COVID crashâ€”excellent stress test)
3. **Test Period:** 2021-Present (out-of-sample)

### Early Stopping: Why It Fails in RL

In supervised learning, validation loss typically looks like a U-curve. You stop at the bottom.

In RL, the validation reward curve looks like a **heart attack monitorâ€”extremely noisy.**

**The Problem:** If you stop early because the agent hit a peak reward in the validation set, you might just be capturing a moment where the agent got lucky or found a loophole (e.g., "Always buy tech stocks").

**The Risk:** The agent hasn't converged; it just got a high score. It is likely unstable.

### The Solution: Save Best Checkpointing

Instead of stopping training, keep training until the end but **save snapshots** whenever the agent achieves a new high score on the evaluation set.

This allows you to analyze the best model later:
- Was it luck?
- Or was it robust?

### Implementation: EvalCallback (Stable Baselines3)

You need two separate environments:
1. `env`: Training gym (2010-2018)
2. `eval_env`: Validation gym (2019-2020)

```python
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Wrap validation environment
eval_env = PortfolioRebalanceEnv(
    price_data=prices_val,
    features_data=features_val,
    benchmark_data=benchmark_val,
    lookback_window=60
)
eval_env = Monitor(eval_env)

# Define callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=".logs/best_model",
    log_path=".logs",
    eval_freq=10000,  # Check the agent every 10,000 steps
    deterministic=True,  # Use deterministic mode for fair evaluation
    render=False,
    n_eval_episodes=5  # Test on 5 different random starts in eval set to average luck
)

# Train with callback
model.learn(
    total_timesteps=100000,
    callback=eval_callback
)
```

### Post-Training: The Robustness Test

Since you didn't stop early, you now have:
1. The `final_model` (state at step 100k)
2. The `best_model` (state at, say, step 74k where it peaked)

**Load both models and run them on the test set (2021-present):**

If `best_model` crashes but `final_model` survives, the best was just overfitting to validation noise.

If both perform similarly, you have a stable solution.

### Summary

| Aspect | Recommendation |
|--------|-----------------|
| Early Stopping | Noâ€”too sensitive to noise in RL |
| Model Checkpointing | Yesâ€”use EvalCallback |
| Evaluation Metric | Average of multiple episodes (n_eval_episodes=5) to smooth noise |
| Validation Frequency | Every 10,000 steps |

---

## Implementation Blueprint: Phase 1 - Data Engineering {#phase1}

### Goal
Create a stationary, mathematically sound 3D tensor from noisy market data.

### 1.1 Data Ingestion

**Source:** yfinance for prototyping or alpaca-py for real-time

**Universe:** A fixed list of your top 10-20 stocks + SPY as a market regime feature

**Format:** OHLCV at daily resolution

### 1.2 Feature Generation (Vectorized)

Use `pandas-ta` to calculate the Golden 4:
- **Momentum:** RSI(14)
- **Trend:** PPO(12, 26, 9)
- **Volatility:** NATR (Normalized ATR)
- **Volume:** RVOL (Relative Volume)
- **Macro Feature:** SPY distance from SMA(200) as a regime filter

### 1.3 Normalization Strategy

```python
import pandas as pd
import pandas_ta as ta
import numpy as np

def preprocess_data(df: pd.DataFrame, window_size=252) -> pd.DataFrame:
    """
    Compute technical indicators and apply rolling normalization.
    
    Args:
        df: DataFrame with OHLCV data
        window_size: Rolling window for normalization (default 252 = 1 trading year)
    
    Returns:
        Normalized feature DataFrame
    """
    # 1. Calculate indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['PPO'] = ta.ppo(df['Close'])[0]  # PPO line (not signal)
    df['NATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # 2. Rolling Z-Score Normalization (no look-ahead bias!)
    cols_to_norm = ['RSI', 'PPO', 'NATR', 'RVOL']
    for col in cols_to_norm:
        rolling_mean = df[col].rolling(window_size).mean()
        rolling_std = df[col].rolling(window_size).std()
        df[f'{col}_norm'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
    
    # 3. Clip outliers (RL hates infinity)
    df[df._norm'].clip(-5, 5)  # Clip to Â±5 std
    
    # 4. Fill NaNs from rolling windows
    df = df.fillna(0)
    
    return df
```

### 1.4 Output Structure

A 3D NumPy array ready for the Gym environment:
\[[\text{TimeSteps}, \text{LookbackWindow}, \text{Features Ã— Assets}]\]

```python
def shape_features(normalized_df, lookback_window=60, n_assets=10):
    """
    Convert normalized DataFrame into 3D tensor for RL agent.
    
    Args:
        normalized_df: Output from preprocess_data()
        lookback_window: Number of days to look back (default 60)
        n_assets: Number of assets in portfolio
    
    Returns:
        3D array [TimeSteps, LookbackWindow, Features Ã— Assets]
    """
    n_features = 4  # RSI, PPO, NATR, RVOL (normalized)
    n_timesteps = len(normalized_df) - lookback_window
    
    features_3d = np.zeros((n_timesteps, lookback_window, n_features))
    
    for t in range(n_timesteps):
        window_df = normalized_df.iloc[t:t+lookback_window]
        # Flatten features for each asset
        features_3d[t] = window_df[['RSI_norm', 'PPO_norm', 'NATR_norm', 'RVOL_norm']].values
    
    return features_3d
```

---

## Implementation Blueprint: Phase 2 - Custom Gym Environment {#phase2}

### Goal
A class that behaves like a game but respects financial math.

### 2.1 Architecture

Inherit from `gym.Env` (or `gymnasium.Env` for newer versions).

### 2.2 Initialization

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PortfolioRebalanceEnv(gym.Env):
    """
    A Monthly Rebalancing Environment with Daily Risk Awareness.
    
    State: Rolling window of technicals (T-60 to T)
    Action: Target portfolio weights (allocations)
    Reward: Dynamic Sortino Ratio (calculated on daily equity curve within the month)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        features_data: np.ndarray,
        benchmark_data: pd.Series,
        initial_balance: float = 100000,
        trading_cost_bps: float = 0.0010,  # 10 basis points = 0.1%
        lookback_window: int = 60,
        max_steps: int = 1000
    ):
        super(PortfolioRebalanceEnv, self).__init__()
        
        # Store data
        self.prices = price_data.values  # NumPy array for speed
        self.features = features_data
        self.benchmark_vol = benchmark_data.rolling(20).std().values  # Pre-calc market vol
        self.dates = price_data.index
        
        # Portfolio dimensions
        self.n_assets = self.prices.shape[1]
        self.n_features = self.features.shape[2]
        self.lookback_window = lookback_window
        
        # Financial parameters
        self.initial_balance = initial_balance
        self.trading_cost_bps = trading_cost_bps
        
        # Action space: continuous weights for each asset + cash
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_assets + 1,),  # +1 for cash
            dtype=np.float32
        )
        
        # Observation space: windowed features
        self.observation_space = spaces.Box(
            low=-5,
            high=5,
            shape=(self.lookback_window, self.n_assets * self.n_features),
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.cash = 0
        self.shares = np.zeros(self.n_assets)
        self.portfolio_value = initial_balance
        self.history = []
    
    def reset(self, seed=None, options=None):
        """Reset the environment to a random starting point."""
        super().reset(seed=seed)
        
        # Start somewhere after the lookback window
        self.current_step = np.random.randint(
            self.lookback_window,
            len(self.prices) - 22  # Leave room for at least one month
        )
        
        # Reset portfolio
        self.cash = self.initial_balance
        self.shares = np.zeros(self.n_assets)
        self.portfolio_value = self.initial_balance
        self.history = []
        
        return self.get_observation(), {}
    
    def get_observation(self):
        """Return the windowed observation."""
        return self.features[
            self.current_step - self.lookback_window:self.current_step
        ].astype(np.float32)
    
    def step(self, action):
        """
        Execute one step of the environment: Daily/Monthly Hybrid
        
        1. Rebalance portfolio based on action
        2. Apply transaction costs
        3. Simulate 20 days of holding with daily volatility capture
        4. Calculate intelligent reward
        5. Return observation, reward, done, info
        """
        # 1. Convert raw action to weights via softmax (ensures sum to 1)
        exp_action = np.exp(action - np.max(action))  # Numerical stability
        weights = exp_action / np.sum(exp_action)
        asset_weights = weights[:-1]  # All but last
        cash_weight = weights[-1]    # Last is cash
        
        # 2. Calculate target values
        current_prices = self.prices[self.current_step]
        target_values = self.portfolio_value * asset_weights
        current_holdings_val = self.shares * current_prices
        
        # 3. Calculate transaction costs and deduct
        trade_diffs = np.abs(target_values - current_holdings_val)
        total_trade_volume = np.sum(trade_diffs)
        cost = total_trade_volume * self.trading_cost_bps
        
        # 4. Rebalance: Update cash and shares
        self.cash = self.portfolio_value * cash_weight - cost
        self.shares = target_values / current_prices
        self.portfolio_value = self.cash + np.sum(self.shares * current_prices)
        
        # 5. Simulate 20 days of holding (capture daily volatility)
        daily_returns = []
        steps_to_take = min(20, len(self.prices) - self.current_step - 1)
        
        for i in range(steps_to_take):
            self.current_step += 1
            daily_prices = self.prices[self.current_step]
            
            # Calculate daily NAV
            daily_value = self.cash + np.sum(self.shares * daily_prices)
            
            # Log return for reward calculation
            if len(self.history) > 0:
                prev_val = self.history[-1]['nav']
                daily_ret = np.log(daily_value / prev_val)
            else:
                daily_ret = 0
            
            daily_returns.append(daily_ret)
            self.history.append({
                'step': self.current_step,
                'nav': daily_value
            })
        
        self.portfolio_value = self.history[-1]['nav'] if self.history else self.portfolio_value
        
        # 6. Calculate intelligent reward
        reward = self.calculate_dynamic_sortino(daily_returns)
        
        # 7. Check termination
        terminated = self.current_step >= len(self.prices) - 2
        truncated = False
        
        obs = self.get_observation()
        info = {'portfolio_value': self.portfolio_value}
        
        return obs, reward, terminated, truncated, info
    
    def calculate_dynamic_sortino(self, daily_log_returns):
        """
        Calculate reward as Dynamic Sortino Ratio.
        
        Args:
            daily_log_returns: List of log returns during the holding period
        
        Returns:
            Reward value
        """
        returns = np.array(daily_log_returns)
        
        # 1. Monthly return = sum of log returns
        total_return = np.sum(returns)
        
        # 2. Downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            downside_std = 0.001  # Avoid division by zero
        else:
            downside_std = np.std(negative_returns)
        
        # 3. Dynamic risk aversion based on market regime
        market_vol = self.benchmark_vol[self.current_step - 20:self.current_step].mean()
        
        # If market vol is high (e.g., 1.5% daily), increase risk penalty
        risk_aversion = 2.0
        if market_vol > 0.015:
            risk_aversion = 5.0 + (market_vol * 100)
        
        # 4. Calculate reward
        reward = total_return - (risk_aversion * downside_std)
        
        return reward
```

---

## Implementation Blueprint: Phase 3 - Training Pipeline {#phase3}

### Goal
Train a stable PPO agent with proper monitoring and checkpointing.

### 3.1 Algorithm Selection

**Algorithm:** PPO (Proximal Policy Optimization)

**Why PPO?**
- Robust, handles continuous action spaces naturally
- Less sensitive to hyperparameter tuning than DDPG or SAC
- Stable-baselines3 implementation is production-ready

### 3.2 Training with EvalCallback

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# 1. Create training and validation environments
env = PortfolioRebalanceEnv(
    price_data=prices_train,
    features_data=features_train,
    benchmark_data=benchmark_train,
    lookback_window=60
)

eval_env = PortfolioRebalanceEnv(
    price_data=prices_val,
    features_data=features_val,
    benchmark_data=benchmark_val,
    lookback_window=60
)

# Wrap in Monitor for statistics tracking
eval_env = Monitor(eval_env)

# 2. Define callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=".logs/best_model",
    log_path=".logs",
    eval_freq=10000,
    deterministic=True,
    render=False,
    n_eval_episodes=5
)

# 3. Define and train model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,  # Big batch size for stable gradients
    batch_size=64,
    gamma=0.99,  # Discount factor: 0.99 = cares about 3-6 months future
    ent_coef=0.01,  # Entropy coefficient: force exploration
    tensorboard_log=".logs/ppo_portfolio_tensorboard"
)

# 4. Train
model.learn(
    total_timesteps=100000,
    callback=eval_callback
)

# 5. Save final model
model.save(".logs/final_model")
```

### 3.3 Hyperparameter Justification

| Parameter | Value | Justification |
|-----------|-------|----------------|
| `learning_rate` | 3e-4 | Conservative; prevents instability in financial data |
| `n_steps` | 2048 | Large batches reduce variance in financial RL |
| `batch_size` | 64 | Standard; balances memory and stability |
| `gamma` | 0.99 | Long discount horizon for portfolio decisions |
| `ent_coef` | 0.01 | Force exploration; small value avoids premature convergence |
| `lookback_window` | 60 | One fiscal quarter (earnings cycle relevance) |
| `eval_freq` | 10000 | Every ~5 trading months in training data |

---

## Implementation Blueprint: Phase 4 - Smart Wrapper {#phase4}

### Goal
The gatekeeper logic that decides whether to execute the RL's suggested trade based on the 5-10% threshold.

### 4.1 SmartPortfolioExecutor Class

```python
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

class SmartPortfolioExecutor:
    """
    The Bodyguard for the RL Agent.
    
    Handles:
    1. Data formatting (shape management)
    2. Inertia logic (the 5-10 rule)
    3. Action smoothing
    """
    
    def __init__(self, model_path: str, threshold_turnover: float = 0.10):
        """
        Args:
            model_path: Path to trained PPO model
            threshold_turnover: Minimum portfolio turnover to execute (default 10%)
                               "10% means we need at least 10% of portfolio to change hands"
        """
        self.model = PPO.load(model_path)
        self.threshold = threshold_turnover
        self.last_weights = None
    
    def get_action(self, current_observation, current_weights):
        """
        Decide whether to listen to AI or stay the course.
        
        Args:
            current_observation: Window [Assets, Features] array
            current_weights: NumPy array of current portfolio allocation
        
        Returns:
            (final_weights, trade_executed, message)
        """
        # 1. Get agent prediction
        action, _ = self.model.predict(current_observation, deterministic=True)
        
        # 2. Convert to weights via softmax
        exp_action = np.exp(action - np.max(action))
        target_weights = exp_action / np.sum(exp_action)
        
        # 3. Calculate turnover
        turnover = np.sum(np.abs(target_weights - current_weights))
        
        # 4. The Decision
        if turnover < self.threshold:
            # Not enough edge to justify trading
            return current_weights, False, f"HOLD (Turnover {turnover:.2%} < {self.threshold:.2%})"
        else:
            # Execute rebalance
            return target_weights, True, f"TRADE (Turnover {turnover:.2%} >= {self.threshold:.2%})"
    
    def get_regime_weights(self, market_vol):
        """
        Get ensemble weights based on market regime.
        
        This is where you can integrate the Bull/Bear/Sideways agents.
        
        Args:
            market_vol: Current market volatility
        
        Returns:
            Weights for agent selection
        """
        if market_vol > 0.02:  # Extreme stress (2% daily vol)
            return {"bull": 0.0, "bear": 0.8, "sniper": 0.2}
        elif market_vol > 0.015:  # High stress
            return {"bull": 0.2, "bear": 0.6, "sniper": 0.2}
        elif market_vol > 0.01:  # Normal
            return {"bull": 0.4, "bear": 0.2, "sniper": 0.4}
        else:  # Calm
            return {"bull": 0.7, "bear": 0.0, "sniper": 0.3}
```

---

## Implementation Blueprint: Phase 5 - Backtest Engine {#phase5}

### Goal
Manual simulation to properly test the wrapper logic. In standard Gym, `env.step()` forces a trade. In a smart backtest, we simulate the option to do nothing.

### 5.1 BacktestEngine Class

```python
import pandas as pd
import numpy as np
from typing import List, Dict

class BacktestEngine:
    """
    A standalone simulator for the Smart Wrapper strategy.
    
    Replicates the Gym environment's physics but strictly separates prediction from execution.
    Records an audit trail of all decisions.
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        features_data: np.ndarray,
        benchmark_data: pd.Series,
        model: PPO,
        executor: SmartPortfolioExecutor,
        initial_capital: float = 100000,
        trading_cost_bps: float = 0.0010
    ):
        self.price_data = price_data
        self.features = features_data
        self.benchmark_vol = benchmark_data.rolling(20).std().values
        self.model = model
        self.executor = executor
        self.initial_capital = initial_capital
        self.trading_cost_bps = trading_cost_bps
        
        # Portfolio state
        self.cash = initial_capital
        self.shares = np.zeros(price_data.shape[1])
        self.portfolio_value = initial_capital
        
        # History for analysis
        self.history = []
    
    def run(self, lookback_window: int = 60):
        """
        Run backtest over entire price history.
        
        Returns:
            DataFrame with daily NAV, decisions, and metrics
        """
        prices = self.price_data.values
        dates = self.price_data.index
        n_assets = prices.shape[1]
        
        # Start after lookback window
        start_index = lookback_window
        end_index = len(prices) - 22  # Leave room for a full step
        
        print(f"Starting Backtest from {dates[start_index]} to {dates[end_index-1]}")
        
        current_step = start_index
        
        while current_step < end_index:
            # 1. Get observation (windowed features)
            obs = self.features[current_step - lookback_window:current_step]
            
            # 2. Get current weights
            current_prices = prices[current_step]
            current_holdings_val = self.shares * current_prices
            total_val = self.cash + np.sum(current_holdings_val)
            
            if total_val > 0:
                current_weights = np.append(
                    current_holdings_val / total_val,
                    self.cash / total_val
                )
            else:
                current_weights = np.zeros(n_assets + 1)
                current_weights[-1] = 1.0  # All cash
            
            # 3. Get executor decision
            target_weights, executed, message = self.executor.get_action(obs, current_weights)
            
            # 4. Execute if approved
            if executed:
                # Calculate transaction cost
                target_asset_values = total_val * target_weights[:-1]
                trade_diffs = np.abs(target_asset_values - current_holdings_val)
                cost = np.sum(trade_diffs) * self.trading_cost_bps
                
                # Rebalance
                self.cash = total_val * target_weights[-1] - cost
                self.shares = target_asset_values / current_prices
                turnover = np.sum(np.abs(target_weights - current_weights))
            else:
                turnover = 0
            
            # 5. Simulate 20 days of holding
            steps_to_take = min(20, end_index - current_step)
            daily_navs = []
            
            for i in range(steps_to_take):
                current_step += 1
                daily_prices = prices[current_step]
                nav = self.cash + np.sum(self.shares * daily_prices)
                daily_navs.append(nav)
                
                self.history.append({
                    'date': dates[current_step],
                    'nav': nav,
                    'action': message if i == 0 else 'HOLD',
                    'turnover': turnover if i == 0 else 0
                })
            
            self.portfolio_value = daily_navs[-1] if daily_navs else self.portfolio_value
        
        # Convert to DataFrame
        results = pd.DataFrame(self.history).set_index('date')
        print(f"Backtest Complete.")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Final Portfolio Value: ${self.portfolio_value:,.0f}")
        print(f"Total Return: {(self.portfolio_value / self.initial_capital - 1)*100:.2f}%")
        
        return results
```

### 5.2 Running the Backtest

```python
# 1. Load trained model and executor
model = PPO.load(".logs/best_model")
executor = SmartPortfolioExecutor(".logs/best_model", threshold_turnover=0.05)

# 2. Run backtest on test set
backtest_engine = BacktestEngine(
    price_data=prices_test,
    features_data=features_test,
    benchmark_data=benchmark_test,
    model=model,
    executor=executor,
    initial_capital=100000,
    trading_cost_bps=0.001
)

results = backtest_engine.run(lookback_window=60)

# 3. Analyze results
print(results.head(20))
print("\n---\n")

# Calculate metrics
returns = results['NAV'].pct_change()
cumulative_return = (results['NAV'] / results['NAV'].iloc[0]) - 1
sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
max_drawdown = ((results['NAV'].cummax() - results['NAV']) / results['NAV'].cummax()).max()

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown*100:.2f}%")
print(f"Final Return: {cumulative_return.iloc[-1]*100:.2f}%")

# 4. Visualize
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# NAV curve
ax1.plot(results.index, results['NAV'], label='RL Agent', linewidth=2)
ax1.set_ylabel('Portfolio Value ($)')
ax1.set_title('Backtest Result: RL Portfolio Optimization')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Drawdown
ax2.plot(results.index, max_drawdown * 100, label='Drawdown', color='red', linewidth=1)
ax2.fill_between(results.index, 0, max_drawdown * 100, alpha=0.2, color='red')
ax2.set_ylabel('Drawdown (%)')
ax2.set_xlabel('Date')
ax2.set_title('Underwater Plot')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Early Stopping in RL: Why It Fails {#early-stopping}

### The Problem

In supervised learning (XGBoost, neural nets), your validation loss curve usually looks like a U. You stop at the bottom and call it done.

In RL, the **validation reward curve looks like a heart attack monitor**â€”extremely noisy.

### The Trap

If you stop early because the agent hit a peak reward in the validation set, you might just be capturing a moment where the agent got lucky or found a loophole in that specific market regime.

**Example:** "Always buy tech stocks" might perform well in a bull market validation set but fail catastrophically in a bear market.

### The Solution

Instead of killing training early, **keep training and save checkpoints** whenever the agent achieves a new high score.

This allows you to:
1. Analyze why the best model was the best
2. Load multiple snapshots and compare their test performance
3. Determine if the best was luck or robust

### Implementation Summary

| Aspect | Old (Bad) | New (Good) |
|--------|-----------|-----------|
| Stopping | Stop when validation reward peaks | Keep training |
| Checkpointing | Save only final model | Save best model via EvalCallback |
| Evaluation | Single episode on validation | Average 5 episodes to smooth noise |
| Post-Training Test | Run once on test set | Run best and final on test, compare |

---

## Hyperparameter Justification {#hyperparameters}

This section explains the "magic numbers" behind the design, so you can defend or tune them.

### 1. The Lookback Window: 60 Days

**Why not 30? Why not 365?**

**The Financial Reason:** Stock prices are heavily influenced by earnings cycles. A fiscal quarter is ~63 trading days. A 60-day window ensures the agent sees exactly **one full cycle**: run-up to earnings, the event, and post-earnings drift.

**The Signal Processing Reason:** If you use 30 days and calculate a 20-day moving average (standard technical indicator), you're left with only 10 days of valid data. 60 days gives enough tail for indicators like RSI and MACD to stabilize.

**The Academic Reference:** Jegadeesh & Titman (1993) on momentum strategies found that momentum is strongest in the 3-12 month range. 60 days = 3 months is the aggressive end of medium-term momentum.

### 2. The Rebalancing Threshold: 5%

**Why not rebalance on every 1% drift?**

**The No-Trade Region:**
- Slippage + commission: 0.10%
- Bid-ask spread: 0.05%
- Market noise (Brownian motion): 1-2% per month

If we rebalance every time the portfolio drifts 1%, we're fighting **noise**, not trading **signal**.

**The Math:** You only rebalance if:
\[\text{Expected Alpha} > \text{Transaction Cost} + \text{Tax Impact}\]

For most passive portfolios, this threshold is 5-10%.

### 3. Transaction Costs: 0.1% (10 basis points)

**Includes:**
- Typical commission: 0.02-0.05% per side
- Bid-ask spread: 0.01-0.05%
- Market impact (small portfolios): negligible
- Slippage: 0.01-0.02%

For a $100k portfolio trading $10k (10% rebalance), total cost â‰ˆ $10, or 0.01% of portfolio. Conservative estimate: 0.1%.

### 4. PPO Hyperparameters

| Param | Value | Reason |
|-------|-------|--------|
| `learning_rate` | 3e-4 | Conservative for financial data (noisy) |
| `n_steps` | 2048 | Large batches reduce variance |
| `batch_size` | 64 | Standard; balances memory and stability |
| `gamma` | 0.99 | Long horizon (values future 3-6 months out) |
| `ent_coef` | 0.01 | Force exploration; prevents premature convergence |
| `clip_range` | 0.2 | PPO default; good for continuous actions |

### 5. Dynamic Risk Aversion: Î» = 2.0 to 5.0

**Base:** Î» = 2.0 in calm markets (allows aggression)

**Spike:** Î» = 5.0+ when market volatility > 1.5% daily

**Why?** In a crash, you want the agent to de-risk dramatically. In calm markets, you want it to take calculated risks.

---

## Ensemble Voting: Board of Directors Architecture {#board-of-directors}

### The Concept: Curriculum Learning

We don't code 3 different algorithms. We train 3 identical PPO clones but feed them different diets of data.

**Agent A (Bull):** Sees mostly bull-market data (2013, 2014, 2017, 2019, 2021)
- Learns to buy the dip and leverage up

**Agent B (Bear):** Sees crash data (2008, 2018 Q4, 2020 Q1, 2022)
- Learns to short or go to cash rapidly

**Agent C (Sideways):** Sees the entire dataset
- Learns the baseline

### Training the 3 Personalities

```python
# 1. Filter data by year
bull_years = df_full[df_full.index.year.isin([2013, 2014, 2017, 2019, 2021])]
bear_years = df_full[df_full.index.year.isin([2008, 2018, 2020, 2022])]
all_years = df_full

# 2. Create environments
env_bull = PortfolioRebalanceEnv(
    price_data=bull_years,
    features_data=features_bull,
    benchmark_data=benchmark_bull
)

env_bear = PortfolioRebalanceEnv(
    price_data=bear_years,
    features_data=features_bear,
    benchmark_data=benchmark_bear
)

env_sniper = PortfolioRebalanceEnv(
    price_data=all_years,
    features_data=features_all,
    benchmark_data=benchmark_all
)

# 3. Train each specialist
model_bull = PPO("MlpPolicy", env_bull, learning_rate=3e-4, verbose=1)
model_bull.learn(total_timesteps=50000)
model_bull.save("agent_bull")

model_bear = PPO("MlpPolicy", env_bear, learning_rate=3e-4, verbose=1)
model_bear.learn(total_timesteps=50000)
model_bear.save("agent_bear")

model_sniper = PPO("MlpPolicy", env_sniper, learning_rate=3e-4, verbose=1)
model_sniper.learn(total_timesteps=100000)
model_sniper.save("agent_sniper")
```

### Dynamic Ensemble Decision

```python
class EnsembleExecutor(SmartPortfolioExecutor):
    """
    3-Agent voting system with market regime gating.
    """
    
    def __init__(self, models_dict, threshold_turnover=0.10):
        """
        Args:
            models_dict: Dict with keys 'bull', 'bear', 'sniper' (PPO models)
            threshold_turnover: Min turnover to execute
        """
        self.models = models_dict
        self.threshold = threshold_turnover
    
    def get_action(self, current_observation, current_weights, market_vol):
        """
        Ensemble vote with market regime gating.
        
        Args:
            current_observation: Windowed features
            current_weights: Current portfolio allocation
            market_vol: Current market volatility (e.g., 252-day realized vol)
        
        Returns:
            (final_weights, trade_executed, message)
        """
        # 1. Get opinions from all three agents
        w_bull, _ = self.models['bull'].predict(current_observation, deterministic=True)
        w_bear, _ = self.models['bear'].predict(current_observation, deterministic=True)
        w_sniper, _ = self.models['sniper'].predict(current_observation, deterministic=True)
        
        # Convert actions to weights
        w_bull_weights = self._softmax(w_bull)
        w_bear_weights = self._softmax(w_bear)
        w_sniper_weights = self._softmax(w_sniper)
        
        # 2. Weighted ensemble based on regime
        if market_vol > 0.02:  # Extreme stress
            weights = np.array([0.0, 0.8, 0.2])  # Listen to Bear
            regime = "CRASH_DEFENSE"
        elif market_vol > 0.015:  # High stress
            weights = np.array([0.2, 0.6, 0.2])
            regime = "CAUTION"
        elif market_vol > 0.01:  # Normal
            weights = np.array([0.4, 0.2, 0.4])
            regime = "BALANCED"
        else:  # Calm
            weights = np.array([0.7, 0.0, 0.3])
            regime = "GROWTH"
        
        # Weighted average
        final_weights = (
            weights[0] * w_bull_weights +
            weights[1] * w_bear_weights +
            weights[2] * w_sniper_weights
        )
        
        # 3. Apply turnover threshold
        turnover = np.sum(np.abs(final_weights - current_weights))
        
        if turnover < self.threshold:
            return current_weights, False, f"HOLD ({regime}, Turnover {turnover:.2%})"
        else:
            return final_weights, True, f"TRADE ({regime}, Turnover {turnover:.2%})"
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
```

### Why This Works

1. **Bull agents** capture upside momentum during expansions
2. **Bear agents** protect capital during downturns
3. **Sideways agents** provide baseline stability
4. **Regime gating** ensures the right personality is dominant

Result: A portfolio that is **adaptive without being reactive**, **aggressive without being reckless**, and **defensive without being paralyzed**.

---

## Best Practices Summary {#best-practices}

### Data Handling
- âœ… Use rolling Z-score normalization (no look-ahead bias)
- âœ… Clip outliers to Â±3 std deviations
- âœ… Use daily data for input, monthly for actions
- âœ… Validate that features are stationary (ADF test if paranoid)

### Environment Design
- âœ… Use Box action space (continuous weights)
- âœ… Ensure softmax normalization inside step() function
- âœ… Track portfolio drift daily even if decisions are monthly
- âœ… Hard-code transaction costs; don't expect agent to learn them

### Reward Function
- âœ… Use Sortino ratio (downside only), not Sharpe (symmetric)
- âœ… Make risk aversion dynamic based on market regime
- âœ… Penalize transactions explicitly in reward
- âœ… Test reward function on synthetic data first

### Training
- âœ… Use EvalCallback to save best checkpoints (not early stopping)
- âœ… Walk-forward validation (train/validation/test split by time)
- âœ… Train on diverse market regimes (bull, bear, sideways)
- âœ… Use separate random seeds for train/eval to avoid overfitting to initialization

### Testing & Validation
- âœ… Load best_model and final_model; compare on test set
- âœ… Calculate Sharpe, max drawdown, and Sortino on test set
- âœ… Compare against equal-weight and buy-hold benchmarks
- âœ… Audit all trades: record why each trade was made or skipped

### Execution (Real or Paper Trading)
- âœ… Use the BacktestEngine to verify wrapper logic before live
- âœ… Monitor turnover: if > 50% annually, model is overtrading
- âœ… Track slippage vs. estimated costs
- âœ… Rebalance during liquid hours (open 30 min to close 30 min, avoid overnight gaps)

---

## Conclusions and Next Steps {#conclusions}

### What You Now Have

A **complete, production-ready framework** for financial RL that addresses:

1. **Stationarity:** Rolling Z-score normalization
2. **Path Dependency:** Windowed observations and daily simulation
3. **Regime Awareness:** Dynamic Sortino reward function
4. **Overfitting:** EvalCallback and save-best strategy
5. **Practical Constraints:** Wrapper logic and circuit breakers
6. **Robustness:** Ensemble voting and curriculum learning

### Immediate Next Steps

1. **Gather Data:** OHLCV for your portfolio from yfinance or alpaca-py
2. **Feature Engineering:** Run Phase 1 (FinancialFeatureEngineer) on your data
3. **Environment Check:** Instantiate PortfolioRebalanceEnv and verify gym compatibility
4. **Training:** Run Phase 3 (train with EvalCallback) on your train/val split
5. **Backtest:** Run Phase 5 (BacktestEngine) on test set with SmartPortfolioExecutor
6. **Visualization:** Plot NAV, drawdown, allocation heatmap

### Customization Points

- **Lookback window:** Tune from 30-90 days based on your market microstructure
- **Rebalancing frequency:** Adjust from weekly to quarterly if desired
- **Reward function:** Switch from Sortino to other metrics (returns/max_drawdown, Calmar ratio, etc.)
- **Transaction costs:** Update with your actual broker fees and spreads
- **Universe:** Expand from 10 stocks to 50+ if you have sufficient data

### Final Validation

Before going live, ensure:
- [ ] Backtest Sharpe > 0.5 on test set
- [ ] Max drawdown < 20% on test set
- [ ] Annual turnover < 200% (sanity check)
- [ ] Agent beats equal-weight baseline on test set
- [ ] Best model and final model perform similarly on test (not overfitting)

---

## Appendix: Code Templates

### A. Complete Feature Engineer Class

```python
import pandas as pd
import pandas_ta as ta
import numpy as np

class FinancialFeatureEngineer:
    """
    KISS Principle: Handles all data preparation OUTSIDE the environment.
    This ensures the Gym environment is fast and stateless regarding data transformations.
    """
    
    def __init__(self, use_technical_indicators=True):
        self.use_tech = use_technical_indicators
    
    def preprocess_data(self, df: pd.DataFrame, window_size=252) -> pd.DataFrame:
        """
        Input: DataFrame with MultiIndex (Date, Ticker) or wide format.
        Assumes columns: Open, High, Low, Close, Volume
        
        Output:
            1. normalized_tensor: 3D Array [Time, Assets, Features]
            2. raw_prices: DataFrame for calculating NAV [Time, Assets]
        """
        
        # 1. Calculate indicators
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['PPO'] = ta.ppo(df['Close'])[0]
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['NATR'] = df['ATR'] / df['Close']
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # 2. Rolling normalization (no look-ahead bias)
        cols_to_norm = ['RSI', 'PPO', 'NATR', 'RVOL']
        for col in cols_to_norm:
            rolling_mean = df[col].rolling(window_size).mean()
            rolling_std = df[col].rolling(window_size).std()
            df[f'{col}_norm'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        # 3. Clip outliers
        for col in cols_to_norm:
            df[f'{col}_norm'] = df[f'{col}_norm'].clip(-5, 5)
        
        # 4. Fill NaNs from rolling windows
        df = df.fillna(0)
        
        return df
```

### B. SmartPortfolioExecutor Usage Example

```python
# In your Jupyter notebook:

# 1. Load best model
from stable_baselines3 import PPO
best_model = PPO.load(".logs/best_model/best_model.zip")

# 2. Create executor
executor = SmartPortfolioExecutor(".logs/best_model/best_model.zip", threshold_turnover=0.05)

# 3. On each decision date:
obs = get_observation_from_features(features, current_step, lookback_window=60)
current_weights = calculate_current_weights(shares, cash, prices[current_step])
target_weights, executed, message = executor.get_action(obs, current_weights)

print(message)
if executed:
    # Rebalance portfolio
    execute_trade(target_weights)
```
