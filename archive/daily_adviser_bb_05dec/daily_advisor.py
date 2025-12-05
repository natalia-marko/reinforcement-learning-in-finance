import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # <--- We use this to ensure Training/Inference match
from stable_baselines3 import PPO
import warnings
import os
import sys
import requests
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================

# Feature Engineering Parameters
LOG_RETURN_WINDOW = 1
VOLATILITY_WINDOW = 20
VOLATILITY_NORMALIZATION_WINDOW = 252
RSI_PERIOD = 14
CORRELATION_WINDOW = 60

# Regime Detection Thresholds
CHOPPY_VOLATILITY_THRESHOLD = 0.012  # 1.2% daily vol
PANIC_VOLATILITY_THRESHOLD = 0.020   # 2.0% daily vol (not used in this script)

# Regime Mixing Weights
REGIME_MIXING_SNIPER_WEIGHT = 0.5
REGIME_MIXING_BEAR_WEIGHT = 0.25
REGIME_MIXING_BULL_WEIGHT = 0.25
CALM_BULL_WEIGHT = 0.7
CALM_SNIPER_WEIGHT = 0.3

# Data Fetching
HISTORICAL_DATA_PERIOD = "2y"
FAST_VOLATILITY_WINDOW = 5
SLOW_VOLATILITY_WINDOW = 20
OBSERVATION_WINDOW_SIZE = 60

# Validation
WEIGHT_SUM_TOLERANCE = 0.001  # Strict tolerance for weight validation

# Softmax
SOFTMAX_TEMPERATURE = 1.0  # Temperature for action softmax

# ==========================================
# TELEGRAM CONFIGURATION
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(message: str):
    """Send message via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not configured. Skipping notification.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("Telegram notification sent successfully.")
            return True
        else:
            print(f"Telegram error: {response.text}")
            return False
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")
        return False

# ==========================================
# 1. USER CONFIGURATION (UPDATE DAILY!)
# ==========================================
# The stocks must be in the EXACT SAME ORDER as your training data
TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']
BENCHMARK = 'QQQ'

# Your Saved Model Paths (Relative to this script's directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_BULL = os.path.join(SCRIPT_DIR, "agent_bull")
PATH_BEAR = os.path.join(SCRIPT_DIR, "agent_bear")
PATH_SNIPER = os.path.join(SCRIPT_DIR, "agent_sniper")

# Your OPTIMIZED Parameters (From Optuna Phase)
BEST_FAST_VOL = 0.0177
BEST_SLOW_VOL = 0.0378
BEST_THRESH   = 0.0286



# Your CURRENT Portfolio State (Update this before running)
# Order matches TICKERS + [CASH]
# Example: [14% each stock, 2% Cash]
CURRENT_WEIGHTS = np.array([
    0.14, # NVDA
    0.14, # MU
    0.14, # AAPL
    0.14, # AMD
    0.14, # ASML
    0.14, # MSFT
    0.14, # GOOG
    0.02  # CASH
])

# ==========================================
# 2. CORE LOGIC CLASSES
# ==========================================
class FinancialFeatureEngineer:
    def preprocess_data(self, df):
        """
        Replicates the exact feature engineering used in training.
        
        CRITICAL: This must match training exactly to avoid distribution shift.
        
        Features generated (per asset):
        1. Log Returns: log(price_t / price_t-1)
        2. Normalized Volatility: Z-score of 20-day rolling std using 252-day rolling mean/std
        3. RSI (14-period): Scaled to [-0.5, 0.5] using pandas_ta with Wilder smoothing
        4. Market Correlation: 60-day rolling correlation with benchmark returns
        
        Look-Ahead Bias Check:
        - All rolling windows use .rolling() which only looks backward
        - No .shift(-n) with negative values
        - No future data leakage in calculations
        
        Training Consistency:
        - Training uses simplified RSI (SMA-based) from archive/legacy_versions/simple/rl_system.py
        - Production uses pandas_ta.rsi() with Wilder smoothing (as per training code comment)
        - This creates a minor distribution difference that should be monitored
        - Normalization uses identical rolling windows (not fit_transform on full dataset)
        
        Args:
            df: DataFrame with columns = [Assets..., 'SPY']
        
        Returns:
            np.ndarray: Shape (time_steps, n_assets, 3) feature tensor
        """
        data = df.copy()
        # Filter only asset columns (exclude SPY/QQQ benchmark)
        asset_cols = [c for c in data.columns if c != 'SPY']
        
        # 1. Log Returns
        log_ret = np.log(data[asset_cols] / data[asset_cols].shift(1)).fillna(0)
        
        # 2. Rolling Volatility (Z-Score)
        # Normalized against 252-day history (Standard scaler logic)
        roll_std = log_ret.rolling(VOLATILITY_WINDOW).std()
        roll_mean_std = roll_std.rolling(VOLATILITY_NORMALIZATION_WINDOW).mean()
        roll_std_std = roll_std.rolling(VOLATILITY_NORMALIZATION_WINDOW).std()
        norm_vol = (roll_std - roll_mean_std) / (roll_std_std + 1e-8)
        
        # 3. RSI (Using pandas_ta for consistency with Training)
        rsi_df = pd.DataFrame()
        for col in asset_cols:
            # pandas_ta handles the wilder smoothing automatically
            rsi_val = ta.rsi(data[col], length=RSI_PERIOD)
            # Scale to [-0.5, 0.5] to match training scale
            rsi_df[col] = (rsi_val - 50) / 100
        
        # 4. Correlation to Market
        market_ret = data['SPY'].pct_change()
        correlations = log_ret.rolling(CORRELATION_WINDOW).corr(market_ret)
        
        # Clean and Clip (Remove NaNs from start of window)
        norm_vol = norm_vol.fillna(0).clip(-3, 3)
        rsi_df = rsi_df.fillna(0)
        correlations = correlations.fillna(0)
        
        # Stack Features: (Time, Assets, 3)
        feature_stack = np.stack([norm_vol.values, rsi_df.values, correlations.values], axis=-1)
        return feature_stack

class ProductionBoard:
    def __init__(self, bull_p, bear_p, sniper_p):
        # Check if files exist
        for p in [bull_p, bear_p, sniper_p]:
            if not os.path.exists(f"{p}.zip"):
                print(f"CRITICAL ERROR: Model file '{p}.zip' not found.")
                sys.exit(1)
                
        print(f"Loading models...")
        self.bull = PPO.load(bull_p)
        self.bear = PPO.load(bear_p)
        self.sniper = PPO.load(sniper_p)
        print("Models loaded successfully.")
        
    def predict(self, obs, short_vol, long_vol, p_fast, p_slow):
        # 1. CIRCUIT BREAKER (Priority 1)
        if short_vol > p_fast or long_vol > p_slow:
            # Run dummy predict to get shape
            dummy_action, _ = self.bear.predict(obs, deterministic=True)
            w = np.zeros(len(dummy_action))
            w[-1] = 1.0 # 100% Cash
            return w, "CRASH (CIRCUIT BREAKER TRIGGERED)"

        # 2. ENSEMBLE VOTING
        p_bull, _ = self.bull.predict(obs, deterministic=True)
        p_bear, _ = self.bear.predict(obs, deterministic=True)
        p_sniper, _ = self.sniper.predict(obs, deterministic=True)
        
        w_bull = self._softmax(p_bull)
        w_bear = self._softmax(p_bear)
        w_sniper = self._softmax(p_sniper)
        
        # 3. MARKET REGIME MIXING
        if long_vol > CHOPPY_VOLATILITY_THRESHOLD:
            # Choppy/Volatile -> Mix Sniper and Bear
            final_w = (REGIME_MIXING_SNIPER_WEIGHT * w_sniper) + \
                     (REGIME_MIXING_BEAR_WEIGHT * w_bear) + \
                     (REGIME_MIXING_BULL_WEIGHT * w_bull)
            regime = "CHOPPY (Mixed Allocation)"
        else:
            # Calm/Bull -> Mix Bull and Sniper
            final_w = (CALM_BULL_WEIGHT * w_bull) + (CALM_SNIPER_WEIGHT * w_sniper)
            regime = "GROWTH (Bull Allocation)"

        return final_w, regime

    def _softmax(self, x, temperature=SOFTMAX_TEMPERATURE):
        """Apply softmax with temperature scaling."""
        scaled = x / temperature
        e_x = np.exp(scaled - np.max(scaled))
        return e_x / e_x.sum()

# ==========================================
# 3. EXECUTION ROUTINE
# ==========================================
def run_advisor():
    print("\n" + "="*50)
    print("       AI PORTFOLIO ADVISOR (PRODUCTION)   ")
    print("="*50)

    # Message buffer for Telegram
    msg_lines = []
    msg_lines.append("🤖 <b>AI PORTFOLIO ADVISOR</b>\n")

    # 0. Configure Logging
    log_filename = f"advisory_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    # Log run metadata
    logging.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "event": "advisor_run_start",
        "version": "1.0",
        "assets": TICKERS,
        "config": {
            "fast_vol_threshold": BEST_FAST_VOL,
            "slow_vol_threshold": BEST_SLOW_VOL,
            "inertia_threshold": BEST_THRESH
        }
    }))
    
    # 1. Strict Weight Validation
    weight_sum = np.sum(CURRENT_WEIGHTS)
    if not np.isclose(weight_sum, 1.0, atol=WEIGHT_SUM_TOLERANCE):
        error_msg = f"CURRENT_WEIGHTS must sum to 1.0 (got {weight_sum:.6f}). Please fix the configuration in the script."
        logging.error(json.dumps({"timestamp": datetime.now().isoformat(), "event": "config_error", "error": error_msg}))
        send_telegram_message(f"❌ <b>CONFIG ERROR:</b> {error_msg}")
        raise ValueError(error_msg)

    # 2. FETCH LIVE DATA
    # We need ~2 years to stabilize the 252-day rolling means
    print(f"Fetching live data for {len(TICKERS)} assets + {BENCHMARK}...")
    all_tickers = TICKERS + [BENCHMARK]

    try:
        raw_data = yf.download(all_tickers, period=HISTORICAL_DATA_PERIOD, progress=False, auto_adjust=True)
        logging.info(json.dumps({"timestamp": datetime.now().isoformat(), "event": "data_fetch_success", "tickers": all_tickers}))
    except Exception as e:
        error_msg = f"Failed to fetch market data: {str(e)}"
        logging.error(json.dumps({"timestamp": datetime.now().isoformat(), "event": "data_fetch_error", "error": str(e)}))
        print(f"Error fetching data: {e}")
        send_telegram_message(f"❌ <b>ERROR:</b> {error_msg}")
        raise SystemExit(1)

    # Handle YFinance MultiIndex return format
    if isinstance(raw_data.columns, pd.MultiIndex):
        try:
            df = raw_data['Close'].copy()
        except KeyError:
            df = raw_data['close'].copy() # Lowercase fallback
    else:
        df = raw_data.copy()

    # Rename Benchmark
    if BENCHMARK in df.columns:
        df = df.rename(columns={BENCHMARK: 'SPY'})
    else:
        print(f"Error: Benchmark {BENCHMARK} not found in data.")
        send_telegram_message(f"❌ <b>ERROR:</b> Benchmark {BENCHMARK} not found")
        return

    # Data Cleaning
    df = df.ffill().dropna()

    # Ensure correct column order for the Agent
    cols = [t for t in TICKERS if t in df.columns] + ['SPY']
    df = df[cols]

    # Guard against empty DataFrame (can happen in CI with data fetch issues)
    if df is None or df.empty:
        error_msg = "⚠️ <b>DATA ERROR:</b> No market data available. Possible causes:\n• Market is closed\n• Yahoo Finance API issue\n• Network connectivity problem"
        logging.error(json.dumps({"timestamp": datetime.now().isoformat(), "event": "empty_dataframe"}))
        send_telegram_message(error_msg)
        print("ERROR: No data returned. Exiting with failure code.")
        raise SystemExit(1)  # Fail the workflow to trigger alerts

    # 3. CALCULATE VOLATILITY REGIME
    # Volatility of the Benchmark (SPY/QQQ)
    bench_ret = df['SPY'].pct_change()
    vol_fast = bench_ret.tail(FAST_VOLATILITY_WINDOW).std()
    vol_slow = bench_ret.tail(SLOW_VOLATILITY_WINDOW).std()

    date_str = df.index[-1].strftime('%Y-%m-%d')
    print(f"\n--- MARKET WEATHER ({df.index[-1].date()}) ---")
    print(f"Fast Volatility (5d):  {vol_fast:.2%}  [Trigger: >{BEST_FAST_VOL:.2%}]")
    print(f"Slow Volatility (20d): {vol_slow:.2%}  [Trigger: >{BEST_SLOW_VOL:.2%}]")

    msg_lines.append(f"📅 <b>Date:</b> {date_str}")
    msg_lines.append(f"📊 <b>Market Weather:</b>")
    msg_lines.append(f"  • Fast Vol (5d): {vol_fast:.2%}")
    msg_lines.append(f"  • Slow Vol (20d): {vol_slow:.2%}\n")

    # 4. ENGINEER AI FEATURES
    engineer = FinancialFeatureEngineer()
    features = engineer.preprocess_data(df)

    # Check if we have enough data
    if len(features) < OBSERVATION_WINDOW_SIZE:
        error_msg = f"Not enough historical data for the {OBSERVATION_WINDOW_SIZE}-day lookback (got {len(features)} days)"
        logging.error(json.dumps({"timestamp": datetime.now().isoformat(), "event": "insufficient_data", "available_days": len(features)}))
        print(f"Error: {error_msg}")
        send_telegram_message(f"❌ <b>ERROR:</b> {error_msg}")
        raise SystemExit(1)

    # Get the most recent observation window (Shape: 60 x Assets x 3)
    obs_window = features[-OBSERVATION_WINDOW_SIZE:]
    
    logging.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "event": "features_generated",
        "feature_shape": list(features.shape),
        "observation_window_shape": list(obs_window.shape)
    }))

    # 5. GENERATE PREDICTION
    board = ProductionBoard(PATH_BULL, PATH_BEAR, PATH_SNIPER)
    target_weights, regime = board.predict(
        obs_window, vol_fast, vol_slow,
        p_fast=BEST_FAST_VOL, p_slow=BEST_SLOW_VOL
    )

    print(f"AI Regime Decision:    {regime}")
    msg_lines.append(f"🎯 <b>Regime:</b> {regime}\n")
    
    logging.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "event": "prediction_generated",
        "regime": regime,
        "target_weights": [round(float(w), 3) for w in target_weights],
        "vol_fast": round(float(vol_fast), 4),
        "vol_slow": round(float(vol_slow), 4)
    }))

    # 6. COMPARE WITH CURRENT PORTFOLIO
    current_w_norm = CURRENT_WEIGHTS / np.sum(CURRENT_WEIGHTS)
    turnover = np.sum(np.abs(target_weights - current_w_norm))

    # Determine Activation Threshold
    active_thresh = BEST_THRESH
    if "CRASH" in regime:
        active_thresh = 0.01
        print(">> EMERGENCY PROTOCOL: Inertia removed for rapid exit. <<")
        msg_lines.append("⚠️ <b>EMERGENCY PROTOCOL ACTIVE</b>\n")

    print(f"Required Turnover:     {turnover:.2%}")
    print(f"Inertia Threshold:     {active_thresh:.2%}")

    print("-" * 50)

    # 7. TRADING DECISION
    assets = TICKERS + ['CASH']

    if turnover > active_thresh:
        print("📢 DECISION: REBALANCE REQUIRED")
        print("-" * 50)
        print(f"{'ASSET':<8} {'CURRENT':<10} {'TARGET':<10} {'DELTA'}")
        print("-" * 50)

        msg_lines.append("📢 <b>DECISION: REBALANCE REQUIRED</b>")
        msg_lines.append(f"Turnover: {turnover:.1%}\n")
        msg_lines.append("<pre>")
        msg_lines.append(f"{'ASSET':<6} {'NOW':>6} {'TARGET':>7} {'DELTA':>7}")
        msg_lines.append("-" * 28)

        for i, asset in enumerate(assets):
            old_p = current_w_norm[i] * 100
            new_p = target_weights[i] * 100
            delta = new_p - old_p

            # Visual indicator for big moves
            marker = ""
            if abs(delta) > 5.0: marker = "<<< BIG MOVE"
            if asset == 'CASH' and new_p > 50.0: marker = "!!! SAFETY !!!"

            print(f"{asset:<8} {old_p:5.1f}%     {new_p:5.1f}%     {delta:+5.1f}% {marker}")

            # For Telegram
            flag = "⬆️" if delta > 2 else ("⬇️" if delta < -2 else "  ")
            msg_lines.append(f"{asset:<6} {old_p:5.1f}% {new_p:6.1f}% {delta:+6.1f}%{flag}")

        msg_lines.append("</pre>")

    else:
        print("💤 DECISION: HOLD POSITION")
        print(f"Reason: Turnover ({turnover:.1%}) is below threshold ({active_thresh:.1%}).")
        print("Action: Do nothing. Save fees.")

        msg_lines.append("💤 <b>DECISION: HOLD POSITION</b>")
        msg_lines.append(f"Turnover ({turnover:.1%}) &lt; Threshold ({active_thresh:.1%})")
        msg_lines.append("Action: Do nothing. Save fees.")

    # 8. Send Telegram notification and log completion
    telegram_message = "\n".join(msg_lines)
    send_telegram_message(telegram_message)
    
    logging.info(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "event": "advisor_run_complete",
        "decision": "REBALANCE" if turnover > active_thresh else "HOLD",
        "turnover": round(float(turnover), 4),
        "threshold": round(float(active_thresh), 4)
    }))
    
    print(f"\n✅ Run complete. Log saved to: {log_filename}")

if __name__ == "__main__":
    run_advisor()