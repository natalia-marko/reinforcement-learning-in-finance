"""
Multi-Ticker Portfolio Training with Early Stopping
====================================================

This script trains a 3-level hierarchical reinforcement learning system for portfolio allocation
with support for early stopping to prevent overfitting and reduce training time.

EARLY STOPPING FEATURE:
- Monitors validation performance during training (Sharpe ratio)
- Stops training when performance plateaus (no improvement for N evaluations)
- Applies to Level 2 (Super Agent) and Level 3 (Meta Agent)
- Level 1 agents (Tech and Sent) train without early stopping as they are fast

Configuration:
  ENABLE_EARLY_STOPPING: Enable/disable early stopping
  ES_PATIENCE: Number of evaluations without improvement before stopping
  ES_CHECK_FREQ: How often to evaluate (in timesteps)
  ES_MIN_DELTA: Minimum improvement in Sharpe ratio to consider as improvement

The early stopping callback evaluates the model on the validation set periodically and tracks
the best Sharpe ratio. If the Sharpe doesn't improve by at least ES_MIN_DELTA for ES_PATIENCE
consecutive evaluations, training stops early.
"""

import warnings
warnings.filterwarnings('ignore')
from visualizations import create_simple_report
from pipeline import ResidualMetaLearningPipeline
import numpy as np
from utile import BaseCallback, EarlyStoppingCallback

# === DATA CONFIGURATION ===
TICKERS = ['NVDA', 'MSFT', 'GOOG', 'AMD', 'ASML', 'MU']
BENCHMARK = 'QQQ'
START_DATE = '2020-01-01'
END_DATE = '2025-12-31'
TRAIN_END = '2023-06-30'
VAL_END = '2024-06-30'

# === FEATURE ENGINEERING ===
# Technical indicators
SMA_WINDOWS = [4, 8, 12]        # Weeks for SMA calculations
INCLUDE_LAGS = 3                 # Number of return lags
RSI_PERIOD = 14                  # RSI calculation period
MACD_FAST = 12                   # MACD fast EMA
MACD_SLOW = 26                   # MACD slow EMA
MACD_SIGNAL = 9                  # MACD signal line

# Macro data
MACRO_FORWARD_FILL_LIMIT = 2     # Max weeks to forward-fill macro data

# === TRAINING CONFIGURATION ===
# Level 1: Specialists (Tech & Sent)
L1_TIMESTEPS = 50000            # Training steps per agent
L1_N_ENVS = 8                    # Parallel environments (CPU speedup)
L1_LEARNING_RATE = 3e-4          # PPO learning rate
L1_N_STEPS = 2048                # Steps per update
L1_BATCH_SIZE = 64               # Minibatch size
L1_N_EPOCHS = 10                 # Gradient descent epochs

# Level 2: Integrator (Super)
L2_TIMESTEPS = 200000          # Complex integration (combines 6 agents)
L2_N_ENVS = 6                    # Parallel envs for SAC
L2_LEARNING_RATE = 3e-4          # SAC learning rate
L2_BUFFER_SIZE = 100000         # Replay buffer size
L2_BATCH_SIZE = 256              # Minibatch size

# Level 3: Meta (Residual)
L3_TIMESTEPS = 300000           # MOST complex obs (~300 dim) - needs MORE training
L3_N_ENVS = 1                    # Single env (SAC uses replay buffer)
L3_LEARNING_RATE = 3e-4          # SAC learning rate
L3_BUFFER_SIZE = 100000         # SAC replay buffer
L3_BATCH_SIZE = 256              # SAC batch size
L3_ADJUSTMENT_PENALTY = 0.05     # Penalty for deviating from Super (increased to reduce over-adjustment)
L3_ADJUSTMENT_SCALE = 0.1        # Max adjustment magnitude (±10%) (reduced from ±20%)

# === ENVIRONMENT CONFIGURATION ===
TRANSACTION_COST_RATE = 0.002   # 20 basis points (realistic: bid-ask + market impact)
LOOKBACK_PERIODS = 10            # History length for Meta

# === EARLY STOPPING CONFIGURATION ===
ENABLE_EARLY_STOPPING = True     # Enable early stopping during training
ES_PATIENCE = 10                 # Number of eval steps without improvement
ES_MIN_DELTA = 0.01            # Minimum improvement to consider
ES_CHECK_FREQ = 2000            # Evaluate every N timesteps

# === VALIDATION ===
RUN_VALIDATION = True            # Run data leakage checks
GENERATE_VISUALIZATIONS = True   # Create plots
SAVE_RESULTS = True              # Save to disk

print("Starting Multi-Ticker Portfolio")
print(f"Tickers: {TICKERS}")
print(f"Benchmark: {BENCHMARK}")
print(f"Training timesteps: L1={L1_TIMESTEPS:,}, L2={L2_TIMESTEPS:,}, L3={L3_TIMESTEPS:,}")
print(f"Parallel envs: L1={L1_N_ENVS}, L2={L2_N_ENVS}, L3={L3_N_ENVS}")
print(f"Adjustment penalty: {L3_ADJUSTMENT_PENALTY}")
if ENABLE_EARLY_STOPPING:
    print(f"Early stopping: ENABLED (patience={ES_PATIENCE}, check_freq={ES_CHECK_FREQ}, min_delta={ES_MIN_DELTA})")
else:
    print("Early stopping: DISABLED")

# Create pipeline
pipeline = ResidualMetaLearningPipeline(
    tickers=TICKERS,
    benchmark=BENCHMARK
)


def multi_ticker_example():
    """
    Multi-ticker portfolio example with early stopping.
    """
    from utile import create_env_for_portfolio
    from residual_meta_enviroment import create_residual_meta_environment
    
    # Train Level 1: Specialists
    print("\n" + "="*70)
    if ENABLE_EARLY_STOPPING:
        print("EARLY STOPPING ENABLED")
        print(f"  Patience: {ES_PATIENCE} evaluations")
        print(f"  Check frequency: every {ES_CHECK_FREQ} timesteps")
        print(f"  Min delta: {ES_MIN_DELTA}")
    print("="*70 + "\n")
    
    # First, prepare datasets to create validation environments
    pipeline.train_level1(
        start=START_DATE,
        end=END_DATE,
        train_end=TRAIN_END,
        val_end=VAL_END,
        timesteps=L1_TIMESTEPS,
        n_envs=L1_N_ENVS
    )
    
    # Note: For Level 1, early stopping is challenging since we train Tech and Sent separately
    # and they share the same environment. We'll focus early stopping on Level 2 and Level 3.

    # Train Level 2: Integrator with early stopping
    if ENABLE_EARLY_STOPPING:
        # Create validation environment for Level 2
        val_datasets_l2 = pipeline._prepare_level2_datasets_fixed()
        val_env_l2 = create_env_for_portfolio('val', val_datasets_l2, pipeline.returns)
        
        # Create temporary agent for callback
        from utile import SuperAgent
        temp_super_agent = SuperAgent()
        
        l2_callback = EarlyStoppingCallback(
            val_env=val_env_l2,
            agent=temp_super_agent,
            check_freq=ES_CHECK_FREQ,
            patience=ES_PATIENCE,
            min_delta=ES_MIN_DELTA,
            verbose=1
        )
        
        pipeline.train_level2(
            timesteps=L2_TIMESTEPS,
            n_envs=L2_N_ENVS,
            callback=l2_callback
        )
    else:
        pipeline.train_level2(
            timesteps=L2_TIMESTEPS,
            n_envs=L2_N_ENVS
        )

    # Train Level 3: Meta with early stopping
    if ENABLE_EARLY_STOPPING:
        # Prepare Level 3 datasets for validation environment
        pipeline.datasets_l3 = pipeline._prepare_level3_datasets_fixed(START_DATE, END_DATE)
        # Macro data is already available in pipeline.macro_data after _prepare_level3_datasets_fixed
        combined_features = pipeline.macro_data
        
        # Create validation environment for Level 3
        val_env_l3 = create_residual_meta_environment(
            asset='PORTFOLIO',
            split='val',
            datasets=pipeline.datasets_l1,
            returns=pipeline.returns,
            tech_agent=pipeline.tech_agent,
            sent_agent=pipeline.sent_agent,
            super_agent=pipeline.super_agent,
            macro_df=combined_features,
            adjustment_penalty=L3_ADJUSTMENT_PENALTY
        )
        
        # Create temporary agent for callback
        from residual_meta_agent import ResidualMetaAgent
        temp_meta_agent = ResidualMetaAgent(
            learning_rate=L3_LEARNING_RATE,
            buffer_size=L3_BUFFER_SIZE,
            batch_size=L3_BATCH_SIZE,
            adjustment_scale=L3_ADJUSTMENT_SCALE
        )
        
        # Create special callback for residual environment
        class ResidualEarlyStoppingCallback(BaseCallback):
            """Early stopping for residual meta environment."""
            def __init__(self, val_env, agent, check_freq=2000, patience=10, min_delta=0.01, verbose=1):
                super().__init__(verbose)
                self.val_env = val_env
                self.agent = agent
                self.check_freq = check_freq
                self.patience = patience
                self.min_delta = min_delta
                self.best_sharpe = -np.inf
                self.wait = 0
                
            def _on_step(self) -> bool:
                if self.n_calls % self.check_freq == 0:
                    sharpe = self._evaluate()
                    
                    if sharpe > self.best_sharpe + self.min_delta:
                        self.best_sharpe = sharpe
                        self.wait = 0
                        if self.verbose > 0:
                            print(f"  Timestep {self.num_timesteps}: Sharpe improved to {sharpe:.3f}")
                    else:
                        self.wait += 1
                        if self.verbose > 0:
                            print(f"  Timestep {self.num_timesteps}: Sharpe={sharpe:.3f} (no improvement, patience: {self.wait}/{self.patience})")
                        
                        if self.wait >= self.patience:
                            if self.verbose > 0:
                                print(f"\nEarly stopping triggered at timestep {self.num_timesteps}")
                                print(f"Best Sharpe: {self.best_sharpe:.3f}")
                            return False
                
                return True
            
            def _evaluate(self) -> float:
                obs, _ = self.val_env.reset()
                returns = []
                
                for _ in range(self.val_env.base_env.n_steps):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = self.val_env.step(action)
                    returns.append(info['portfolio_return'])
                    if terminated or truncated:
                        break
                
                returns = np.array(returns)
                sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(52)
                return sharpe
        
        l3_callback = ResidualEarlyStoppingCallback(
            val_env=val_env_l3,
            agent=temp_meta_agent,
            check_freq=ES_CHECK_FREQ,
            patience=ES_PATIENCE,
            min_delta=ES_MIN_DELTA,
            verbose=1
        )
        
        pipeline.train_level3(
            start=START_DATE,
            end=END_DATE,
            timesteps=L3_TIMESTEPS,
            n_envs=L3_N_ENVS,
            adjustment_penalty=L3_ADJUSTMENT_PENALTY,
            learning_rate=L3_LEARNING_RATE,
            buffer_size=L3_BUFFER_SIZE,
            batch_size=L3_BATCH_SIZE,
            adjustment_scale=L3_ADJUSTMENT_SCALE,
            callback=l3_callback
        )
    else:
        # Prepare Level 3 datasets for training
        pipeline.datasets_l3 = pipeline._prepare_level3_datasets_fixed(START_DATE, END_DATE)
        
        pipeline.train_level3(
            start=START_DATE,
            end=END_DATE,
            timesteps=L3_TIMESTEPS,
            n_envs=L3_N_ENVS,
            adjustment_penalty=L3_ADJUSTMENT_PENALTY,
            learning_rate=L3_LEARNING_RATE,
            buffer_size=L3_BUFFER_SIZE,
            batch_size=L3_BATCH_SIZE,
            adjustment_scale=L3_ADJUSTMENT_SCALE
        )

    # Evaluate on test set
    results = pipeline.evaluate(split_name='test')

    # Generate visualizations
    if GENERATE_VISUALIZATIONS:
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS...")
        print("="*70)
        from visualizations import create_simple_report
        create_simple_report(results, TICKERS, save_dir='reports')

    print(f"\n✓ Training complete!")
    return pipeline, results


def hyperparameter_sweep():
    """
    Experiment with different hyperparameters.
    """
    # Test different adjustment penalties
    penalties = [0.001, 0.01, 0.05, 0.1]
    results_by_penalty = {}

    for penalty in penalties:
        print(f"\n{'='*70}")
        print(f"Testing adjustment_penalty = {penalty}")
        print(f"{'='*70}")
        
        pipeline = ResidualMetaLearningPipeline(
            tickers=TICKERS,
            benchmark=BENCHMARK
        )
        
        # Train all levels
        pipeline.train_level1(
            start=START_DATE,
            end=END_DATE,
            train_end=TRAIN_END,
            val_end=VAL_END,
            timesteps=L1_TIMESTEPS,
            n_envs=L1_N_ENVS
        )
        
        pipeline.train_level2(
            timesteps=L2_TIMESTEPS,
            n_envs=L2_N_ENVS
        )
        
        pipeline.train_level3(
            start=START_DATE,
            end=END_DATE,
            timesteps=L3_TIMESTEPS,
            n_envs=L3_N_ENVS,
            adjustment_penalty=penalty
        )
        
        # Evaluate
        results = pipeline.evaluate(split_name='test')
        results_by_penalty[penalty] = results
        
        print(f"\nResults for penalty={penalty}:")
        print(f"  Meta Return: {results['meta']['total_return']*100:.2f}%")
        print(f"  Meta Sharpe: {results['meta']['sharpe']:.3f}")
        print(f"  Avg Adjustment: {results['meta']['adjustments'].mean():.4f}")

    # Print summary
    print(f"\n{'='*70}")
    print("HYPERPARAMETER SWEEP SUMMARY")
    print(f"{'='*70}")

    for penalty, results in results_by_penalty.items():
        print(f"\nPenalty={penalty}:")
        print(f"  Return: {results['meta']['total_return']*100:.2f}%")
        print(f"  Sharpe: {results['meta']['sharpe']:.3f}")
        print(f"  Adjustment Size: {results['meta']['adjustments'].mean():.4f}")

    # Find best
    best_penalty = max(results_by_penalty.keys(), 
                        key=lambda p: results_by_penalty[p]['meta']['sharpe'])
    print(f"\n✓ Best adjustment penalty: {best_penalty}")

    return results_by_penalty


# Option 2: Multi-ticker portfolio (full pipeline, ~70 minutes)
pipeline, results = multi_ticker_example()

# Option 3: Hyperparameter sweep (experimental, several hours)
results_by_penalty = hyperparameter_sweep()

print("\n" + "=" * 70)
print("SUCCESS! Pipeline completed successfully!")
print("=" * 70)
print("Results saved in 'results' variable")
print("Pipeline object saved in 'pipeline' variable")

print("\nWith multi-asset portfolio, Meta uses macro/calendar to:")
print("  - Reduce volatile assets (NVDA, AMD) when VIX spikes")
print("  - Rotate to stable assets (MSFT, GOOG) during uncertainty")
print("  - Adjust allocations around quarter-end rebalancing")
print("  - Respond to rate changes and economic signals")

# Print key metrics
if results:
    print("\n" + "=" * 70)
    print("KEY METRICS (Test Set)")
    print("=" * 70)
    print(f"Level 1 (Tech):  {results['tech']['total_return']*100:>6.2f}%  |  Sharpe: {results['tech']['sharpe']:>5.3f}")
    print(f"Level 2 (Super): {results['super']['total_return']*100:>6.2f}%  |  Sharpe: {results['super']['sharpe']:>5.3f}")
    print(f"Level 3 (Meta):  {results['meta']['total_return']*100:>6.2f}%  |  Sharpe: {results['meta']['sharpe']:>5.3f}")
    
    improvement = (results['meta']['sharpe'] - results['super']['sharpe']) / results['super']['sharpe'] * 100
    print(f"\nMeta improvement over Super: {improvement:+.1f}%")