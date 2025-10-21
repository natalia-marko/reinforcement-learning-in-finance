"""
Linear Meta Pipeline - Simplified Architecture

Changes from original:
1. Early stopping for L1 (Tech/Sent) and L2 (Super) RL agents
2. Linear regression for L3 (Meta) instead of RL
3. Much faster training
4. More interpretable results

Architecture:
- L1 (Tech/Sent): PPO with early stopping
- L2 (Super): SAC with early stopping  
- L3 (Meta): Linear regression (Ridge + Softmax)
"""

from linear_pipeline import LinearMetaLearningPipeline
from linear_meta_agent import LinearMetaAgent
import numpy as np

# Configuration
TICKERS = ['NVDA', 'MU', 'AMD', 'ASML', 'MSFT', 'GOOG', 'AI']
BENCHMARK = 'QQQ'
START_DATE = '2015-01-01'
TRAIN_END = '2023-12-31'
VAL_END = '2024-06-30'
END_DATE = '2025-10-19'
L1_TIMESTEPS = 50000
L1_N_ENVS = 8
L2_TIMESTEPS = 200000
L2_N_ENVS = 6
L3_ALPHA = 1.0
RUN_VALIDATION = True
GENERATE_VISUALIZATIONS = True

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LINEAR META PIPELINE")
    print("="*70)
    print(f"Tickers: {TICKERS}")
    print(f"Training: {START_DATE} to {TRAIN_END}")
    print(f"Validation: {TRAIN_END} to {VAL_END}")
    print(f"Test: {VAL_END} to {END_DATE}")
    print(f"L1 Timesteps: {L1_TIMESTEPS:,}")
    print(f"L2 Timesteps: {L2_TIMESTEPS:,}")
    print(f"L3: Linear Regression (alpha={L3_ALPHA})")
    print("="*70)
    
    # Initialize pipeline
    pipeline = LinearMetaLearningPipeline(
        tickers=TICKERS,
        benchmark=BENCHMARK,
        start_date=START_DATE,
        end_date=END_DATE,
        train_end=TRAIN_END,
        val_end=VAL_END,
        l1_timesteps=L1_TIMESTEPS,
        l1_n_envs=L1_N_ENVS,
        l2_timesteps=L2_TIMESTEPS,
        l2_n_envs=L2_N_ENVS,
        l3_alpha=L3_ALPHA
    )
    
    # Train Level 1 (includes data preparation)
    pipeline.train_level1()
    
    # Train Level 2
    pipeline.train_level2()
    
    # Train Level 3 with Linear Regression
    pipeline.train_level3()
    
    # Evaluate all levels
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    
    results = pipeline.evaluate('test')
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"Volatility: {results['volatility']*100:.2f}%")
    print("="*70)
    
    # Generate visualizations
    if GENERATE_VISUALIZATIONS:
        from visualizations import create_simple_report
        create_simple_report(results, TICKERS)
    
    print("\n✓ Linear pipeline completed successfully!")