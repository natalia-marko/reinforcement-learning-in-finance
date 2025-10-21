"""
SAC Meta Pipeline Orchestration - Dedicated orchestration for SAC meta agents.
This script trains a 3-level hierarchical reinforcement learning system with SAC at Level 3.
"""

from sac_pipeline import SACMetaLearningPipeline
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
L3_TIMESTEPS = 300000
L3_N_ENVS = 4
RUN_VALIDATION = True
GENERATE_VISUALIZATIONS = True

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SAC META PIPELINE")
    print("="*70)
    print(f"Tickers: {TICKERS}")
    print(f"Training: {START_DATE} to {TRAIN_END}")
    print(f"Validation: {TRAIN_END} to {VAL_END}")
    print(f"Test: {VAL_END} to {END_DATE}")
    print(f"L1 Timesteps: {L1_TIMESTEPS:,}")
    print(f"L2 Timesteps: {L2_TIMESTEPS:,}")
    print(f"L3 Timesteps: {L3_TIMESTEPS:,}")
    print(f"L3: SAC Meta Agent")
    print("="*70)
    
    # Initialize pipeline
    pipeline = SACMetaLearningPipeline(
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
        l3_timesteps=L3_TIMESTEPS,
        l3_n_envs=L3_N_ENVS
    )
    
    # Train Level 1 (includes data preparation)
    pipeline.train_level1()
    
    # Train Level 2
    pipeline.train_level2()
    
    # Train Level 3 with SAC
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
    
    print("\n✓ SAC pipeline completed successfully!")
