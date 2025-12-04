
import os
import shutil
import pandas as pd
import numpy as np
from training import train_super_agent_sac, train_meta_agent
from stable_baselines3 import PPO, SAC

def test_system_integration():
    print("="*70)
    print("STARTING SYSTEM INTEGRATION TEST")
    print("="*70)

    # 1. Load Data
    print("\n1. Loading Data...")
    try:
        price_data = pd.read_csv('data/price_data.csv', index_col=0, parse_dates=True)
        technical_features = pd.read_csv('data/technical_features.csv', index_col=0, parse_dates=True)
        sentiment_features = pd.read_csv('data/sentiment_features.csv', index_col=0, parse_dates=True)
        # Create dummy regime indicators if not present or just for testing
        if os.path.exists('data/regime_indicators.csv'):
             regime_indicators = pd.read_csv('data/regime_indicators.csv', index_col=0, parse_dates=True)
        else:
             print("   Creating dummy regime indicators...")
             regime_indicators = pd.DataFrame(np.random.randint(0, 2, size=(len(price_data), 2)), 
                                              index=price_data.index, columns=['regime_1', 'regime_2'])

        print(f"   Price data: {price_data.shape}")
        print(f"   Technical: {technical_features.shape}")
        print(f"   Sentiment: {sentiment_features.shape}")
    except Exception as e:
        print(f"   FAILED to load data: {e}")
        return

    # 2. Create Train/Val Split (Small subset for speed)
    print("\n2. Creating Data Splits...")
    # Use last 100 points for quick test
    subset_size = 100
    train_size = 80
    
    train_slice = slice(0, train_size)
    val_slice = slice(train_size, subset_size)
    
    train_data = (
        price_data.iloc[train_slice],
        technical_features.iloc[train_slice],
        sentiment_features.iloc[train_slice],
        regime_indicators.iloc[train_slice]
    )
    
    val_data = (
        price_data.iloc[val_slice],
        technical_features.iloc[val_slice],
        sentiment_features.iloc[val_slice],
        regime_indicators.iloc[val_slice]
    )
    print(f"   Train size: {train_size}, Val size: {subset_size - train_size}")

    # 3. Define Minimal Config
    print("\n3. Defining Test Configuration...")
    # Override configs for speed
    test_config = {
        'transaction_cost': 0.001,
        'max_position': 0.5,
        'max_turnover': 0.5,
        'constraint_penalty': 10.0,
        'alpha_returns': 1.0,
        'alpha_mdd': 0.5,
        'alpha_vol': 0.5,
        'alpha_concentration': 0.1,
        'seed': 42,
        
        # Super Agent (SAC)
        'super_timesteps': 1000,  # Very short training
        'super_learning_rate': 0.0003,
        'super_buffer_size': 10000,
        'super_batch_size': 32,
        'super_tau': 0.005,
        'super_gamma': 0.99,
        'super_ent_coef': 'auto',
        'super_network': [64, 64],
        'super_eval_freq': 500,
        'super_patience': 5,
        'super_min_delta': 0.01,
        
        # Meta Agent (PPO)
        'meta_timesteps': 1000, # Very short training
        'meta_learning_rate': 0.0003,
        'meta_n_steps': 128,
        'meta_batch_size': 32,
        'meta_n_epochs': 3,
        'meta_gamma': 0.99,
        'meta_ent_coef': 0.01,
        'meta_max_grad_norm': 0.5,
        'meta_gae_lambda': 0.95,
        'meta_network': [64, 64],
        'meta_eval_freq': 500,
        'meta_patience': 5,
        'meta_min_delta': 0.01
    }

    # 4. Check Base Models
    print("\n4. Checking Base Models...")
    required_models = [
        'models/best_technical_PPO.zip',
        'models/best_technical_SAC.zip',
        'models/best_sentiment_PPO.zip',
        'models/best_sentiment_SAC.zip'
    ]
    missing = [m for m in required_models if not os.path.exists(m)]
    if missing:
        print(f"   FAILED: Missing base models: {missing}")
        print("   Please ensure base models are in 'models/' directory.")
        # Create dummy models for testing ONLY if they don't exist? 
        # No, better to fail and ask user to provide them or run base training.
        # But for this test to pass in this environment, we might need to mock them if they aren't there.
        # Let's assume they are there based on previous context.
        return

    # 5. Run Super Agent Training
    print("\n5. Running Super Agent Training...")
    try:
        super_model, super_sharpe = train_super_agent_sac(train_data, val_data, test_config)
        print(f"   ✓ Super Agent Trained. Val Sharpe: {super_sharpe:.4f}")
        
        # Save dummy best model for meta agent to load if needed, 
        # though train_meta_agent takes the model object directly.
    except Exception as e:
        print(f"   FAILED Super Agent Training: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Run Meta Agent Training
    print("\n6. Running Meta Agent Training...")
    try:
        meta_model, meta_sharpe = train_meta_agent(train_data, val_data, super_model, test_config)
        print(f"   ✓ Meta Agent Trained. Val Sharpe: {meta_sharpe:.4f}")
    except Exception as e:
        print(f"   FAILED Meta Agent Training: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    test_system_integration()
