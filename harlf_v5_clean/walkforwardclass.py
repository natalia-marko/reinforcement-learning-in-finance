import pandas as pd
import numpy as np
import os
import json
from stable_baselines3 import PPO, SAC
from production_agent_wrapper_frozen import ProductionAgentWrapper
from super_agent_env import SuperAgentEnv
from meta_agent_enviroment import MetaAgentEnv
from utile import EarlyStoppingCallback
# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

class WalkForwardValidator:
    """Walk-forward validation to eliminate regime mismatch"""
    
    def __init__(self, n_windows: int = 3, min_train_months: int = 60):
        self.n_windows = n_windows
        self.min_train_months = min_train_months
    
    def create_windows(self, data_length: int, dates: pd.DatetimeIndex):
        """Create walk-forward validation windows"""
        
        test_size = int(data_length * 0.20)
        trainval_size = data_length - test_size
        val_size = int(trainval_size * 0.15)
        
        windows = []
        
        for i in range(self.n_windows):
            train_end = self.min_train_months + (i * val_size)
            val_start = train_end
            val_end = min(val_start + val_size, trainval_size)
            
            if val_end - val_start < 10:
                continue
            
            windows.append({
                'train_start': 0,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
                'train_dates': (dates[0], dates[train_end-1]),
                'val_dates': (dates[val_start], dates[val_end-1])
            })
        
        test_window = {
            'test_start': trainval_size,
            'test_end': data_length,
            'test_dates': (dates[trainval_size], dates[-1])
        }
        
        return windows, test_window
    
    def split_data_walk_forward(self, price_data, technical_features, 
                                sentiment_features, regime_indicators):
        """Split data using walk-forward validation"""
        
        print("\n" + "="*70)
        print("WALK-FORWARD VALIDATION SPLIT")
        print("="*70)
        
        data_length = len(price_data)
        dates = price_data.index
        
        windows, test_window = self.create_windows(data_length, dates)
        
        print(f"\nTotal data: {data_length} months")
        print(f"Date range: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
        print(f"\nCreated {len(windows)} walk-forward windows:")
        
        for i, window in enumerate(windows, 1):
            print(f"\n  Window {i}:")
            print(f"    Train: {window['train_dates'][0].strftime('%Y-%m')} to "
                  f"{window['train_dates'][1].strftime('%Y-%m')} "
                  f"({window['train_end']} months)")
            print(f"    Val:   {window['val_dates'][0].strftime('%Y-%m')} to "
                  f"{window['val_dates'][1].strftime('%Y-%m')} "
                  f"({window['val_end'] - window['val_start']} months)")
        
        print(f"\n  Final Test:")
        print(f"    Test:  {test_window['test_dates'][0].strftime('%Y-%m')} to "
              f"{test_window['test_dates'][1].strftime('%Y-%m')} "
              f"({test_window['test_end'] - test_window['test_start']} months)")
        
        data_splits = []
        
        for window in windows:
            train_slice = slice(window['train_start'], window['train_end'])
            val_slice = slice(window['val_start'], window['val_end'])
            
            data_splits.append({
                'train': (
                    price_data.iloc[train_slice],
                    technical_features.iloc[train_slice],
                    sentiment_features.iloc[train_slice],
                    regime_indicators.iloc[train_slice] if regime_indicators is not None else None
                ),
                'val': (
                    price_data.iloc[val_slice],
                    technical_features.iloc[val_slice],
                    sentiment_features.iloc[val_slice],
                    regime_indicators.iloc[val_slice] if regime_indicators is not None else None
                )
            })
        
        test_slice = slice(test_window['test_start'], test_window['test_end'])
        test_data = (
            price_data.iloc[test_slice],
            technical_features.iloc[test_slice],
            sentiment_features.iloc[test_slice],
            regime_indicators.iloc[test_slice] if regime_indicators is not None else None
        )
        
        print("\n" + "="*70)
        print("✓ Walk-forward splits created - no regime mismatch!")
        print("Walk-forward splits created - no regime mismatch!")
        
        return data_splits, test_data



# ============================================================================
# TRAINING FUNCTIONS (ENHANCED)
# ============================================================================

def train_super_agent_walk_forward_sac(data_splits, config):
    """
    ✨ Train Super agent with SAC + Enhanced observations
    Returns ensemble of all windows instead of just best
    """
    
    print("\n" + "="*70)
    print("TRAINING ENHANCED SUPER AGENT (SAC + FULL FEATURES)")
    print("="*70)
    
    n_windows = len(data_splits)
    all_val_sharpes = []
    models = []
    
    for window_idx, split in enumerate(data_splits, 1):
        print(f"\n{'='*70}")
        print(f"WINDOW {window_idx}/{n_windows}")
        print(f"{'='*70}")
        
        train_prices, train_tech, train_sent, train_regime = split['train']
        val_prices, val_tech, val_sent, val_regime = split['val']
        
        n_assets = len(train_prices.columns)
        
        # Load base agents
        tech_ppo_model = PPO.load('models/best_technical_PPO.zip')
        tech_sac_model = SAC.load('models/best_technical_SAC.zip')
        sent_ppo_model = PPO.load('models/best_sentiment_PPO.zip')
        sent_sac_model = SAC.load('models/best_sentiment_SAC.zip')
        
        base_agents_train = {
            'tech_PPO': ProductionAgentWrapper(tech_ppo_model, train_tech, n_assets, 'tech_PPO'),
            'tech_SAC': ProductionAgentWrapper(tech_sac_model, train_tech, n_assets, 'tech_SAC'),
            'sent_PPO': ProductionAgentWrapper(sent_ppo_model, train_sent, n_assets, 'sent_PPO'),
            'sent_SAC': ProductionAgentWrapper(sent_sac_model, train_sent, n_assets, 'sent_SAC'),
        }
        
        base_agents_val = {
            'tech_PPO': ProductionAgentWrapper(tech_ppo_model, val_tech, n_assets, 'tech_PPO'),
            'tech_SAC': ProductionAgentWrapper(tech_sac_model, val_tech, n_assets, 'tech_SAC'),
            'sent_PPO': ProductionAgentWrapper(sent_ppo_model, val_sent, n_assets, 'sent_PPO'),
            'sent_SAC': ProductionAgentWrapper(sent_sac_model, val_sent, n_assets, 'sent_SAC'),
        }
        
        # ✨ Create ENHANCED environments (with full features)
        super_train_env = SuperAgentEnv(
            train_prices, base_agents_train, train_tech, train_sent, **config
        )
        super_val_env = SuperAgentEnv(
            val_prices, base_agents_val, val_tech, val_sent, **config
        )
        
        # ✨ Train with SAC instead of PPO
        print(f"\n   Training with SAC algorithm...")
        model = SAC(
            'MlpPolicy',
            super_train_env,
            learning_rate=config['super_learning_rate'],
            buffer_size=config['super_buffer_size'],
            batch_size=config['super_batch_size'],
            tau=config['super_tau'],
            gamma=config['super_gamma'],
            ent_coef=config['super_ent_coef'],
            verbose=0,
            seed=config['seed'],
            policy_kwargs=dict(net_arch=config['super_network'])
        )
        
        callback = EarlyStoppingCallback(
            val_env=super_val_env,
            eval_freq=config['super_eval_freq'],
            patience=config['super_patience'],
            min_delta=config['super_min_delta'],
            min_steps=10000,
            save_path=f'models/super_agent_sac_window{window_idx}',
            verbose=1
        )
        
        print(f"\nTraining window {window_idx} for up to {config['super_timesteps']} steps...")
        model.learn(total_timesteps=config['super_timesteps'], callback=callback)
        
        if callback.best_model_path:
            model = SAC.load(callback.best_model_path)
        
        models.append(model)
        all_val_sharpes.append(callback.best_val_sharpe)
        
        print(f"\nWindow {window_idx} best val Sharpe: {callback.best_val_sharpe:.3f}")
    
    # ✨ Create ensemble instead of selecting best window
    ensemble = MultiWindowEnsemble(models, all_val_sharpes, temperature=1.0)
    
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll window Sharpes: {[f'{s:.3f}' for s in all_val_sharpes]}")
    print(f"Average val Sharpe: {np.mean(all_val_sharpes):.3f}")
    print(f"Std dev val Sharpe: {np.std(all_val_sharpes):.3f}")
    print(f"\n✨ Using multi-window ensemble (not just best window)")
    
    return ensemble, all_val_sharpes


def train_meta_agent_walk_forward(data_splits, super_ensemble, config):
    """Train Meta agent with walk-forward validation"""
    
    print("\n" + "="*70)
    print("TRAINING META AGENT (WALK-FORWARD)")
    print("="*70)
    
    n_windows = len(data_splits)
    all_val_sharpes = []
    models = []
    
    for window_idx, split in enumerate(data_splits, 1):
        print(f"\n{'='*70}")
        print(f"WINDOW {window_idx}/{n_windows}")
        print(f"{'='*70}")
        
        train_prices, train_tech, train_sent, train_regime = split['train']
        val_prices, val_tech, val_sent, val_regime = split['val']
        
        n_assets = len(train_prices.columns)
        
        # Load base agents
        tech_ppo_model = PPO.load('models/best_technical_PPO.zip')
        tech_sac_model = SAC.load('models/best_technical_SAC.zip')
        sent_ppo_model = PPO.load('models/best_sentiment_PPO.zip')
        sent_sac_model = SAC.load('models/best_sentiment_SAC.zip')
        
        base_agents_train = {
            'tech_PPO': ProductionAgentWrapper(tech_ppo_model, train_tech, n_assets, 'tech_PPO'),
            'tech_SAC': ProductionAgentWrapper(tech_sac_model, train_tech, n_assets, 'tech_SAC'),
            'sent_PPO': ProductionAgentWrapper(sent_ppo_model, train_sent, n_assets, 'sent_PPO'),
            'sent_SAC': ProductionAgentWrapper(sent_sac_model, train_sent, n_assets, 'sent_SAC'),
        }
        
        base_agents_val = {
            'tech_PPO': ProductionAgentWrapper(tech_ppo_model, val_tech, n_assets, 'tech_PPO'),
            'tech_SAC': ProductionAgentWrapper(tech_sac_model, val_tech, n_assets, 'tech_SAC'),
            'sent_PPO': ProductionAgentWrapper(sent_ppo_model, val_sent, n_assets, 'sent_PPO'),
            'sent_SAC': ProductionAgentWrapper(sent_sac_model, val_sent, n_assets, 'sent_SAC'),
        }
        
        # Create Super environments
        super_train_env = EnhancedSuperAgentEnv(
            train_prices, base_agents_train, train_tech, train_sent, **config
        )
        super_val_env = EnhancedSuperAgentEnv(
            val_prices, base_agents_val, val_tech, val_sent, **config
        )
        
        # Wrap Super ensemble
        super_train_wrap = ProductionAgentWrapper(
            super_ensemble, super_train_env, n_assets, 'super', freeze_weights=True
        )
        super_val_wrap = ProductionAgentWrapper(
            super_ensemble, super_val_env, n_assets, 'super', freeze_weights=True
        )
        
        # Create Meta environments
        meta_train_env = MetaAgentEnv(train_prices, super_train_wrap, train_regime, **config)
        meta_val_env = MetaAgentEnv(val_prices, super_val_wrap, val_regime, **config)
        
        # Train Meta with PPO
        model = PPO(
            'MlpPolicy',
            meta_train_env,
            learning_rate=config['meta_learning_rate'],
            n_steps=config['meta_n_steps'],
            batch_size=config['meta_batch_size'],
            n_epochs=config['meta_n_epochs'],
            gamma=config['meta_gamma'],
            ent_coef=config['meta_ent_coef'],
            max_grad_norm=config['meta_max_grad_norm'],
            gae_lambda=config['meta_gae_lambda'],
            verbose=0,
            seed=config['seed'],
            policy_kwargs=dict(net_arch=[dict(pi=config['meta_network'], 
                                              vf=config['meta_network'])])
        )
        
        callback = EarlyStoppingCallback(
            val_env=meta_val_env,
            eval_freq=config['meta_eval_freq'],
            patience=config['meta_patience'],
            min_delta=config['meta_min_delta'],
            min_steps=10000,
            save_path=f'models/meta_agent_window{window_idx}',
            verbose=1
        )
        
        print(f"\nTraining window {window_idx} for up to {config['meta_timesteps']} steps...")
        model.learn(total_timesteps=config['meta_timesteps'], callback=callback)
        
        if callback.best_model_path:
            model = PPO.load(callback.best_model_path, env=meta_train_env)
        
        models.append(model)
        all_val_sharpes.append(callback.best_val_sharpe)
        
        print(f"\nWindow {window_idx} best val Sharpe: {callback.best_val_sharpe:.3f}")
    
    # ✨ Create ensemble for Meta too
    meta_ensemble = MultiWindowEnsemble(models, all_val_sharpes, temperature=1.0)
    
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll window Sharpes: {[f'{s:.3f}' for s in all_val_sharpes]}")
    print(f"Average val Sharpe: {np.mean(all_val_sharpes):.3f}")
    print(f"Std dev val Sharpe: {np.std(all_val_sharpes):.3f}")
    print(f"\n✨ Using multi-window ensemble (not just best window)")
    
    return meta_ensemble, all_val_sharpes


# ============================================================================
# MULTI-WINDOW ENSEMBLE WRAPPER
# ============================================================================

class MultiWindowEnsemble:
    """
    Ensemble predictor that combines multiple window models
    Uses softmax-weighted average based on validation performance
    """
    
    def __init__(self, models, val_sharpes, temperature=1.0):
        self.models = models
        self.val_sharpes = np.array(val_sharpes)
        self.temperature = temperature
        
        # Compute softmax weights
        exp_sharpes = np.exp(self.val_sharpes / temperature)
        self.weights = exp_sharpes / exp_sharpes.sum()
        
        print(f"\n  Multi-Window Ensemble created:")
        for i, (sharpe, weight) in enumerate(zip(val_sharpes, self.weights), 1):
            print(f"      Window {i}: Sharpe={sharpe:.3f}, Weight={weight:.3f}")
    
    def predict(self, obs, deterministic=True):
        """Ensemble prediction: weighted average of all models"""
        
        predictions = []
        for model in self.models:
            action, _ = model.predict(obs, deterministic=deterministic)
            predictions.append(action)
        
        # Weighted average
        ensemble_action = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_action, None
    
    def save(self, path):
        """Save ensemble (saves individual models + metadata)"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save metadata
        metadata = {
            'val_sharpes': self.val_sharpes.tolist(),
            'weights': self.weights.tolist(),
            'temperature': self.temperature,
            'n_models': len(self.models)
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save individual models
        for i, model in enumerate(self.models):
            model.save(f"{path}_model{i}.zip")
        
        print(f"  Ensemble saved to {path}")

