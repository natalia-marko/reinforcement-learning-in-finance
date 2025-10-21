"""
SAC Meta Learning Pipeline - Dedicated pipeline for SAC meta agents.
This pipeline is optimized for SAC-based meta learning at Level 3.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from utile import (
    create_env_for_portfolio, 
    get_macro_and_calendar_features,
    TechnicalAgent, 
    SentimentAgent, 
    SuperAgent,
    EarlyStoppingCallback
)
from residual_meta_agent import ResidualMetaAgent


class SACMetaLearningPipeline:
    """Pipeline optimized for SAC meta learning approach."""
    
    def __init__(
        self,
        tickers: List[str],
        benchmark: str,
        start_date: str,
        end_date: str,
        train_end: str,
        val_end: str,
        l1_timesteps: int = 50000,
        l1_n_envs: int = 8,
        l2_timesteps: int = 200000,
        l2_n_envs: int = 6,
        l3_timesteps: int = 100000,
        l3_n_envs: int = 4
    ):
        self.tickers = tickers
        self.benchmark = benchmark
        self.start_date = start_date
        self.end_date = end_date
        self.train_end = train_end
        self.val_end = val_end
        self.l1_timesteps = l1_timesteps
        self.l1_n_envs = l1_n_envs
        self.l2_timesteps = l2_timesteps
        self.l2_n_envs = l2_n_envs
        self.l3_timesteps = l3_timesteps
        self.l3_n_envs = l3_n_envs
        
        # Initialize agents
        self.tech_agent = TechnicalAgent()
        self.sent_agent = SentimentAgent()
        self.super_agent = SuperAgent()
        self.meta_agent = ResidualMetaAgent()
        
        # Data storage
        self.datasets_l1 = None
        self.datasets_l2 = None
        self.datasets_l3 = None
        self.returns = None
        self.macro_data = None
        self.calendar_data = None
        
    def prepare_data(self):
        """Prepare all datasets for the pipeline."""
        print("Preparing datasets with enhanced technical indicators...")
        
        # Import the data preparation function
        from utile import prepare_datasets
        
        self.datasets_l1, norm_stats, self.returns = prepare_datasets(
            tickers=self.tickers,
            benchmark=self.benchmark,
            start=self.start_date,
            end=self.end_date,
            train_end=self.train_end,
            val_end=self.val_end
        )
        
        print(f"Successfully prepared datasets for {len(self.tickers)} tickers")
        return self.datasets_l1, norm_stats, self.returns
    
    def train_level1(self):
        """Train Level 1 agents (Tech and Sent)."""
        print("\n" + "="*70)
        print("LEVEL 1: TRAINING SPECIALISTS")
        print("="*70)
        
        print("\n" + "="*50)
        print("TRAINING LEVEL 1: Tech & Sent Agents")
        print("="*50)
        
        # Prepare Level 1 datasets
        self.datasets_l1, norm_stats, self.returns = self.prepare_data()
        
        print(f"Level 1 datasets ready: {len(self.tickers)} tickers")
        print(f"Using {self.l1_n_envs} parallel environments for CPU speedup")
        
        # Train Tech Agent
        print("\nTraining Tech Agent...")
        tech_env = create_env_for_portfolio('train', self.datasets_l1, self.returns)
        self.tech_agent.train(tech_env, timesteps=self.l1_timesteps, n_envs=self.l1_n_envs)
        print("Tech Agent trained")
        
        # Train Sent Agent
        print("\nTraining Sent Agent...")
        sent_env = create_env_for_portfolio('train', self.datasets_l1, self.returns)
        self.sent_agent.train(sent_env, timesteps=self.l1_timesteps, n_envs=self.l1_n_envs)
        print("Sent Agent trained")
        
        # Save Level 1 models
        self._save_level1_models()
        
        # Prepare Level 2 datasets
        self.datasets_l2 = self._prepare_level2_datasets()
        print(f"Level 2 datasets ready: {len(self.tickers)} tickers")
        
    def train_level2(self):
        """Train Level 2 Super Agent."""
        print("\n" + "="*70)
        print("LEVEL 2: TRAINING INTEGRATOR")
        print("="*70)
        
        print("\n" + "="*50)
        print("TRAINING LEVEL 2: Super Agent")
        print("="*50)
        
        # Create Level 2 environment
        l2_env = create_env_for_portfolio('train', self.datasets_l2, self.returns)
        
        # Train Super Agent
        self.super_agent.train(l2_env, timesteps=self.l2_timesteps, n_envs=self.l2_n_envs)
        print("Super Agent trained")
        
        # Save Level 2 models
        self._save_level2_models()
        
        # Prepare Level 3 datasets
        self.datasets_l3 = self._prepare_level3_datasets()
        print(f"Level 3 datasets ready: {len(self.tickers)} tickers")
        
    def train_level3(self):
        """Train Level 3 SAC Meta Agent."""
        print("\n" + "="*70)
        print("LEVEL 3: TRAINING SAC META AGENT")
        print("="*70)
        
        # Create Level 3 environment
        l3_env = create_env_for_portfolio('train', self.datasets_l3, self.returns)
        
        # Train Meta Agent
        self.meta_agent.train(l3_env, timesteps=self.l3_timesteps, n_envs=self.l3_n_envs)
        print("SAC Meta Agent trained")
        
        # Save Level 3 models
        self._save_level3_models()
        
    def _prepare_level2_datasets(self):
        """Prepare Level 2 datasets with L1 predictions."""
        print("Creating Level 2 datasets (NO look-ahead)...")
        print("Generating Level 2 predictions (NO look-ahead)...")
        
        datasets_l2 = {}
        
        for ticker in self.tickers:
            print(f"  Processing {ticker}...")
            
            splits = {}
            for split_name in ['train', 'val', 'test']:
                # Create environment for this split
                l1_env = create_env_for_portfolio(split_name, self.datasets_l1, self.returns)
                
                # Get Tech predictions
                tech_predictions = []
                obs, _ = l1_env.reset()
                done = False
                step_count = 0
                while not done and step_count < len(self.datasets_l1[ticker][split_name]):
                    action, _ = self.tech_agent.model.predict(obs, deterministic=True)
                    tech_predictions.append(action)
                    obs, _, terminated, truncated, _ = l1_env.step(action)
                    done = terminated or truncated
                    step_count += 1
                
                # Get Sent predictions
                sent_predictions = []
                obs, _ = l1_env.reset()
                done = False
                step_count = 0
                while not done and step_count < len(self.datasets_l1[ticker][split_name]):
                    action, _ = self.sent_agent.model.predict(obs, deterministic=True)
                    sent_predictions.append(action)
                    obs, _, terminated, truncated, _ = l1_env.step(action)
                    done = terminated or truncated
                    step_count += 1
                
                # Ensure predictions match dataset length
                dataset_length = len(self.datasets_l1[ticker][split_name])
                if len(tech_predictions) < dataset_length:
                    last_tech = tech_predictions[-1] if tech_predictions else np.array([1.0])
                    tech_predictions.extend([last_tech] * (dataset_length - len(tech_predictions)))
                elif len(tech_predictions) > dataset_length:
                    tech_predictions = tech_predictions[:dataset_length]
                
                if len(sent_predictions) < dataset_length:
                    last_sent = sent_predictions[-1] if sent_predictions else np.array([1.0])
                    sent_predictions.extend([last_sent] * (dataset_length - len(sent_predictions)))
                elif len(sent_predictions) > dataset_length:
                    sent_predictions = sent_predictions[:dataset_length]
                
                # Combine predictions
                tech_array = np.array(tech_predictions)
                sent_array = np.array(sent_predictions)
                
                # Create L2 dataset
                l2_data = self.datasets_l1[ticker][split_name].copy()
                
                # Add prediction columns - flatten if needed
                if tech_array.ndim > 1:
                    tech_array = tech_array[:, 0] if tech_array.shape[1] > 0 else tech_array.mean(axis=1)
                if sent_array.ndim > 1:
                    sent_array = sent_array[:, 0] if sent_array.shape[1] > 0 else sent_array.mean(axis=1)
                
                l2_data['tech_pred'] = tech_array
                l2_data['sent_pred'] = sent_array
                
                splits[split_name] = l2_data
            
            datasets_l2[ticker] = splits
        
        return datasets_l2
    
    def _prepare_level3_datasets(self):
        """Prepare Level 3 datasets with L2 predictions and macro features."""
        print("Generating Level 3 datasets with macro features...")
        
        # Get macro and calendar features
        self.macro_data, self.calendar_data = get_macro_and_calendar_features(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        datasets_l3 = {}
        
        for ticker in self.tickers:
            print(f"  Processing {ticker}...")
            
            splits = {}
            for split_name in ['train', 'val', 'test']:
                # Create environment for this split
                l2_env = create_env_for_portfolio(split_name, self.datasets_l2, self.returns)
                
                # Get Super Agent predictions
                super_predictions = []
                obs, _ = l2_env.reset()
                done = False
                step_count = 0
                while not done and step_count < len(self.datasets_l2[ticker][split_name]):
                    action, _ = self.super_agent.model.predict(obs, deterministic=True)
                    super_predictions.append(action)
                    obs, _, terminated, truncated, _ = l2_env.step(action)
                    done = terminated or truncated
                    step_count += 1
                
                # Ensure predictions match dataset length
                dataset_length = len(self.datasets_l2[ticker][split_name])
                if len(super_predictions) < dataset_length:
                    last_super = super_predictions[-1] if super_predictions else np.array([1.0])
                    super_predictions.extend([last_super] * (dataset_length - len(super_predictions)))
                elif len(super_predictions) > dataset_length:
                    super_predictions = super_predictions[:dataset_length]
                
                # Combine predictions
                super_array = np.array(super_predictions)
                
                # Create L3 dataset with macro features
                l3_data = self.datasets_l2[ticker][split_name].copy()
                
                # Add prediction columns - flatten if needed
                if super_array.ndim > 1:
                    super_array = super_array[:, 0] if super_array.shape[1] > 0 else super_array.mean(axis=1)
                
                l3_data['super_pred'] = super_array
                
                # Add macro features
                macro_features = self.macro_data.loc[l3_data.index].values
                for i, col in enumerate(self.macro_data.columns):
                    l3_data[f'macro_{col}'] = macro_features[:, i]
                
                # Add calendar features
                calendar_features = self.calendar_data.loc[l3_data.index].values
                for i, col in enumerate(self.calendar_data.columns):
                    l3_data[f'calendar_{col}'] = calendar_features[:, i]
                
                splits[split_name] = l3_data
            
            datasets_l3[ticker] = splits
        
        return datasets_l3
    
    def _save_level1_models(self, save_dir='models/level1'):
        """Save Level 1 models."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        if self.tech_agent and self.tech_agent.model:
            self.tech_agent.model.save(os.path.join(save_dir, 'tech_agent.zip'))
            print(f"✓ Tech agent saved to {save_dir}/tech_agent.zip")
        if self.sent_agent and self.sent_agent.model:
            self.sent_agent.model.save(os.path.join(save_dir, 'sent_agent.zip'))
            print(f"✓ Sent agent saved to {save_dir}/sent_agent.zip")
    
    def _save_level2_models(self, save_dir='models/level2'):
        """Save Level 2 models."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        if self.super_agent and self.super_agent.model:
            self.super_agent.model.save(os.path.join(save_dir, 'super_agent.zip'))
            print(f"✓ Super agent saved to {save_dir}/super_agent.zip")
    
    def _save_level3_models(self, save_dir='models/level3'):
        """Save Level 3 models."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        if self.meta_agent and self.meta_agent.model:
            self.meta_agent.model.save(os.path.join(save_dir, 'sac_meta_agent.zip'))
            print(f"✓ SAC meta agent saved to {save_dir}/sac_meta_agent.zip")
    
    def load_level1_models(self, save_dir='models/level1'):
        """Load Level 1 models."""
        import os
        from stable_baselines3 import PPO
        if os.path.exists(os.path.join(save_dir, 'tech_agent.zip')):
            self.tech_agent.model = PPO.load(os.path.join(save_dir, 'tech_agent.zip'))
        if os.path.exists(os.path.join(save_dir, 'sent_agent.zip')):
            self.sent_agent.model = PPO.load(os.path.join(save_dir, 'sent_agent.zip'))
    
    def load_level2_models(self, save_dir='models/level2'):
        """Load Level 2 models."""
        import os
        from stable_baselines3 import SAC
        if os.path.exists(os.path.join(save_dir, 'super_agent.zip')):
            self.super_agent.model = SAC.load(os.path.join(save_dir, 'super_agent.zip'))
    
    def load_level3_models(self, save_dir='models/level3'):
        """Load Level 3 models."""
        import os
        from stable_baselines3 import SAC
        if os.path.exists(os.path.join(save_dir, 'sac_meta_agent.zip')):
            self.meta_agent.model = SAC.load(os.path.join(save_dir, 'sac_meta_agent.zip'))
    
    def evaluate(self, split='test'):
        """Evaluate the complete pipeline."""
        print(f"\nEvaluating on {split} set...")
        
        # Create test environment
        test_env = create_env_for_portfolio(split, self.datasets_l3, self.returns)
        
        # Run evaluation
        obs, _ = test_env.reset()
        done = False
        portfolio_returns = []
        
        while not done:
            action, _ = self.meta_agent.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            portfolio_returns.append(reward)
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate metrics
        sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-9)
        total_return = np.prod(1 + portfolio_returns) - 1
        volatility = np.std(portfolio_returns)
        
        results = {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'volatility': volatility,
            'portfolio_returns': portfolio_returns
        }
        
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Total Return: {total_return:.4f}")
        print(f"Volatility: {volatility:.4f}")
        
        return results
