"""
Residual Meta Learning Pipeline
========================================

CRITICAL FIXES:
1. L2/L3 predictions generated ONLY on respective splits (no look-ahead)
2. Meta trains on TRAIN set (not validation)
3. Portfolio-wide Meta (not per-ticker)
4. Proper observation shape handling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from utile import (
    prepare_datasets,
    create_env_for_portfolio,
    TechnicalAgent,
    SentimentAgent,
    SuperAgent,
    get_macro_indicators,
    build_calendar_frame
)
from residual_meta_enviroment import create_residual_meta_environment
from residual_meta_agent import ResidualMetaAgent


class ResidualMetaLearningPipeline:
    """
    Complete 3-level training pipeline with data leakage fixes.
    """
    
    def __init__(self, tickers: List[str], benchmark: str = 'QQQ'):
        """Initialize the pipeline."""
        if not tickers:
            raise ValueError("tickers list cannot be empty")
        if not benchmark:
            raise ValueError("benchmark cannot be empty")
            
        self.tickers = tickers
        self.benchmark = benchmark
        
        # Agents
        self.tech_agent = None
        self.sent_agent = None
        self.super_agent = None
        self.meta_agent = None  # Single portfolio-wide Meta agent
        
        # Datasets
        self.datasets_l1 = None
        self.datasets_l2 = None
        self.datasets_l3 = None
        self.returns = None
        self.macro_data = None
        self.calendar_data = None
    
    def train_level1(
        self,
        start: str,
        end: str,
        train_end: str,
        val_end: str,
        timesteps: int = 50_000,
        n_envs: int = 4,
        tech_callback=None,
        sent_callback=None
    ):
        """Train Level 1: Tech and Sent agents on technical indicators."""
        # Validate dates
        start_dt = pd.to_datetime(start)
        train_end_dt = pd.to_datetime(train_end)
        val_end_dt = pd.to_datetime(val_end)
        end_dt = pd.to_datetime(end)
        
        if not (start_dt < train_end_dt < val_end_dt < end_dt):
            raise ValueError("Dates must be: start < train_end < val_end < end")
        
        print("\n" + "=" * 70)
        print("TRAINING LEVEL 1: Tech & Sent Agents")
        print("=" * 70)
        
        # Prepare datasets with technical indicators only
        print("Preparing datasets with enhanced technical indicators...")
        self.datasets_l1, _, self.returns = prepare_datasets(
            tickers=self.tickers,
            benchmark=self.benchmark,
            start=start,
            end=end,
            train_end=train_end,
            val_end=val_end,
            windows=[4, 8, 12],
            include_lags=3,
            include_returns=True,
            macro=False,
            calendar=False,
        )
        
        print(f"Level 1 datasets ready: {len(self.datasets_l1)} tickers")
        print(f"Using {n_envs} parallel environments for CPU speedup")
        
        # Create portfolio environment (TRAIN ONLY)
        train_env = create_env_for_portfolio('train', self.datasets_l1, self.returns)
        
        # Train Tech agent
        print("\nTraining Tech Agent...")
        self.tech_agent = TechnicalAgent()
        self.tech_agent.train(train_env, timesteps=timesteps, verbose=1, n_envs=n_envs, callback=tech_callback)
        print("Tech Agent trained")
        
        # Train Sent agent
        print("\nTraining Sent Agent...")
        self.sent_agent = SentimentAgent()
        self.sent_agent.train(train_env, timesteps=timesteps, verbose=1, n_envs=n_envs, callback=sent_callback)
        print("Sent Agent trained")
    
    def train_level2(self, timesteps: int = 50_000, n_envs: int = 4, callback=None):
        """Train Level 2: Super agent (NO LOOK-AHEAD BIAS)."""
        print("\n" + "=" * 70)
        print("TRAINING LEVEL 2: Super Agent")
        print("=" * 70)
        
        # FIX: Generate L2 datasets WITHOUT look-ahead
        print("Creating Level 2 datasets (NO look-ahead)...")
        self.datasets_l2 = self._prepare_level2_datasets_fixed()
        print(f"Using {n_envs} parallel environments for CPU speedup")
        
        # Create environment (TRAIN ONLY)
        train_env = create_env_for_portfolio('train', self.datasets_l2, self.returns)
        
        # Train Super agent
        print("\nTraining Super Agent...")
        self.super_agent = SuperAgent()
        self.super_agent.train(train_env, timesteps=timesteps, verbose=1, n_envs=n_envs, callback=callback)
        print("Super Agent trained")
    
    def train_level3(
        self,
        start: str,
        end: str,
        timesteps: int = 300_000,
        n_envs: int = 1,
        adjustment_penalty: float = 0.01,
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        adjustment_scale: float = 0.2,
        callback=None
    ):
        """Train Level 3: Meta agent (TRAIN SET, not val)."""
        print("\n" + "=" * 70)
        print("TRAINING LEVEL 3: Meta Agent (Portfolio-wide)")
        print("=" * 70)
        
        # FIX: Generate L3 datasets WITHOUT look-ahead
        print("Creating Level 3 datasets with macro context...")
        self.datasets_l3 = self._prepare_level3_datasets_fixed(start, end)
        
        print(f"\nTraining Meta Agent on TRAIN set (fixed)...")
        if n_envs > 1:
            print(f"Using {n_envs} parallel environments")
        
        # Combine macro and calendar features
        combined_features = self._combine_macro_calendar()
        
        # FIX: Create residual meta environment on TRAIN set
        meta_env = create_residual_meta_environment(
            asset='PORTFOLIO',
            split='train',  # FIX: Was 'val', now 'train'
            datasets=self.datasets_l1,
            returns=self.returns,
            tech_agent=self.tech_agent,
            sent_agent=self.sent_agent,
            super_agent=self.super_agent,
            macro_df=combined_features,
            adjustment_penalty=adjustment_penalty  # Now configurable
        )
        
        # Train Meta agent
        self.meta_agent = ResidualMetaAgent(
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            adjustment_scale=adjustment_scale
        )
        
        # Pass custom callback if provided (e.g., for early stopping)
        if callback is not None:
            print("  Note: Using custom callback for early stopping")
        
        self.meta_agent.train(meta_env, timesteps=timesteps, verbose=1, n_envs=n_envs, custom_callback=callback)
        print("Meta Agent trained (portfolio-wide)")
    
    def evaluate(self, split_name: str = 'test'):
        """Evaluate all levels on specified split."""
        if split_name not in ['train', 'val', 'test']:
            raise ValueError(f"split_name must be 'train', 'val', or 'test', got {split_name}")
        
        print("\n" + "=" * 70)
        print(f"EVALUATION ON {split_name.upper()} SET")
        print("=" * 70)
        
        results = {}
        
        # Level 1 evaluation
        print("\nLevel 1 Performance:")
        l1_env = create_env_for_portfolio(split_name, self.datasets_l1, self.returns)
        l1_results = self._evaluate_env(l1_env, self.tech_agent, "Tech")
        results['tech'] = l1_results
        
        # Level 2 evaluation
        print("\nLevel 2 Performance:")
        # FIX: Generate L2 predictions for this split only
        l2_datasets_eval = self._generate_l2_predictions_for_split(split_name)
        l2_env = create_env_for_portfolio(split_name, l2_datasets_eval, self.returns)
        l2_results = self._evaluate_env(l2_env, self.super_agent, "Super")
        results['super'] = l2_results
        
        # Level 3 evaluation
        print("\nLevel 3 Performance (with residual adjustments):")
        combined_features = self._combine_macro_calendar()
        
        meta_env = create_residual_meta_environment(
            asset='PORTFOLIO',
            split=split_name,
            datasets=self.datasets_l1,
            returns=self.returns,
            tech_agent=self.tech_agent,
            sent_agent=self.sent_agent,
            super_agent=self.super_agent,
            macro_df=combined_features,
            adjustment_penalty=0.01  # Match training
        )
        
        meta_result = self._evaluate_residual_env(meta_env, self.meta_agent, "Meta")
        results['meta'] = meta_result
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _prepare_level2_datasets_fixed(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        FIX: Generate L2 predictions separately for each split (NO look-ahead).
        """
        super_datasets = {}
        
        # Initialize structure
        for ticker in self.tickers:
            super_datasets[ticker] = {}
        
        # Process each split INDEPENDENTLY
        for split_name in ['train', 'val', 'test']:
            print(f"  Generating L1 predictions for {split_name} split...")
            
            # Create environment for THIS split only
            env = create_env_for_portfolio(split_name, self.datasets_l1, self.returns)
            
            # Collect predictions
            tech_preds = []
            sent_preds = []
            
            obs, _ = env.reset()
            for _ in range(env.n_steps):
                tech_pred = self.tech_agent.predict(obs)
                sent_pred = self.sent_agent.predict(obs)
                
                tech_preds.append(tech_pred.flatten())
                sent_preds.append(sent_pred.flatten())
                
                obs, _, terminated, truncated, _ = env.step(tech_pred)
                if terminated or truncated:
                    break
            
            # FIX: Use environment dates (common dates across all tickers)
            n_predictions = len(tech_preds)
            
            # Get common dates from environment (these are the actual dates used)
            common_dates = None
            for ticker in self.tickers:
                ticker_dates = self.datasets_l1[ticker][split_name].index
                common_dates = ticker_dates if common_dates is None else common_dates.intersection(ticker_dates)
            
            # Sort and take the first n_predictions dates
            common_dates = sorted(common_dates)[:n_predictions]
            
            # Now split predictions by asset using environment dates
            for ticker_idx, ticker in enumerate(self.tickers):
                original_feats = self.datasets_l1[ticker][split_name]
                
                # Extract predictions for this ticker
                tech_ticker_preds = [pred[ticker_idx] if len(pred) > ticker_idx else pred[0] 
                                    for pred in tech_preds]
                sent_ticker_preds = [pred[ticker_idx] if len(pred) > ticker_idx else pred[0] 
                                    for pred in sent_preds]
                
                # Create prediction DataFrames using environment dates
                tech_df = pd.DataFrame(
                    [[p] for p in tech_ticker_preds],
                    index=common_dates,
                    columns=['tech_w0']
                )
                sent_df = pd.DataFrame(
                    [[p] for p in sent_ticker_preds],
                    index=common_dates,
                    columns=['sent_w0']
                )
                
                # Get original features for common dates
                common_feats = original_feats.loc[common_dates]
                
                # Combine with original features
                super_feats = pd.concat([
                    common_feats,
                    tech_df,
                    sent_df
                ], axis=1)
                
                super_datasets[ticker][split_name] = super_feats
        
        return super_datasets
    
    def _generate_l2_predictions_for_split(self, split_name: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Generate L2 predictions for a specific split (used in evaluation)."""
        datasets = {}
        
        # Create environment for THIS split only
        # The environment already handles common dates across all tickers
        env = create_env_for_portfolio(split_name, self.datasets_l1, self.returns)
        
        # Collect predictions
        tech_preds = []
        sent_preds = []
        
        obs, _ = env.reset()
        for _ in range(env.n_steps):
            tech_pred = self.tech_agent.predict(obs)
            sent_pred = self.sent_agent.predict(obs)
            
            tech_preds.append(tech_pred.flatten())
            sent_preds.append(sent_pred.flatten())
            
            obs, _, terminated, truncated, _ = env.step(tech_pred)
            if terminated or truncated:
                break
        
        # Portfolio environment ensures all tickers have same length (common dates)
        n_preds = len(tech_preds)
        
        # Get common dates from environment (these are the actual dates used)
        common_dates = None
        for ticker in self.tickers:
            ticker_dates = self.datasets_l1[ticker][split_name].index
            common_dates = ticker_dates if common_dates is None else common_dates.intersection(ticker_dates)
        
        # Sort and take the first n_preds dates
        common_dates = sorted(common_dates)[:n_preds]
        
        # Split by ticker
        for ticker_idx, ticker in enumerate(self.tickers):
            datasets[ticker] = {}
            original_feats = self.datasets_l1[ticker][split_name]
            
            # Extract predictions for this ticker
            tech_ticker_preds = [pred[ticker_idx] if len(pred) > ticker_idx else pred[0] 
                                for pred in tech_preds]
            sent_ticker_preds = [pred[ticker_idx] if len(pred) > ticker_idx else pred[0] 
                                for pred in sent_preds]
            
            # Create DataFrames using environment dates
            tech_df = pd.DataFrame(
                [[p] for p in tech_ticker_preds],
                index=common_dates,
                columns=['tech_w0']
            )
            sent_df = pd.DataFrame(
                [[p] for p in sent_ticker_preds],
                index=common_dates,
                columns=['sent_w0']
            )
            
            # Get original features for common dates
            common_feats = original_feats.loc[common_dates]
            
            super_feats = pd.concat([
                common_feats,
                tech_df,
                sent_df
            ], axis=1)
            
            datasets[ticker][split_name] = super_feats
        
        return datasets
    
    def _prepare_level3_datasets_fixed(self, start: str, end: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        FIX: Generate L3 predictions separately for each split (NO look-ahead).
        """
        print("Fetching macro indicators...")
        self.macro_data = get_macro_indicators(start, end)
        
        print("Building calendar features...")
        all_dates = []
        for ticker in self.tickers:
            for split in self.datasets_l2[ticker].keys():
                all_dates.extend(self.datasets_l2[ticker][split].index)
        unique_dates = pd.DatetimeIndex(sorted(set(all_dates)))
        self.calendar_data = build_calendar_frame(unique_dates)
        
        meta_datasets = {}
        
        # Initialize structure
        for ticker in self.tickers:
            meta_datasets[ticker] = {}
        
        # Process each split INDEPENDENTLY
        for split_name in ['train', 'val', 'test']:
            print(f"  Generating L2 predictions for {split_name} split...")
            
            # Generate L2 datasets for this split
            l2_split_datasets = self._generate_l2_predictions_for_split(split_name)
            
            # Create environment for THIS split only
            env = create_env_for_portfolio(split_name, l2_split_datasets, self.returns)
            
            # Collect Super predictions
            super_preds = []
            obs, _ = env.reset()
            
            for _ in range(env.n_steps):
                super_pred = self.super_agent.predict(obs)
                super_preds.append(super_pred.flatten())
                obs, _, terminated, truncated, _ = env.step(super_pred)
                if terminated or truncated:
                    break
            
            # Portfolio environment ensures all tickers have same length
            n_preds = len(super_preds)
            
            # Get common dates from L2 datasets (these are already aligned)
            common_dates = None
            for ticker in self.tickers:
                ticker_dates = l2_split_datasets[ticker][split_name].index
                common_dates = ticker_dates if common_dates is None else common_dates.intersection(ticker_dates)
            
            # Sort and take the first n_preds dates
            common_dates = sorted(common_dates)[:n_preds]
            
            # Split by ticker
            for ticker_idx, ticker in enumerate(self.tickers):
                super_feats = l2_split_datasets[ticker][split_name]
                
                # Extract Super predictions for this ticker
                super_ticker_preds = [pred[ticker_idx] if len(pred) > ticker_idx else pred[0] 
                                     for pred in super_preds]
                
                # Create DataFrame using environment dates
                super_df = pd.DataFrame(
                    [[p] for p in super_ticker_preds],
                    index=common_dates,
                    columns=['super_w0']
                )
                
                # Get L2 features for common dates
                common_feats = super_feats.loc[common_dates]
                meta_feats = common_feats.copy()
                meta_feats = meta_feats.join(super_df)
                
                # Add macro features
                if self.macro_data is not None and not self.macro_data.empty:
                    meta_feats = meta_feats.join(self.macro_data, how='left')
                
                # Add calendar features
                if self.calendar_data is not None and not self.calendar_data.empty:
                    meta_feats = meta_feats.join(self.calendar_data, how='left')
                
                meta_feats = meta_feats.dropna()
                
                meta_datasets[ticker][split_name] = meta_feats
        
        return meta_datasets
    
    def _combine_macro_calendar(self) -> pd.DataFrame:
        """Combine macro and calendar features into a single DataFrame."""
        combined_features = pd.DataFrame()
        
        if self.macro_data is not None and not self.macro_data.empty:
            combined_features = self.macro_data.copy()
        
        if self.calendar_data is not None and not self.calendar_data.empty:
            if combined_features.empty:
                combined_features = self.calendar_data.copy()
            else:
                combined_features = combined_features.join(self.calendar_data, how='outer')
        
        return combined_features
    
    def _evaluate_env(self, env, agent, name):
        """Evaluate a standard agent on an environment."""
        obs, _ = env.reset()
        returns = []
        
        for _ in range(env.n_steps):
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            returns.append(info['portfolio_return'])
            if terminated or truncated:
                break
        
        returns = np.array(returns)
        total_return = np.exp(returns.sum()) - 1
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(52)
        
        print(f"  {name} - Return: {total_return*100:.2f}%, Sharpe: {sharpe:.3f}")
        return {'returns': returns, 'total_return': total_return, 'sharpe': sharpe}
    
    def _evaluate_residual_env(self, env, agent, name):
        """Evaluate a Meta agent on a residual environment."""
        obs, _ = env.reset()
        returns = []
        rewards = []
        improvements = []
        adjustments = []
        
        for _ in range(env.base_env.n_steps):
            action, _ = agent.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            returns.append(info['portfolio_return'])
            rewards.append(reward)
            
            if 'meta_residual' in info:
                improvements.append(info['meta_residual']['base_reward'])
                adjustments.append(info['meta_residual']['adjustment_magnitude'])
            
            if terminated or truncated:
                break
        
        returns = np.array(returns)
        rewards = np.array(rewards)
        improvements = np.array(improvements) if improvements else np.array([0])
        adjustments = np.array(adjustments) if adjustments else np.array([0])
        
        total_return = np.exp(returns.sum()) - 1
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(52)
        mean_reward = np.mean(rewards)
        total_penalty = returns.sum() - rewards.sum()
        avg_improvement = np.mean(improvements)
        avg_adjustment_size = np.mean(adjustments)
        
        print(f"    Return: {total_return*100:.2f}%, Sharpe: {sharpe:.3f}")
        print(f"    Mean reward: {mean_reward:.6f}, Total penalty: {total_penalty:.6f}")
        print(f"    Avg improvement: {avg_improvement*100:.4f}%")
        print(f"    Avg adjustment: {avg_adjustment_size:.4f}")
        
        return {
            'returns': returns,
            'rewards': rewards,
            'total_return': total_return,
            'sharpe': sharpe,
            'mean_reward': mean_reward,
            'total_penalty': total_penalty,
            'improvements': improvements,
            'adjustments': adjustments
        }
    
    def _get_meta_training_data(self, split: str = 'train'):
        """
        Extract data for training linear meta agent.
        
        Parameters
        ----------
        split : str
            'train', 'val', or 'test'
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (super_weights, macro_features, returns)
        """
        # FIX: Use datasets_l2 (Level 2 data) not datasets_l1
        # Super agent was trained on Level 2 data which includes Tech/Sent predictions
        datasets_split = {ticker: self.datasets_l2[ticker][split] 
                         for ticker in self.datasets_l2.keys()}
        
        n_timesteps = len(next(iter(datasets_split.values())))
        n_assets = len(self.datasets_l2)
        
        super_weights = []
        
        # Get Super predictions
        for t in range(n_timesteps):
            # Build observation for each asset at time t
            obs_list = []
            for ticker in sorted(self.datasets_l2.keys()):
                asset_data = datasets_split[ticker].iloc[t:t+1].values
                obs_list.append(asset_data)
            
            # Stack observations (n_assets, n_features, 1)
            obs = np.stack(obs_list, axis=0).astype(np.float32)
            obs = obs.reshape((n_assets, -1, 1))
            
            # Get Super prediction
            action, _ = self.super_agent.model.predict(obs, deterministic=True)
            super_weights.append(action)
        
        super_weights = np.array(super_weights)
        
        # Get macro features (already aligned by date)
        if hasattr(self, 'macro_data') and self.macro_data is not None:
            # Extract dates from first ticker
            first_ticker = sorted(self.datasets_l2.keys())[0]
            dates = datasets_split[first_ticker].index
            
            # Align macro features with dates
            macro_features = []
            for date in dates:
                if date in self.macro_data.index:
                    macro_features.append(self.macro_data.loc[date].values)
                else:
                    # Use forward fill for missing dates
                    prev_date = self.macro_data.index[self.macro_data.index < date]
                    if len(prev_date) > 0:
                        macro_features.append(self.macro_data.loc[prev_date[-1]].values)
                    else:
                        # Use first available if before all data
                        macro_features.append(self.macro_data.iloc[0].values)
            
            macro_features = np.array(macro_features)
        else:
            # No macro features - use zeros
            macro_features = np.zeros((n_timesteps, 1))
        
        # Get realized returns
        returns_data = self.returns[self.returns.index.isin(dates)]
        returns_array = []
        for ticker in sorted(self.datasets_l2.keys()):
            ticker_returns = returns_data[returns_data['ticker'] == ticker]['return'].values
            returns_array.append(ticker_returns)
        
        returns_array = np.array(returns_array).T  # Shape: (n_timesteps, n_assets)
        
        # Ensure all arrays have same length
        min_len = min(len(super_weights), len(macro_features), len(returns_array))
        super_weights = super_weights[:min_len]
        macro_features = macro_features[:min_len]
        returns_array = returns_array[:min_len]
        
        return super_weights, macro_features, returns_array
    
    def _print_summary(self, results):
        """Print evaluation summary for all levels."""
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nLevel 1 - Tech: Return={results['tech']['total_return']*100:.2f}%, "
              f"Sharpe={results['tech']['sharpe']:.3f}")
        print(f"Level 2 - Super: Return={results['super']['total_return']*100:.2f}%, "
              f"Sharpe={results['super']['sharpe']:.3f}")
        print(f"Level 3 - Meta: Return={results['meta']['total_return']*100:.2f}%, "
              f"Sharpe={results['meta']['sharpe']:.3f}")
        
        print(f"{'='*70}")