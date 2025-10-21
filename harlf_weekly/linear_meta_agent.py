"""
Linear Meta Agent - Simple and Interpretable

Uses linear regression instead of RL:
meta_weights = softmax(W @ super_weights + b @ macro_features)

Advantages:
- Much simpler and faster to train
- Interpretable coefficients
- No RL complexity for a simple adjustment task
- L2 regularization prevents overfitting
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax


class LinearMetaAgent:
    """
    Meta agent using linear regression.
    
    Learns: meta_weights = softmax(W @ [super_weights; macro_features] + bias)
    """
    
    def __init__(
        self,
        n_assets: int,
        alpha: float = 1.0  # L2 regularization strength
    ):
        self.n_assets = n_assets
        self.alpha = alpha
        self.models = None  # Will be a list of Ridge models
        self.scaler_macro = StandardScaler()
        self.scaler_super = StandardScaler()
        self.training_history = []
        
        print(f"\nLinear Meta Agent initialized")
        print(f"  Assets: {n_assets}")
        print(f"  L2 regularization (alpha): {alpha}")
        print(f"  Model: Ridge Regression + Softmax")
    
    def train(
        self,
        super_weights: np.ndarray,
        macro_features: np.ndarray,
        returns: np.ndarray,
        val_super_weights: Optional[np.ndarray] = None,
        val_macro_features: Optional[np.ndarray] = None,
        val_returns: Optional[np.ndarray] = None
    ):
        """
        Train linear meta model.
        
        Parameters
        ----------
        super_weights : np.ndarray
            Shape (n_samples, n_assets) - Super agent's predictions
        macro_features : np.ndarray
            Shape (n_samples, n_macro_features) - Macro indicators
        returns : np.ndarray
            Shape (n_samples, n_assets) - Next period returns for each asset
        val_* : Optional validation data for monitoring
        """
        print(f"\n{'='*70}")
        print("TRAINING LINEAR META AGENT")
        print(f"{'='*70}")
        print(f"Training samples: {len(super_weights)}")
        print(f"Assets: {self.n_assets}")
        print(f"Macro features: {macro_features.shape[1]}")
        
        # Standardize inputs
        super_weights_scaled = self.scaler_super.fit_transform(super_weights)
        macro_features_scaled = self.scaler_macro.fit_transform(macro_features)
        
        # Combine features
        X = np.hstack([super_weights_scaled, macro_features_scaled])
        
        # Target: which assets performed best?
        # We'll predict the logits before softmax
        # Target is the realized return-weighted portfolio
        y_targets = []
        for i in range(len(returns)):
            # Normalize returns to create target weights
            ret = returns[i]
            if np.sum(ret) > 0:
                # Emphasize assets that did well
                target = np.maximum(ret, 0)
                target = target / (np.sum(target) + 1e-8)
            else:
                # If all negative, equal weight
                target = np.ones(self.n_assets) / self.n_assets
            y_targets.append(target)
        
        y_targets = np.array(y_targets)
        
        # Train separate Ridge model for each asset's weight
        self.models = []
        for asset_idx in range(self.n_assets):
            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(X, y_targets[:, asset_idx])
            self.models.append(model)
        
        # Evaluate on training set
        train_pred = self.predict(super_weights, macro_features)
        train_returns = np.sum(train_pred * returns, axis=1)
        train_total_return = np.sum(train_returns)
        train_sharpe = np.mean(train_returns) / (np.std(train_returns) + 1e-8) * np.sqrt(52)
        
        print(f"\nTraining Results:")
        print(f"  Total Return: {train_total_return*100:.2f}%")
        print(f"  Sharpe Ratio: {train_sharpe:.2f}")
        print(f"  Avg Weekly Return: {np.mean(train_returns)*100:.2f}%")
        
        self.training_history.append({
            'train_return': train_total_return,
            'train_sharpe': train_sharpe
        })
        
        # Evaluate on validation if provided
        if val_super_weights is not None and val_macro_features is not None and val_returns is not None:
            val_pred = self.predict(val_super_weights, val_macro_features)
            val_returns_arr = np.sum(val_pred * val_returns, axis=1)
            val_total_return = np.sum(val_returns_arr)
            val_sharpe = np.mean(val_returns_arr) / (np.std(val_returns_arr) + 1e-8) * np.sqrt(52)
            
            print(f"\nValidation Results:")
            print(f"  Total Return: {val_total_return*100:.2f}%")
            print(f"  Sharpe Ratio: {val_sharpe:.2f}")
            print(f"  Avg Weekly Return: {np.mean(val_returns_arr)*100:.2f}%")
            
            self.training_history[-1]['val_return'] = val_total_return
            self.training_history[-1]['val_sharpe'] = val_sharpe
        
        print(f"{'='*70}\n")
        
        # Print feature importance
        self._print_feature_importance(macro_features.shape[1])
    
    def predict(self, super_weights: np.ndarray, macro_features: np.ndarray) -> np.ndarray:
        """
        Predict meta weights.
        
        Returns
        -------
        np.ndarray
            Shape (n_samples, n_assets) - Predicted portfolio weights
        """
        if self.models is None:
            raise ValueError("Model not trained yet")
        
        # Standardize inputs
        super_weights_scaled = self.scaler_super.transform(super_weights)
        macro_features_scaled = self.scaler_macro.transform(macro_features)
        
        # Combine features
        X = np.hstack([super_weights_scaled, macro_features_scaled])
        
        # Predict raw scores for each asset
        raw_scores = np.zeros((len(X), self.n_assets))
        for asset_idx, model in enumerate(self.models):
            raw_scores[:, asset_idx] = model.predict(X)
        
        # Apply softmax to get valid weights
        weights = np.apply_along_axis(softmax, 1, raw_scores)
        
        return weights
    
    def evaluate(self, super_weights: np.ndarray, macro_features: np.ndarray, 
                 returns: np.ndarray) -> Dict:
        """
        Evaluate on test set.
        
        Returns
        -------
        Dict
            Performance metrics and arrays
        """
        pred_weights = self.predict(super_weights, macro_features)
        portfolio_returns = np.sum(pred_weights * returns, axis=1)
        
        # Calculate metrics
        total_return = np.sum(portfolio_returns)
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        sharpe = mean_return / (std_return + 1e-8) * np.sqrt(52)
        
        # Max drawdown
        cum_returns = np.exp(np.cumsum(portfolio_returns))
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Adjustment magnitudes (how much we deviate from Super)
        adjustment_magnitudes = np.sum(np.abs(pred_weights - super_weights), axis=1)
        
        results = {
            'total_return': total_return,
            'sharpe': sharpe,
            'mean_return': mean_return,
            'std_return': std_return,
            'max_drawdown': max_drawdown,
            'returns': portfolio_returns,
            'weights': pred_weights,
            'adjustment_magnitudes': adjustment_magnitudes
        }
        
        return results
    
    def save(self, path: str):
        """Save trained models."""
        import pickle
        if self.models is not None:
            with open(path, 'wb') as f:
                pickle.dump({
                    'models': self.models,
                    'scaler_macro': self.scaler_macro,
                    'scaler_super': self.scaler_super,
                    'n_assets': self.n_assets,
                    'alpha': self.alpha
                }, f)
            print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load trained models."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.scaler_macro = data['scaler_macro']
            self.scaler_super = data['scaler_super']
            self.n_assets = data['n_assets']
            self.alpha = data['alpha']
        print(f"✓ Model loaded from {path}")
    
    def _print_feature_importance(self, n_macro_features: int):
        """Print feature importance from coefficients."""
        if self.models is None:
            return
        
        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE (Average Absolute Coefficients)")
        print(f"{'='*70}")
        
        # Aggregate coefficients across all asset models
        all_coefs = np.array([model.coef_ for model in self.models])
        avg_abs_coefs = np.mean(np.abs(all_coefs), axis=0)
        
        # Feature names
        feature_names = [f'Super_W{i+1}' for i in range(self.n_assets)] + \
                       [f'Macro_{i+1}' for i in range(n_macro_features)]
        
        # Sort by importance
        importance_idx = np.argsort(avg_abs_coefs)[::-1]
        
        print(f"\nTop 10 Most Important Features:")
        for i, idx in enumerate(importance_idx[:10]):
            print(f"  {i+1}. {feature_names[idx]}: {avg_abs_coefs[idx]:.4f}")
        
        print(f"{'='*70}\n")


if __name__ == "__main__":
    print(__doc__)
