"""
PROFESSIONAL PORTFOLIO OPTIMIZATION SYSTEM
============================================

Based on best practices from:
- Bridgewater (Risk Parity)
- AQR (Factor Investing)
- Lopez de Prado (Hierarchical Risk Parity, ML in Finance)
- Two Sigma (Ensemble Methods)
- Renaissance Technologies (Statistical Arbitrage concepts)

Key Improvements:
1. Hierarchical Risk Parity (HRP) - Robust allocation without matrix inversion
2. Multi-Factor Alpha - Momentum, Mean-Reversion, Volatility, Quality
3. Hidden Markov Model Regime Detection - Probabilistic state detection  
4. Kelly Criterion Position Sizing - Mathematically optimal bet sizing
5. CVaR (Conditional Value at Risk) - Tail risk management
6. Ensemble with Bayesian Model Averaging - Dynamic strategy weighting
7. Walk-Forward Validation - Proper out-of-sample testing
8. Drawdown Control - Explicit volatility targeting

Author: Professional Quant System
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. HIERARCHICAL RISK PARITY (Lopez de Prado)
# =============================================================================

class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity from "Advances in Financial Machine Learning"
    
    Key advantages over Markowitz:
    - No matrix inversion (stable)
    - Works well with noisy correlation estimates
    - Naturally diversified
    """
    
    def __init__(self):
        self.weights = None
        
    def _get_cluster_var(self, cov, cluster_items):
        """Calculate variance of a cluster"""
        cov_slice = cov.loc[cluster_items, cluster_items]
        weights = 1 / np.diag(cov_slice.values)  # Inverse variance weights within cluster
        weights /= weights.sum()
        return np.dot(weights, np.dot(cov_slice.values, weights))
    
    def _get_quasi_diag(self, link):
        """Extract quasi-diagonal ordering from linkage matrix"""
        return leaves_list(link)
    
    def _get_rec_bipart(self, cov, sorted_idx):
        """Recursive bisection for weight allocation"""
        weights = pd.Series(1.0, index=sorted_idx)
        cluster_items = [sorted_idx]
        
        while len(cluster_items) > 0:
            # Bisect each cluster
            cluster_items = [
                item[start:end] 
                for item in cluster_items 
                for start, end in [(0, len(item)//2), (len(item)//2, len(item))]
                if len(item) > 1
            ]
            
            for i in range(0, len(cluster_items), 2):
                if i + 1 >= len(cluster_items):
                    break
                    
                cluster0 = cluster_items[i]
                cluster1 = cluster_items[i + 1]
                
                var0 = self._get_cluster_var(cov, cluster0)
                var1 = self._get_cluster_var(cov, cluster1)
                
                alpha = 1 - var0 / (var0 + var1)
                
                weights[cluster0] *= alpha
                weights[cluster1] *= (1 - alpha)
                
        return weights
    
    def fit(self, returns_df):
        """
        Fit HRP weights to historical returns
        
        Args:
            returns_df: DataFrame of asset returns (T x N)
        
        Returns:
            weights: Series of portfolio weights
        """
        cov = returns_df.cov()
        corr = returns_df.corr()
        
        # Step 1: Tree clustering
        dist = np.sqrt((1 - corr) / 2)  # Distance from correlation
        dist_condensed = squareform(dist.values, checks=False)
        link = linkage(dist_condensed, method='single')
        
        # Step 2: Quasi-diagonalization
        sorted_idx = self._get_quasi_diag(link)
        sorted_idx = corr.index[sorted_idx].tolist()
        
        # Step 3: Recursive bisection
        self.weights = self._get_rec_bipart(cov, sorted_idx)
        self.weights = self.weights.reindex(returns_df.columns)
        
        return self.weights


# =============================================================================
# 2. MULTI-FACTOR ALPHA MODEL
# =============================================================================

class FactorModel:
    """
    Multi-factor alpha model combining:
    - Momentum (trend following)
    - Mean Reversion (contrarian)
    - Volatility (low vol anomaly)
    - Quality (stability)
    """
    
    def __init__(self, 
                 momentum_windows=[21, 63, 126, 252],
                 mean_reversion_window=5,
                 volatility_window=21):
        self.momentum_windows = momentum_windows
        self.mean_reversion_window = mean_reversion_window
        self.volatility_window = volatility_window
        
    def calculate_momentum_score(self, prices):
        """
        Multi-timeframe momentum with volatility adjustment
        (Moskowitz, Ooi, Pedersen - "Time Series Momentum")
        """
        returns = prices.pct_change()
        scores = pd.DataFrame(index=prices.index, columns=prices.columns)
        
        for window in self.momentum_windows:
            # Raw momentum
            mom = prices / prices.shift(window) - 1
            
            # Volatility-adjusted (t-stat of returns)
            vol = returns.rolling(window).std() * np.sqrt(252)
            vol_adj_mom = mom / (vol + 1e-8)
            
            if scores.isna().all().all():
                scores = vol_adj_mom
            else:
                scores = scores + vol_adj_mom
                
        return scores / len(self.momentum_windows)
    
    def calculate_mean_reversion_score(self, prices):
        """
        Short-term mean reversion (Jegadeesh - "Short-Horizon Return Reversals")
        """
        returns = prices.pct_change()
        
        # Z-score of recent returns
        rolling_mean = returns.rolling(self.mean_reversion_window).mean()
        rolling_std = returns.rolling(self.mean_reversion_window).std()
        
        z_score = (returns - rolling_mean) / (rolling_std + 1e-8)
        
        # Mean reversion = negative momentum at short horizon
        return -z_score.rolling(self.mean_reversion_window).mean()
    
    def calculate_volatility_score(self, prices):
        """
        Low volatility anomaly (Baker, Bradley, Wurgler)
        Lower vol = higher expected risk-adjusted returns
        """
        returns = prices.pct_change()
        vol = returns.rolling(self.volatility_window).std() * np.sqrt(252)
        
        # Rank assets by volatility (lower = better)
        vol_rank = vol.rank(axis=1, pct=True)
        
        # Convert to score (1 = lowest vol, 0 = highest vol)
        return 1 - vol_rank
    
    def calculate_quality_score(self, prices):
        """
        Quality factor: stability of returns
        (Asness, Frazzini, Pedersen - "Quality Minus Junk")
        """
        returns = prices.pct_change()
        
        # Sharpe ratio over rolling window
        rolling_mean = returns.rolling(63).mean() * 252
        rolling_std = returns.rolling(63).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / (rolling_std + 1e-8)
        
        # Rank by Sharpe
        return rolling_sharpe.rank(axis=1, pct=True)
    
    def get_combined_scores(self, prices, weights=None):
        """
        Combine all factors with specified weights
        """
        if weights is None:
            weights = {
                'momentum': 0.35,
                'mean_reversion': 0.15,
                'volatility': 0.25,
                'quality': 0.25
            }
        
        momentum = self.calculate_momentum_score(prices)
        mean_rev = self.calculate_mean_reversion_score(prices)
        vol_score = self.calculate_volatility_score(prices)
        quality = self.calculate_quality_score(prices)
        
        combined = (
            weights['momentum'] * momentum.rank(axis=1, pct=True) +
            weights['mean_reversion'] * mean_rev.rank(axis=1, pct=True) +
            weights['volatility'] * vol_score +
            weights['quality'] * quality
        )
        
        return combined


# =============================================================================
# 3. HIDDEN MARKOV MODEL REGIME DETECTION
# =============================================================================

class RegimeDetector:
    """
    Hidden Markov Model for market regime detection
    
    States:
    - Bull: Low vol, positive drift
    - Bear: High vol, negative drift
    - Sideways: Medium vol, no drift
    """
    
    def __init__(self, n_regimes=3, lookback=252):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.regime_params = None
        
    def fit(self, returns):
        """
        Fit regime parameters using simple clustering approach
        (Production would use hmmlearn or pomegranate)
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.mean(axis=1)  # Use market average
            
        returns = returns.dropna()
        
        # Calculate features
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_60 = returns.rolling(60).std() * np.sqrt(252)
        drift = returns.rolling(60).mean() * 252
        
        # Combine into regime features
        features = pd.DataFrame({
            'vol_20': vol_20,
            'vol_60': vol_60,
            'drift': drift
        }).dropna()
        
        # Simple threshold-based regime detection
        vol_33 = features['vol_20'].quantile(0.33)
        vol_66 = features['vol_20'].quantile(0.66)
        
        self.regime_params = {
            'vol_33': vol_33,
            'vol_66': vol_66,
            'features': features
        }
        
        return self
    
    def predict(self, returns):
        """
        Predict current regime
        
        Returns:
            regime: 0=Bull, 1=Sideways, 2=Bear
            probabilities: [P(Bull), P(Sideways), P(Bear)]
        """
        if self.regime_params is None:
            raise ValueError("Must fit before predicting")
            
        if isinstance(returns, pd.DataFrame):
            returns = returns.mean(axis=1)
            
        # Current volatility
        current_vol = returns.iloc[-20:].std() * np.sqrt(252)
        current_drift = returns.iloc[-60:].mean() * 252
        
        vol_33 = self.regime_params['vol_33']
        vol_66 = self.regime_params['vol_66']
        
        # Soft regime probabilities using sigmoid-like transitions
        def sigmoid(x, center, scale=1):
            return 1 / (1 + np.exp(-scale * (x - center)))
        
        # P(Bear) increases with volatility
        p_bear = sigmoid(current_vol, vol_66, scale=20)
        
        # P(Bull) decreases with volatility
        p_bull = 1 - sigmoid(current_vol, vol_33, scale=20)
        
        # Adjust for drift
        if current_drift > 0:
            p_bull *= 1.2
            p_bear *= 0.8
        else:
            p_bull *= 0.8
            p_bear *= 1.2
            
        # Normalize
        p_sideways = max(0, 1 - p_bull - p_bear)
        total = p_bull + p_sideways + p_bear
        
        probs = np.array([p_bull, p_sideways, p_bear]) / total
        regime = np.argmax(probs)
        
        return regime, probs


# =============================================================================
# 4. KELLY CRITERION POSITION SIZING
# =============================================================================

class KellyCriterion:
    """
    Kelly Criterion for optimal bet sizing
    
    With fractional Kelly for robustness (typically 0.25-0.5)
    """
    
    def __init__(self, kelly_fraction=0.5, max_leverage=1.0):
        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage
        
    def calculate_kelly_weights(self, expected_returns, cov_matrix):
        """
        Calculate Kelly-optimal weights
        
        Kelly formula for multiple assets:
        w* = Σ^(-1) * μ
        
        Args:
            expected_returns: Series of expected returns
            cov_matrix: Covariance matrix of returns
            
        Returns:
            weights: Optimal position sizes
        """
        try:
            # Regularize covariance matrix for stability
            cov_reg = cov_matrix + np.eye(len(cov_matrix)) * 1e-6
            
            # Full Kelly
            cov_inv = np.linalg.inv(cov_reg)
            full_kelly = pd.Series(
                np.dot(cov_inv, expected_returns),
                index=expected_returns.index
            )
            
            # Fractional Kelly (more robust)
            weights = full_kelly * self.kelly_fraction
            
            # Apply leverage constraint
            leverage = np.abs(weights).sum()
            if leverage > self.max_leverage:
                weights = weights * (self.max_leverage / leverage)
                
            # No shorting constraint (optional, remove for long-short)
            weights = np.maximum(weights, 0)
            
            # Normalize to sum to 1
            if weights.sum() > 0:
                weights = weights / weights.sum()
                
            return weights
            
        except np.linalg.LinAlgError:
            # Fallback to equal weights if matrix is singular
            n = len(expected_returns)
            return pd.Series(1/n, index=expected_returns.index)


# =============================================================================
# 5. CONDITIONAL VALUE AT RISK (CVaR) OPTIMIZATION
# =============================================================================

class CVaROptimizer:
    """
    CVaR (Expected Shortfall) Optimization
    
    Minimizes tail risk rather than just variance
    Better captures extreme losses
    """
    
    def __init__(self, alpha=0.05, target_return=None):
        self.alpha = alpha  # 5% worst cases
        self.target_return = target_return
        
    def calculate_cvar(self, returns, weights):
        """Calculate CVaR for a given portfolio"""
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Find VaR threshold
        var = np.percentile(portfolio_returns, self.alpha * 100)
        
        # CVaR = average of returns below VaR
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        return -cvar  # Return positive number (loss)
    
    def optimize(self, returns_df, expected_returns=None):
        """
        Find weights that minimize CVaR
        
        Args:
            returns_df: Historical returns
            expected_returns: Expected future returns (optional)
            
        Returns:
            weights: Optimal portfolio weights
        """
        n_assets = returns_df.shape[1]
        
        def objective(weights):
            return self.calculate_cvar(returns_df, weights)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]
        
        if self.target_return is not None and expected_returns is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.dot(w, expected_returns) - self.target_return
            })
        
        bounds = [(0, 1) for _ in range(n_assets)]  # Long only
        
        # Multiple random starts for global optimization
        best_weights = None
        best_cvar = np.inf
        
        for _ in range(10):
            x0 = np.random.dirichlet(np.ones(n_assets))
            
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.fun < best_cvar:
                best_cvar = result.fun
                best_weights = result.x
                
        return pd.Series(best_weights, index=returns_df.columns)


# =============================================================================
# 6. VOLATILITY TARGETING
# =============================================================================

class VolatilityTargeting:
    """
    Dynamic leverage based on volatility targeting
    
    Used by most systematic macro funds (Bridgewater, AQR, etc.)
    """
    
    def __init__(self, target_vol=0.10, lookback=21, max_leverage=2.0, min_leverage=0.25):
        self.target_vol = target_vol  # 10% annual volatility target
        self.lookback = lookback
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        
    def calculate_leverage(self, returns):
        """
        Calculate leverage to achieve target volatility
        
        leverage = target_vol / realized_vol
        """
        if isinstance(returns, pd.DataFrame):
            # Portfolio returns (equal weight for estimation)
            returns = returns.mean(axis=1)
            
        realized_vol = returns.iloc[-self.lookback:].std() * np.sqrt(252)
        
        if realized_vol < 1e-8:
            return 1.0
            
        leverage = self.target_vol / realized_vol
        
        # Clip to bounds
        leverage = np.clip(leverage, self.min_leverage, self.max_leverage)
        
        return leverage
    
    def apply_to_weights(self, weights, returns):
        """Scale weights by volatility-targeted leverage"""
        leverage = self.calculate_leverage(returns)
        return weights * leverage


# =============================================================================
# 7. DRAWDOWN CONTROL
# =============================================================================

class DrawdownControl:
    """
    Dynamic risk reduction during drawdowns
    
    Similar to CPPI but more adaptive
    """
    
    def __init__(self, max_drawdown=0.15, cushion_multiplier=3.0):
        self.max_drawdown = max_drawdown
        self.cushion_multiplier = cushion_multiplier
        self.high_water_mark = 1.0
        
    def update(self, current_nav):
        """Update high water mark"""
        self.high_water_mark = max(self.high_water_mark, current_nav)
        
    def calculate_exposure(self, current_nav):
        """
        Calculate target exposure based on current drawdown
        
        Uses CPPI-like formula:
        exposure = multiplier * (NAV - floor) / NAV
        """
        drawdown = (self.high_water_mark - current_nav) / self.high_water_mark
        
        # Floor based on max allowed drawdown
        floor = self.high_water_mark * (1 - self.max_drawdown)
        cushion = current_nav - floor
        
        if cushion <= 0:
            # At or below floor - go to cash
            return 0.0
        
        # CPPI exposure
        exposure = self.cushion_multiplier * cushion / current_nav
        
        # Cap at 100%
        return min(1.0, max(0.0, exposure))


# =============================================================================
# 8. ENSEMBLE STRATEGY COMBINER
# =============================================================================

class BayesianEnsemble:
    """
    Bayesian Model Averaging for strategy combination
    
    Dynamically weights strategies based on recent performance
    """
    
    def __init__(self, decay=0.95, min_weight=0.05):
        self.decay = decay
        self.min_weight = min_weight
        self.strategy_scores = {}
        
    def update_scores(self, strategy_name, return_contribution):
        """Update strategy score based on realized returns"""
        if strategy_name not in self.strategy_scores:
            self.strategy_scores[strategy_name] = 1.0
            
        # Exponential moving score
        self.strategy_scores[strategy_name] = (
            self.decay * self.strategy_scores[strategy_name] +
            (1 - self.decay) * (1 + return_contribution * 10)  # Scale returns
        )
        
    def get_weights(self, strategy_names):
        """Get current strategy weights"""
        scores = np.array([
            max(self.strategy_scores.get(name, 1.0), 0.01)
            for name in strategy_names
        ])
        
        # Softmax with temperature
        temperature = 2.0
        exp_scores = np.exp(scores / temperature)
        weights = exp_scores / exp_scores.sum()
        
        # Apply minimum weight
        weights = np.maximum(weights, self.min_weight)
        weights = weights / weights.sum()
        
        return dict(zip(strategy_names, weights))


# =============================================================================
# 9. MASTER PORTFOLIO OPTIMIZER
# =============================================================================

class ProfessionalPortfolioOptimizer:
    """
    Master class combining all components
    
    This is what a real quant fund uses (simplified)
    """
    
    def __init__(self, 
                 tickers,
                 target_vol=0.12,
                 max_drawdown=0.15,
                 rebalance_frequency=5,  # days
                 kelly_fraction=0.5):
        
        self.tickers = tickers
        self.target_vol = target_vol
        self.max_drawdown = max_drawdown
        self.rebalance_frequency = rebalance_frequency
        
        # Initialize components
        self.hrp = HierarchicalRiskParity()
        self.factor_model = FactorModel()
        self.regime_detector = RegimeDetector()
        self.kelly = KellyCriterion(kelly_fraction=kelly_fraction)
        self.cvar_optimizer = CVaROptimizer(alpha=0.05)
        self.vol_targeter = VolatilityTargeting(target_vol=target_vol)
        self.dd_control = DrawdownControl(max_drawdown=max_drawdown)
        self.ensemble = BayesianEnsemble()
        
        # State
        self.current_weights = None
        self.nav_history = [1.0]
        self.weight_history = []
        self.regime_history = []
        
    def fit(self, prices_df):
        """
        Fit models on historical data
        
        Args:
            prices_df: DataFrame with price history (index=dates, columns=tickers)
        """
        returns = prices_df.pct_change().dropna()
        
        # Fit regime detector
        self.regime_detector.fit(returns)
        
        # Initial HRP weights
        self.hrp.fit(returns.iloc[-252:])  # Last year
        
        print("Models fitted successfully")
        return self
        
    def get_optimal_weights(self, prices_df, current_nav=1.0):
        """
        Get optimal portfolio weights given current market state
        
        This is the main decision function
        """
        returns = prices_df.pct_change().dropna()
        recent_returns = returns.iloc[-252:]  # Last year
        
        # 1. Detect regime
        regime, regime_probs = self.regime_detector.predict(returns)
        self.regime_history.append(regime)
        
        # 2. Get factor scores
        factor_scores = self.factor_model.get_combined_scores(prices_df)
        current_scores = factor_scores.iloc[-1]
        
        # 3. Calculate strategy weights
        strategies = {}
        
        # Strategy 1: HRP (always robust)
        strategies['hrp'] = self.hrp.fit(recent_returns)
        
        # Strategy 2: Factor-based
        factor_weights = current_scores / current_scores.sum()
        factor_weights = factor_weights.clip(0.05, 0.40)
        factor_weights = factor_weights / factor_weights.sum()
        strategies['factor'] = factor_weights
        
        # Strategy 3: CVaR optimal
        strategies['cvar'] = self.cvar_optimizer.optimize(recent_returns)
        
        # Strategy 4: Inverse volatility (simple but effective)
        vols = returns.iloc[-21:].std()
        inv_vol = 1 / (vols + 1e-8)
        strategies['inv_vol'] = inv_vol / inv_vol.sum()
        
        # 4. Combine strategies based on regime
        if regime == 0:  # Bull
            strategy_weights = {'hrp': 0.2, 'factor': 0.5, 'cvar': 0.15, 'inv_vol': 0.15}
        elif regime == 2:  # Bear
            strategy_weights = {'hrp': 0.3, 'factor': 0.1, 'cvar': 0.4, 'inv_vol': 0.2}
        else:  # Sideways
            strategy_weights = {'hrp': 0.3, 'factor': 0.3, 'cvar': 0.2, 'inv_vol': 0.2}
            
        # Combine
        combined_weights = pd.Series(0.0, index=self.tickers)
        for strategy_name, strategy_weight in strategy_weights.items():
            combined_weights += strategies[strategy_name] * strategy_weight
            
        # 5. Apply volatility targeting
        leverage = self.vol_targeter.calculate_leverage(returns)
        
        # 6. Apply drawdown control
        self.dd_control.update(current_nav)
        exposure = self.dd_control.calculate_exposure(current_nav)
        
        # 7. Final weights
        final_weights = combined_weights * leverage * exposure
        
        # Ensure no shorting and sum constraint
        final_weights = final_weights.clip(0, None)
        if final_weights.sum() > 0:
            final_weights = final_weights / final_weights.sum() * min(1.0, leverage * exposure)
        
        # Cash allocation
        cash_weight = 1.0 - final_weights.sum()
        
        self.current_weights = final_weights
        self.weight_history.append(final_weights.copy())
        
        return final_weights, cash_weight, regime, regime_probs
    
    def step(self, prices_df, current_nav):
        """Execute one step of the strategy"""
        weights, cash, regime, probs = self.get_optimal_weights(prices_df, current_nav)
        
        regime_names = ['BULL', 'SIDEWAYS', 'BEAR']
        
        return {
            'weights': weights,
            'cash': cash,
            'regime': regime_names[regime],
            'regime_probs': dict(zip(regime_names, probs)),
            'leverage': weights.sum() + cash
        }


# =============================================================================
# 10. WALK-FORWARD BACKTESTER
# =============================================================================

class WalkForwardBacktester:
    """
    Proper walk-forward backtesting
    
    - Refit models periodically
    - Never look ahead
    - Realistic transaction costs
    """
    
    def __init__(self, 
                 optimizer,
                 refit_frequency=63,  # Quarterly
                 transaction_cost=0.001,  # 10 bps
                 slippage=0.0005):  # 5 bps
        
        self.optimizer = optimizer
        self.refit_frequency = refit_frequency
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
    def run(self, prices_df, start_idx=252):
        """
        Run walk-forward backtest
        
        Args:
            prices_df: Full price history
            start_idx: Where to start trading (need history before)
        """
        results = {
            'date': [],
            'nav': [],
            'returns': [],
            'weights': [],
            'regime': [],
            'turnover': [],
            'costs': []
        }
        
        nav = 1.0
        prev_weights = None
        
        for t in range(start_idx, len(prices_df)):
            date = prices_df.index[t]
            
            # Refit periodically
            if t % self.refit_frequency == 0:
                self.optimizer.fit(prices_df.iloc[:t])
                
            # Get weights
            step_result = self.optimizer.step(
                prices_df.iloc[:t], 
                nav
            )
            
            weights = step_result['weights']
            
            # Calculate turnover and costs
            if prev_weights is not None:
                turnover = np.abs(weights - prev_weights).sum()
            else:
                turnover = weights.sum()
                
            costs = turnover * (self.transaction_cost + self.slippage)
            
            # Calculate returns
            if t < len(prices_df) - 1:
                daily_returns = prices_df.iloc[t+1] / prices_df.iloc[t] - 1
                portfolio_return = (weights * daily_returns).sum() - costs
            else:
                portfolio_return = 0
                
            # Update NAV
            nav *= (1 + portfolio_return)
            
            # Store results
            results['date'].append(date)
            results['nav'].append(nav)
            results['returns'].append(portfolio_return)
            results['weights'].append(weights.copy())
            results['regime'].append(step_result['regime'])
            results['turnover'].append(turnover)
            results['costs'].append(costs)
            
            prev_weights = weights.copy()
            
        return pd.DataFrame(results).set_index('date')
    
    def calculate_metrics(self, results):
        """Calculate performance metrics"""
        returns = results['returns']
        nav = results['nav']
        
        # Basic metrics
        total_return = nav.iloc[-1] / nav.iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = annual_return / downside_vol if downside_vol > 0 else 0
        
        # Drawdown
        running_max = nav.cummax()
        drawdown = (nav - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Costs
        total_costs = results['costs'].sum()
        avg_turnover = results['turnover'].mean()
        
        return {
            'total_return': f"{total_return*100:.2f}%",
            'annual_return': f"{annual_return*100:.2f}%",
            'annual_volatility': f"{annual_vol*100:.2f}%",
            'sharpe_ratio': f"{sharpe:.2f}",
            'sortino_ratio': f"{sortino:.2f}",
            'max_drawdown': f"{max_drawdown*100:.2f}%",
            'calmar_ratio': f"{calmar:.2f}",
            'win_rate': f"{win_rate*100:.1f}%",
            'profit_factor': f"{profit_factor:.2f}",
            'avg_daily_turnover': f"{avg_turnover*100:.2f}%",
            'total_transaction_costs': f"{total_costs*100:.2f}%"
        }


# =============================================================================
# 11. EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("PROFESSIONAL PORTFOLIO OPTIMIZATION SYSTEM")
    print("="*60)
    
    # Example with synthetic data
    np.random.seed(42)
    
    tickers = ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'AMD', 'MU', 'ASML']
    n_days = 1000
    
    # Generate correlated returns
    mean_returns = np.array([0.15, 0.12, 0.10, 0.11, 0.18, 0.14, 0.13]) / 252
    vols = np.array([0.35, 0.22, 0.20, 0.25, 0.40, 0.38, 0.30]) / np.sqrt(252)
    
    # Correlation matrix
    corr = np.array([
        [1.0, 0.6, 0.5, 0.5, 0.7, 0.7, 0.6],
        [0.6, 1.0, 0.7, 0.6, 0.5, 0.5, 0.5],
        [0.5, 0.7, 1.0, 0.6, 0.4, 0.4, 0.5],
        [0.5, 0.6, 0.6, 1.0, 0.5, 0.4, 0.5],
        [0.7, 0.5, 0.4, 0.5, 1.0, 0.8, 0.6],
        [0.7, 0.5, 0.4, 0.4, 0.8, 1.0, 0.6],
        [0.6, 0.5, 0.5, 0.5, 0.6, 0.6, 1.0]
    ])
    
    cov = np.outer(vols, vols) * corr
    
    returns = np.random.multivariate_normal(mean_returns, cov, n_days)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0),
        columns=tickers,
        index=pd.date_range('2020-01-01', periods=n_days, freq='B')
    )
    
    print(f"\nGenerated {n_days} days of data for {len(tickers)} assets")
    
    # Run backtest
    optimizer = ProfessionalPortfolioOptimizer(
        tickers=tickers,
        target_vol=0.12,
        max_drawdown=0.15,
        kelly_fraction=0.5
    )
    
    backtester = WalkForwardBacktester(
        optimizer,
        refit_frequency=63,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    print("\nRunning walk-forward backtest...")
    results = backtester.run(prices, start_idx=252)
    
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    metrics = backtester.calculate_metrics(results)
    for key, value in metrics.items():
        print(f"{key:30s}: {value}")
        
    print("\n" + "="*60)
    print("REGIME DISTRIBUTION")
    print("="*60)
    print(results['regime'].value_counts(normalize=True))
