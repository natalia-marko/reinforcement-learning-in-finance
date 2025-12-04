"""
🚀 ULTIMATE PORTFOLIO SYSTEM - MOON EDITION 🚀
==============================================

Combines the best of BOTH worlds:
1. RL "Board of Directors" ensemble (Bull, Bear, Sniper, Alpha)
2. Traditional quant methods (HRP, Factor Models, CVaR)
3. Regime detection with dynamic weighting
4. Professional risk management

This is what a real quant fund looks like (simplified).

Usage:
    python moon_portfolio.py

Author: Professional Quant System
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Import our professional systems
from professional_portfolio_system import (
    HierarchicalRiskParity,
    FactorModel,
    RegimeDetector,
    KellyCriterion,
    CVaROptimizer,
    VolatilityTargeting,
    DrawdownControl
)

from enhanced_rl_system import (
    EnhancedPortfolioEnv,
    BullAgent,
    BearAgent,
    SniperAgent,
    AlphaAgent,
    EnhancedBoardOfDirectors,
    train_specialist_agents
)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration for the entire system"""
    
    # Assets
    TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']
    
    # Risk Management
    TARGET_VOLATILITY = 0.12       # 12% annual volatility target
    MAX_DRAWDOWN = 0.15            # 15% max drawdown before reducing exposure
    MAX_POSITION = 0.35            # 35% max single position
    MAX_LEVERAGE = 1.5             # 150% max gross exposure
    
    # Costs
    TRANSACTION_COST = 0.001       # 10 bps
    SLIPPAGE = 0.0005              # 5 bps
    
    # RL Training
    TOTAL_TIMESTEPS = 100000
    REBALANCE_PERIOD = 4           # Weekly (4 = monthly)
    
    # Ensemble Weights (base)
    RL_WEIGHT = 0.60               # 60% RL ensemble
    QUANT_WEIGHT = 0.40            # 40% traditional quant
    
    # Paths
    MODELS_DIR = 'models'
    DATA_DIR = 'data'
    OUTPUT_DIR = 'outputs'


# =============================================================================
# HYBRID STRATEGY: RL + QUANT
# =============================================================================

class MoonPortfolioStrategy:
    """
    The ULTIMATE portfolio strategy combining:
    
    1. RL Ensemble (60%):
       - Bull Agent: Aggressive momentum following
       - Bear Agent: Defensive tail-risk focused
       - Sniper Agent: Balanced Sharpe optimization
       - Alpha Agent: Benchmark-relative performance
       
    2. Quant Overlay (40%):
       - HRP: Hierarchical Risk Parity for diversification
       - Factor Model: Momentum + Value + Quality + Low Vol
       - CVaR Optimizer: Tail risk minimization
       
    3. Risk Management:
       - Volatility targeting
       - Drawdown control
       - Kelly-based position sizing
       - Regime-aware adjustments
    """
    
    def __init__(self, config=Config):
        self.config = config
        self.tickers = config.TICKERS
        self.n_assets = len(self.tickers)
        
        # Initialize quant components
        self.hrp = HierarchicalRiskParity()
        self.factor_model = FactorModel()
        self.regime_detector = RegimeDetector()
        self.kelly = KellyCriterion(kelly_fraction=0.5)
        self.cvar = CVaROptimizer(alpha=0.05)
        self.vol_targeter = VolatilityTargeting(target_vol=config.TARGET_VOLATILITY)
        self.dd_control = DrawdownControl(max_drawdown=config.MAX_DRAWDOWN)
        
        # RL ensemble (will be loaded/trained)
        self.rl_ensemble = None
        
        # State
        self.current_weights = None
        self.nav = 1.0
        self.nav_history = [1.0]
        self.returns_history = []
        self.regime_history = []
        
        # Performance tracking
        self.performance = {
            'returns': [],
            'sharpe_rolling': [],
            'drawdown': [],
            'turnover': [],
            'regime': []
        }
        
    def fit(self, prices_df, features=None, benchmark=None):
        """
        Fit all models on historical data
        
        Args:
            prices_df: DataFrame of prices (dates x tickers)
            features: Optional pre-computed features
            benchmark: Optional benchmark series
        """
        print("="*60)
        print("FITTING MOON PORTFOLIO STRATEGY")
        print("="*60)
        
        returns = prices_df.pct_change().dropna()
        
        # 1. Fit regime detector
        print("\n1. Fitting regime detector...")
        self.regime_detector.fit(returns)
        
        # 2. Fit HRP
        print("2. Fitting Hierarchical Risk Parity...")
        self.hrp.fit(returns.iloc[-252:])
        
        # 3. Train RL agents if not already trained
        if self.rl_ensemble is None:
            print("\n3. Training RL specialist agents...")
            self._train_rl_agents(prices_df, features, benchmark)
        
        print("\n✅ All models fitted successfully!")
        return self
    
    def _train_rl_agents(self, prices_df, features=None, benchmark=None):
        """Train the RL ensemble"""
        
        # Convert to numpy if needed
        if isinstance(prices_df, pd.DataFrame):
            prices = prices_df.values
        else:
            prices = prices_df
            
        # Create dummy features if not provided
        if features is None:
            n_days = len(prices)
            n_assets = prices.shape[1]
            features = np.random.randn(n_days, n_assets, 3)  # Placeholder
            
        # Create benchmark if not provided
        if benchmark is None:
            benchmark = np.mean(prices, axis=1)
        elif isinstance(benchmark, pd.Series):
            benchmark = benchmark.values
            
        # Train agents
        train_specialist_agents(
            prices, features, benchmark,
            output_dir=self.config.MODELS_DIR,
            total_timesteps=self.config.TOTAL_TIMESTEPS
        )
        
        # Load ensemble
        self.rl_ensemble = EnhancedBoardOfDirectors(
            bull_model_path=os.path.join(self.config.MODELS_DIR, 'agent_bull'),
            bear_model_path=os.path.join(self.config.MODELS_DIR, 'agent_bear'),
            sniper_model_path=os.path.join(self.config.MODELS_DIR, 'agent_sniper'),
            alpha_model_path=os.path.join(self.config.MODELS_DIR, 'agent_alpha')
        )
    
    def get_optimal_weights(self, prices_df, current_obs=None):
        """
        Get optimal portfolio weights combining RL and quant methods
        
        Args:
            prices_df: Recent price history
            current_obs: Current observation for RL models
            
        Returns:
            weights: Portfolio weights
            info: Dict with diagnostics
        """
        returns = prices_df.pct_change().dropna()
        recent_returns = returns.iloc[-63:]  # Last quarter
        
        # 1. REGIME DETECTION
        regime, regime_probs = self.regime_detector.predict(returns)
        regime_names = ['BULL', 'SIDEWAYS', 'BEAR']
        current_regime = regime_names[regime]
        self.regime_history.append(current_regime)
        
        # 2. QUANT STRATEGIES
        quant_strategies = {}
        
        # HRP weights
        quant_strategies['hrp'] = self.hrp.fit(recent_returns)
        
        # Factor-based weights
        factor_scores = self.factor_model.get_combined_scores(prices_df)
        current_scores = factor_scores.iloc[-1]
        factor_weights = current_scores / current_scores.sum()
        factor_weights = factor_weights.clip(0.05, 0.35)
        quant_strategies['factor'] = factor_weights / factor_weights.sum()
        
        # CVaR optimal
        quant_strategies['cvar'] = self.cvar.optimize(recent_returns)
        
        # Inverse volatility
        vols = returns.iloc[-21:].std()
        inv_vol = 1 / (vols + 1e-8)
        quant_strategies['inv_vol'] = inv_vol / inv_vol.sum()
        
        # Combine quant strategies based on regime
        if current_regime == 'BULL':
            quant_blend = {'hrp': 0.2, 'factor': 0.5, 'cvar': 0.1, 'inv_vol': 0.2}
        elif current_regime == 'BEAR':
            quant_blend = {'hrp': 0.3, 'factor': 0.1, 'cvar': 0.4, 'inv_vol': 0.2}
        else:
            quant_blend = {'hrp': 0.3, 'factor': 0.3, 'cvar': 0.2, 'inv_vol': 0.2}
        
        quant_weights = pd.Series(0.0, index=self.tickers)
        for strat, blend_weight in quant_blend.items():
            quant_weights += quant_strategies[strat].reindex(self.tickers, fill_value=1/self.n_assets) * blend_weight
        
        # 3. RL ENSEMBLE (if available)
        if self.rl_ensemble is not None and current_obs is not None:
            rl_action, rl_info = self.rl_ensemble.get_action(
                current_obs, 
                recent_returns=self.returns_history
            )
            
            # Convert RL action to weights
            exp_action = np.exp(rl_action - np.max(rl_action))
            rl_weights = pd.Series(
                exp_action / exp_action.sum(),
                index=self.tickers
            )
        else:
            rl_weights = quant_weights.copy()  # Fallback
            rl_info = {'regime': current_regime}
        
        # 4. COMBINE RL + QUANT
        combined_weights = (
            self.config.RL_WEIGHT * rl_weights + 
            self.config.QUANT_WEIGHT * quant_weights
        )
        
        # 5. APPLY RISK MANAGEMENT
        
        # Volatility targeting
        leverage = self.vol_targeter.calculate_leverage(returns)
        
        # Drawdown control
        self.dd_control.update(self.nav)
        exposure = self.dd_control.calculate_exposure(self.nav)
        
        # Apply leverage and exposure
        final_weights = combined_weights * min(leverage, self.config.MAX_LEVERAGE) * exposure
        
        # Apply constraints
        final_weights = final_weights.clip(0, self.config.MAX_POSITION)
        
        # Normalize
        if final_weights.sum() > 0:
            total_exposure = min(final_weights.sum(), 1.0)
            final_weights = final_weights / final_weights.sum() * total_exposure
        
        cash_weight = 1.0 - final_weights.sum()
        
        self.current_weights = final_weights
        
        info = {
            'regime': current_regime,
            'regime_probs': dict(zip(regime_names, regime_probs)),
            'leverage': leverage,
            'exposure': exposure,
            'quant_weights': quant_weights,
            'rl_weights': rl_weights,
            'final_weights': final_weights,
            'cash': cash_weight
        }
        
        return final_weights, info
    
    def step(self, new_prices):
        """
        Execute one time step
        
        Args:
            new_prices: New price data
            
        Returns:
            portfolio_return: Return for this period
            info: Diagnostics
        """
        if self.current_weights is None:
            return 0.0, {}
        
        # Calculate return
        if isinstance(new_prices, pd.Series):
            returns = new_prices / new_prices.shift(1) - 1
        else:
            returns = new_prices
            
        portfolio_return = (self.current_weights * returns).sum()
        
        # Update NAV
        self.nav *= (1 + portfolio_return)
        self.nav_history.append(self.nav)
        self.returns_history.append(portfolio_return)
        
        # Track performance
        self.performance['returns'].append(portfolio_return)
        self.performance['drawdown'].append(self._calculate_drawdown())
        
        return portfolio_return, {'nav': self.nav, 'drawdown': self.performance['drawdown'][-1]}
    
    def _calculate_drawdown(self):
        """Calculate current drawdown"""
        if len(self.nav_history) < 2:
            return 0.0
        nav = np.array(self.nav_history)
        running_max = np.maximum.accumulate(nav)
        return float((nav[-1] - running_max[-1]) / running_max[-1])
    
    def get_performance_summary(self):
        """Get comprehensive performance metrics"""
        if len(self.returns_history) < 10:
            return {}
        
        returns = np.array(self.returns_history)
        nav = np.array(self.nav_history)
        
        # Calculate metrics
        total_return = nav[-1] / nav[0] - 1
        n_periods = len(returns)
        annual_factor = 52 / max(n_periods, 1) * len(returns)  # Assuming weekly
        annual_return = (1 + total_return) ** (52 / n_periods) - 1 if n_periods > 0 else 0
        annual_vol = np.std(returns) * np.sqrt(52)
        
        # Sharpe
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Sortino
        downside = returns[returns < 0]
        downside_vol = np.std(downside) * np.sqrt(52) if len(downside) > 0 else 0.01
        sortino = annual_return / downside_vol
        
        # Max drawdown
        running_max = np.maximum.accumulate(nav)
        drawdowns = (nav - running_max) / running_max
        max_dd = np.min(drawdowns)
        
        # Calmar
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Win rate
        win_rate = np.mean(returns > 0)
        
        # Regime breakdown
        if self.regime_history:
            from collections import Counter
            regime_counts = Counter(self.regime_history)
            total = len(self.regime_history)
            regime_pct = {k: v/total for k, v in regime_counts.items()}
        else:
            regime_pct = {}
        
        return {
            'total_return': f"{total_return*100:.2f}%",
            'annual_return': f"{annual_return*100:.2f}%",
            'annual_volatility': f"{annual_vol*100:.2f}%",
            'sharpe_ratio': f"{sharpe:.2f}",
            'sortino_ratio': f"{sortino:.2f}",
            'max_drawdown': f"{max_dd*100:.2f}%",
            'calmar_ratio': f"{calmar:.2f}",
            'win_rate': f"{win_rate*100:.1f}%",
            'n_periods': n_periods,
            'regime_distribution': regime_pct,
            'final_nav': f"${nav[-1]*100:.2f}"
        }


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class MoonBacktester:
    """Walk-forward backtester for the Moon Portfolio Strategy"""
    
    def __init__(self, 
                 strategy,
                 refit_frequency=63,
                 transaction_cost=0.001):
        
        self.strategy = strategy
        self.refit_frequency = refit_frequency
        self.transaction_cost = transaction_cost
        
    def run(self, prices_df, features=None, benchmark=None, start_idx=252):
        """
        Run full backtest
        
        Args:
            prices_df: Full price history
            features: Optional features array
            benchmark: Optional benchmark series
            start_idx: Warmup period
            
        Returns:
            results: DataFrame with daily results
        """
        print("="*60)
        print("🚀 RUNNING MOON PORTFOLIO BACKTEST 🚀")
        print("="*60)
        
        results = []
        prev_weights = None
        
        for t in range(start_idx, len(prices_df)):
            date = prices_df.index[t]
            
            # Refit periodically
            if t % self.refit_frequency == 0 or t == start_idx:
                print(f"\n📊 Refitting models at {date}...")
                self.strategy.fit(prices_df.iloc[:t], features[:t] if features is not None else None)
            
            # Create observation for RL
            current_obs = self._create_observation(prices_df.iloc[:t], features[:t] if features is not None else None)
            
            # Get optimal weights
            weights, info = self.strategy.get_optimal_weights(prices_df.iloc[:t], current_obs)
            
            # Calculate turnover and costs
            if prev_weights is not None:
                turnover = np.abs(weights - prev_weights).sum()
            else:
                turnover = weights.sum()
            
            costs = turnover * self.transaction_cost
            
            # Calculate returns (next period)
            if t < len(prices_df) - 1:
                next_returns = prices_df.iloc[t+1] / prices_df.iloc[t] - 1
                portfolio_return = (weights * next_returns).sum() - costs
                self.strategy.step(next_returns)
            else:
                portfolio_return = 0
            
            results.append({
                'date': date,
                'return': portfolio_return,
                'nav': self.strategy.nav,
                'regime': info['regime'],
                'leverage': info['leverage'],
                'exposure': info['exposure'],
                'cash': info['cash'],
                'turnover': turnover,
                'costs': costs
            })
            
            prev_weights = weights.copy()
            
            # Progress update
            if t % 50 == 0:
                print(f"  Step {t}/{len(prices_df)} - NAV: ${self.strategy.nav*100:.2f} - Regime: {info['regime']}")
        
        results_df = pd.DataFrame(results).set_index('date')
        
        # Print summary
        print("\n" + "="*60)
        print("📈 BACKTEST RESULTS")
        print("="*60)
        
        summary = self.strategy.get_performance_summary()
        for key, value in summary.items():
            if key != 'regime_distribution':
                print(f"  {key:25s}: {value}")
        
        print("\n  Regime Distribution:")
        for regime, pct in summary.get('regime_distribution', {}).items():
            print(f"    {regime:15s}: {pct*100:.1f}%")
        
        return results_df
    
    def _create_observation(self, prices_df, features=None):
        """Create observation vector for RL"""
        n_assets = len(prices_df.columns)
        
        if features is not None and len(features) > 0:
            obs = features[-1].flatten()
        else:
            # Create basic features from prices
            returns = prices_df.pct_change().iloc[-21:]
            
            obs = np.concatenate([
                returns.iloc[-1].values,  # Last return
                returns.mean().values,    # Mean return
                returns.std().values,     # Volatility
            ])
        
        # Add current weights and market state
        current_weights = self.strategy.current_weights if self.strategy.current_weights is not None else np.ones(n_assets) / n_assets
        if isinstance(current_weights, pd.Series):
            current_weights = current_weights.values
            
        market_state = np.array([
            0.0,  # Placeholder
            0.0,
            0.0,
            0.0,
            0.0
        ])
        
        obs = np.concatenate([obs, current_weights, market_state]).astype(np.float32)
        
        return obs


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    🚀🌙 MOON PORTFOLIO SYSTEM 🌙🚀
    ================================
    
    Combining:
    - RL "Board of Directors" Ensemble
    - Hierarchical Risk Parity
    - Multi-Factor Alpha Model
    - CVaR Tail Risk Management
    - Regime-Aware Allocation
    - Volatility Targeting
    - Drawdown Control
    
    Let's go to the moon! 🚀
    """)
    
    # Create synthetic data for demo
    np.random.seed(42)
    
    tickers = Config.TICKERS
    n_days = 500
    
    # Generate correlated returns
    mean_returns = np.array([0.20, 0.15, 0.12, 0.18, 0.16, 0.11, 0.10]) / 252
    vols = np.array([0.40, 0.35, 0.22, 0.45, 0.38, 0.20, 0.23]) / np.sqrt(252)
    
    # Correlation matrix (tech stocks are correlated)
    corr = np.array([
        [1.0, 0.7, 0.5, 0.8, 0.7, 0.5, 0.5],
        [0.7, 1.0, 0.5, 0.7, 0.8, 0.5, 0.5],
        [0.5, 0.5, 1.0, 0.5, 0.5, 0.7, 0.6],
        [0.8, 0.7, 0.5, 1.0, 0.8, 0.5, 0.5],
        [0.7, 0.8, 0.5, 0.8, 1.0, 0.5, 0.5],
        [0.5, 0.5, 0.7, 0.5, 0.5, 1.0, 0.7],
        [0.5, 0.5, 0.6, 0.5, 0.5, 0.7, 1.0]
    ])
    
    cov = np.outer(vols, vols) * corr
    
    returns = np.random.multivariate_normal(mean_returns, cov, n_days)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0),
        columns=tickers,
        index=pd.date_range('2020-01-01', periods=n_days, freq='B')
    )
    
    print(f"Generated {n_days} days of data for {len(tickers)} assets\n")
    
    # Initialize strategy
    strategy = MoonPortfolioStrategy(Config)
    
    # Run backtest
    backtester = MoonBacktester(strategy, refit_frequency=63)
    
    # Note: For full functionality, you need to train the RL agents first
    # This demo shows the quant-only version
    print("⚠️  Running in QUANT-ONLY mode (no RL agents trained)")
    print("   To enable RL, train agents first with train_specialist_agents()")
    
    # Simplified backtest without RL
    strategy.fit(prices)
    
    print("\n✅ Strategy fitted successfully!")
    print("\nTo run full backtest with RL + Quant:")
    print("  results = backtester.run(prices)")
    print("\nTo train RL agents:")
    print("  from enhanced_rl_system import train_specialist_agents")
    print("  train_specialist_agents(prices.values, features, benchmark)")
