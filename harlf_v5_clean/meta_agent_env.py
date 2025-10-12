import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from utile import EarlyStoppingCallback
from production_agent_wrapper_frozen import ProductionAgentWrapper
from super_agent_env import SuperAgentEnv
from stable_baselines3 import PPO, SAC


# ============================================================================
# META AGENT ENVIRONMENT
# ============================================================================

class MetaAgentEnv(gymnasium.Env):
    """Meta Agent: Adjusts Super Agent based on market regimes"""
    
    def __init__(self, price_data, super_agent, regime_indicators=None, **config):
        super().__init__()
        
        self.price_data = price_data
        self.super_agent = super_agent
        self.regime_indicators = regime_indicators
        self.n_assets = len(price_data.columns)
        
        self.initial_capital = 100_000
        self.transaction_cost = config.get('transaction_cost', 0.002)
        self.max_position = config.get('max_position', 0.30)
        self.max_turnover = config.get('max_turnover', 0.25)
        self.constraint_penalty = config.get('constraint_penalty', 50.0)
        
        self.alpha_returns = config.get('alpha_returns', 1.0)
        self.alpha_mdd = config.get('alpha_mdd', 1.0)
        self.alpha_vol = config.get('alpha_vol', 0.5)
        self.alpha_concentration = config.get('alpha_concentration', 0.3)
        
        self.obs_scaler = None
        self.normalize = True
        
        self.common_dates = price_data.index
        if regime_indicators is not None:
            self.regime_indicators = regime_indicators.loc[self.common_dates]
        
        self.action_space = spaces.Box(
            low=0.0, high=self.max_position,
            shape=(self.n_assets,), dtype=np.float32
        )
        
        obs_dim = self.n_assets
        if regime_indicators is not None:
            obs_dim += regime_indicators.shape[1]
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.initial_capital]
        self.returns_history = []
        
        self.super_agent.reset(seed)
        
        if self.normalize and self.obs_scaler is None:
            self._fit_scaler()
        
        self._update_super_agent()
        
        return self._get_observation(), {}
    
    def _fit_scaler(self):
        obs_samples = []
        n_samples = min(len(self.common_dates), 200)
        
        for step in range(n_samples):
            self.current_step = step
            self._update_super_agent()
            obs = self._get_observation_raw()
            obs_samples.append(obs)
        
        self.obs_scaler = StandardScaler()
        self.obs_scaler.fit(obs_samples)
        self.current_step = 0
    
    def _get_observation_raw(self):
        super_weights = self.super_agent.weights
        obs = super_weights.copy()
        
        if self.regime_indicators is not None and self.current_step < len(self.common_dates):
            current_date = self.common_dates[self.current_step]
            if current_date in self.regime_indicators.index:
                regime_obs = self.regime_indicators.loc[current_date].values
                regime_obs = np.nan_to_num(regime_obs, nan=0.0)
                obs = np.concatenate([obs, regime_obs])
        
        return obs
    
    def _get_observation(self):
        obs = self._get_observation_raw()
        
        if self.normalize and self.obs_scaler is not None:
            obs = self.obs_scaler.transform(obs.reshape(1, -1))[0]
            obs = np.clip(obs, -10, 10)
        
        obs = np.nan_to_num(obs, nan=0.0)
        return obs.astype(np.float32)
    
    def _apply_constraints(self, weights):
        violation_penalty = 0.0
        
        max_weight = np.max(weights)
        if max_weight > self.max_position:
            violation_penalty += self.constraint_penalty * (max_weight - self.max_position)
        
        weights = np.clip(weights, 0, self.max_position)
        weights = weights / weights.sum()
        
        turnover = np.sum(np.abs(weights - self.weights))
        if turnover > self.max_turnover:
            violation_penalty += self.constraint_penalty * (turnover - self.max_turnover)
            scale = self.max_turnover / turnover
            weights = self.weights + scale * (weights - self.weights)
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()
        
        return weights, violation_penalty
    
    def step(self, action):
        action = np.clip(action, 0, 1)
        total = action.sum()
        weights = action / total if total > 1e-6 else np.ones(self.n_assets) / self.n_assets
        
        weights, constraint_penalty = self._apply_constraints(weights)
        
        current_date = self.common_dates[self.current_step]
        next_date = self.common_dates[min(self.current_step + 1, len(self.common_dates) - 1)]
        
        log_returns = np.log(
            self.price_data.loc[next_date] / self.price_data.loc[current_date]
        )
        log_returns = np.nan_to_num(log_returns.values, nan=0.0, posinf=0.0, neginf=0.0)
        arithmetic_returns = np.exp(log_returns) - 1
        
        portfolio_return = np.sum(weights * arithmetic_returns)
        turnover = np.sum(np.abs(weights - self.weights))
        portfolio_return -= self.transaction_cost * turnover
        
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        self.returns_history.append(portfolio_return)
        
        portfolio_series = pd.Series(self.portfolio_history)
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        current_mdd = abs(drawdown.iloc[-1])
        
        volatility = np.std(self.returns_history) if len(self.returns_history) > 1 else 0.0
        hhi = np.sum(weights ** 2)
        
        reward = (
            self.alpha_returns * np.log(1 + portfolio_return + 1e-8) -
            self.alpha_mdd * current_mdd -
            self.alpha_vol * volatility -
            self.alpha_concentration * hhi -
            constraint_penalty
        )
        
        self.weights = weights
        self.current_step += 1
        done = self.current_step >= len(self.common_dates) - 1
        
        if not done:
            self._update_super_agent()
        
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": weights,
            "turnover": turnover,
            "hhi": hhi
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _update_super_agent(self):
        if self.current_step >= len(self.common_dates):
            return
        
        current_date = self.common_dates[self.current_step]
        super_obs = self.super_agent.get_observation(current_date)
        self.super_agent.predict(super_obs, deterministic=True)
    
    def get_portfolio_metrics(self):
        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        sharpe_ratio = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(12)
        
        max_drawdown = 0.0
        if len(portfolio_series) > 0:
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak
            max_drawdown = abs(drawdown.min())
        
        volatility = returns.std() * np.sqrt(12) if len(returns) > 0 else 0.0
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "final_value": self.portfolio_value,
            "win_rate": win_rate
        }





def train_meta_agent(train_data, val_data, super_model, config):
    """
    Train Meta agent
    SIMPLIFIED: Single training run (no windows, no ensemble)
    """
    
    print("\n" + "="*70)
    print("TRAINING META AGENT")
    print("="*70)
    
    train_prices, train_tech, train_sent, train_regime = train_data
    val_prices, val_tech, val_sent, val_regime = val_data
    
    n_assets = len(train_prices.columns)
    
    # Load base agents
    print("\n   Loading base agents...")
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
    print("\n   Creating Super agent environments...")
    super_train_env = SuperAgentEnv(
        train_prices, base_agents_train, train_tech, train_sent, **config
    )
    super_val_env = SuperAgentEnv(
        val_prices, base_agents_val, val_tech, val_sent, **config
    )
    
    # Wrap Super model
    super_train_wrap = ProductionAgentWrapper(
        super_model, super_train_env, n_assets, 'super', freeze_weights=True
    )
    super_val_wrap = ProductionAgentWrapper(
        super_model, super_val_env, n_assets, 'super', freeze_weights=True
    )
    
    # Create Meta environments
    print("   Creating Meta agent environments...")
    meta_train_env = MetaAgentEnv(train_prices, super_train_wrap, train_regime, **config)
    meta_val_env = MetaAgentEnv(val_prices, super_val_wrap, val_regime, **config)
    
    # Train Meta with PPO
    print(f"\n   Training with PPO algorithm...")
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
        save_path='models/meta_agent',
        verbose=1
    )
    
    print(f"\n   Training for up to {config['meta_timesteps']} steps...")
    model.learn(total_timesteps=config['meta_timesteps'], callback=callback)
    
    if callback.best_model_path:
        model = PPO.load(callback.best_model_path, env=meta_train_env)
        print(f"\n   ✓ Loaded best model (Val Sharpe: {callback.best_val_sharpe:.3f})")
    
    return model, callback.best_val_sharpe