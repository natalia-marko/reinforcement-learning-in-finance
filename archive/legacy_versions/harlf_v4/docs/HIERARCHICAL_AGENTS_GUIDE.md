# Complete Guide: Super Agent and Meta Agent Training

**Project:** HARLF v4 - Hierarchical Reinforcement Learning
**Focus:** Training Super Agent (combines base agents) and Meta Agent (top-level coordinator)
**Date:** October 8, 2025

---

## TABLE OF CONTENTS

1. Understanding Hierarchical Architecture
2. Super Agent: Complete Implementation
3. Meta Agent: Complete Implementation
4. Hyperparameter Tuning for Hierarchical Agents
5. Troubleshooting Common Issues
6. Expected Results and Benchmarks

---

## PART 1: UNDERSTANDING HIERARCHICAL ARCHITECTURE

### The HARLF Hierarchy:

```
┌─────────────────────────────────────────────────────────────┐
│                         META AGENT                           │
│           (Top-level coordinator - final decision)           │
│                                                              │
│  Observes: All features + all agent recommendations         │
│  Decides: Final portfolio allocation                         │
└──────────────────────┬───────────────────────────────────────┘
                       │ Coordinates
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                        SUPER AGENT                           │
│            (Strategic blender of base agents)                │
│                                                              │
│  Observes: Base agent recommendations + regime indicators   │
│  Decides: How to blend base agent strategies                │
└──────────────────────┬──────────────────────────────────────┘
                       │ Blends
                       ↓
        ┌──────────────┴──────────────┐
        │                             │
┌───────┴────────┐           ┌────────┴────────┐
│ SENTIMENT AGENT│           │ TECHNICAL AGENT │
│   (Base Agent) │           │   (Base Agent)  │
│                │           │                 │
│ Observes: NLP  │           │ Observes: TA    │
│ features       │           │ indicators      │
└────────────────┘           └─────────────────┘
```

### Key Principles:

**1. Sequential Training (Bottom-Up):**
```
Step 1: Train Sentiment + Technical Agents (DONE ✓)
        ↓
Step 2: Freeze base agents → Train Super Agent
        ↓
Step 3: Freeze all → Train Meta Agent
```

**2. Information Flow:**
```
Base Agents:
  - Specialized feature extraction
  - Domain-specific signals
  - Narrow focus, deep expertise

Super Agent:
  - Learns when to trust which base agent
  - Regime-aware blending
  - "Manager" level decision

Meta Agent:
  - Portfolio-level optimization
  - Risk management
  - "Executive" level decision
```

**3. Why Hierarchical?**
- **Specialization:** Each agent focuses on different aspects
- **Robustness:** Ensemble effect reduces individual agent failures
- **Interpretability:** Can analyze each level's decisions
- **Scalability:** Easy to add more base agents

---

## PART 2: SUPER AGENT - COMPLETE IMPLEMENTATION

### What Super Agent Does:

**Role:** Learns to optimally blend recommendations from Sentiment and Technical agents.

**Input:** 
- Sentiment agent's portfolio weights (10 numbers)
- Technical agent's portfolio weights (10 numbers)
- Regime indicators (10 binary values) [if enabled]
- Total observation: 20 or 30 dimensions

**Output:**
- Final blended portfolio weights (10 numbers)

**Learning Objective:**
- Maximize risk-adjusted returns
- Learn when to trust sentiment vs technical
- Adapt strategy based on market regime

---

### Step 1: Prepare Base Agents

After training base agents (your Cell 7), you need to select the best ones:

```python
# Cell 11 Content - Part 1: Select Best Base Agents
```markdown
## Hierarchical Training: Super Agent

Super Agent learns to blend Sentiment and Technical agents' recommendations
```

```python
from super_agent_envoriment import SuperAgentEnv
from agent_wrapper import AgentWrapper, train_agent, evaluate_agent
import numpy as np

print("="*70)
print("STEP 1: PREPARING BASE AGENTS FOR HIERARCHICAL TRAINING")
print("="*70)

# Select best agents from each type based on test performance
sentiment_agents = {
    'ppo': results['sentiment_ppo'],
    'sac': results['sentiment_sac']
}

technical_agents = {
    'ppo': results['technical_ppo'],
    'sac': results['technical_sac']
}

# Find best of each type
best_sentiment_key = max(sentiment_agents.keys(), 
                         key=lambda k: sentiment_agents[k]['test_metrics']['sharpe_ratio'])
best_technical_key = max(technical_agents.keys(),
                         key=lambda k: technical_agents[k]['test_metrics']['sharpe_ratio'])

sentiment_best = sentiment_agents[best_sentiment_key]
technical_best = technical_agents[best_technical_key]

print(f"\nSelected Base Agents:")
print(f"  Sentiment: {best_sentiment_key.upper()}")
print(f"    - Train Sharpe: {sentiment_best['train_metrics']['sharpe_ratio']:.3f}")
print(f"    - Val Sharpe:   {sentiment_best['val_metrics']['sharpe_ratio']:.3f}")
print(f"    - Test Sharpe:  {sentiment_best['test_metrics']['sharpe_ratio']:.3f}")

print(f"\n  Technical: {best_technical_key.upper()}")
print(f"    - Train Sharpe: {technical_best['train_metrics']['sharpe_ratio']:.3f}")
print(f"    - Val Sharpe:   {technical_best['val_metrics']['sharpe_ratio']:.3f}")
print(f"    - Test Sharpe:  {technical_best['test_metrics']['sharpe_ratio']:.3f}")
```

---

### Step 2: Create Agent Wrappers

Base agents need to be wrapped to be compatible with hierarchical envs:

```python
# Cell 11 Content - Part 2: Create Wrappers

print("\n" + "="*70)
print("STEP 2: CREATING AGENT WRAPPERS")
print("="*70)

# Create wrapped agents with proper interface
class HierarchicalAgentWrapper:
    """
    Wrapper that makes trained agents compatible with hierarchical environments.
    Provides .predict(), .weights, and .reset() interface.
    """
    def __init__(self, model, env, agent_type):
        self.model = model
        self.env = env
        self.agent_type = agent_type
        self.n_assets = len(env.price_data.columns)
        self.weights = np.ones(self.n_assets) / self.n_assets
    
    def predict(self, obs, deterministic=True):
        """Get action from trained model"""
        action, state = self.model.predict(obs, deterministic=deterministic)
        action = np.clip(action, 0, 1)
        total = action.sum()
        if total > 1e-6:
            self.weights = action / total
        else:
            self.weights = np.ones(self.n_assets) / self.n_assets
        return action, state
    
    def reset(self, seed=None):
        """Reset weights to equal allocation"""
        self.weights = np.ones(self.n_assets) / self.n_assets
        return None

# Wrap the selected best agents
sentiment_wrapped = HierarchicalAgentWrapper(
    sentiment_best['model'],
    sentiment_best['train_env'],
    'sentiment'
)

technical_wrapped = HierarchicalAgentWrapper(
    technical_best['model'],
    technical_best['train_env'],
    'technical'
)

print("Agent wrappers created successfully")
print(f"  Sentiment wrapper: {sentiment_wrapped.agent_type}")
print(f"  Technical wrapper: {technical_wrapped.agent_type}")
print(f"  Assets: {sentiment_wrapped.n_assets}")
```

---

### Step 3: Create Super Agent Environments

```python
# Cell 11 Content - Part 3: Create Environments

print("\n" + "="*70)
print("STEP 3: CREATING SUPER AGENT ENVIRONMENTS")
print("="*70)

# Get data splits (already done in train_and_evaluate_with_split)
from custom_function import split_data_chronologically

splits = split_data_chronologically(
    price_data, technical_features, sentiment_features,
    training_config['train_ratio'], training_config['val_ratio']
)

train_prices, train_technical, train_sentiment = splits['train']
val_prices, val_technical, val_sentiment = splits['val']
test_prices, test_technical, test_sentiment = splits['test']

# Get regime indicators for each split
if regime_config['enabled']:
    train_regime = regime_indicators.loc[train_prices.index]
    val_regime = regime_indicators.loc[val_prices.index]
    test_regime = regime_indicators.loc[test_prices.index]
else:
    train_regime = None
    val_regime = None
    test_regime = None

# Create Super Agent environments
super_train_env = SuperAgentEnv(
    price_data=train_prices,
    sentiment_agent=sentiment_wrapped,
    technical_agent=technical_wrapped,
    regime_indicators=train_regime,
    initial_capital=100000,
    **super_agent_config
)

super_val_env = SuperAgentEnv(
    price_data=val_prices,
    sentiment_agent=sentiment_wrapped,
    technical_agent=technical_wrapped,
    regime_indicators=val_regime,
    initial_capital=100000,
    **super_agent_config
)

super_test_env = SuperAgentEnv(
    price_data=test_prices,
    sentiment_agent=sentiment_wrapped,
    technical_agent=technical_wrapped,
    regime_indicators=test_regime,
    initial_capital=100000,
    **super_agent_config
)

print("Super Agent environments created:")
print(f"  Train: {len(train_prices)} months")
print(f"  Val:   {len(val_prices)} months")
print(f"  Test:  {len(test_prices)} months")
print(f"  Regime indicators: {'Enabled' if regime_config['enabled'] else 'Disabled'}")
```

---

### Step 4: Train Super Agent

```python
# Cell 11 Content - Part 4: Train Super Agent

print("\n" + "="*70)
print("STEP 4: TRAINING SUPER AGENT")
print("="*70)

# Super Agent typically works better with PPO (more stable for hierarchical tasks)
print("\nTraining Super Agent with PPO...")

# Use adjusted hyperparameters for hierarchical agent
super_ppo_config = {
    'learning_rate': 1e-4,      # Lower LR (hierarchical agents need careful updates)
    'n_steps': 1024,            # Smaller n_steps (fewer samples, shorter episodes)
    'batch_size': 128,          # Larger batch for stability
    'n_epochs': 3,              # Few epochs (avoid overfitting to blending strategy)
    'ent_coef': 0.1,            # High entropy (explore different blending strategies)
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'clip_range': 0.2,
}

super_model = PPO("MlpPolicy", super_train_env, **super_ppo_config, verbose=1)

# Train with early stopping
from agent_wrapper import EarlyStoppingCallback

super_callback = EarlyStoppingCallback(
    super_val_env, 
    eval_freq=2000,     # Evaluate more frequently (smaller dataset)
    patience=5,
    verbose=1
)

# Reduced timesteps for hierarchical agent (smaller observation space)
super_timesteps = 50000  # Less than base agents

super_model.learn(
    total_timesteps=super_timesteps,
    callback=super_callback
)

print("\nSuper Agent training completed")
```

---

### Step 5: Evaluate Super Agent

```python
# Cell 11 Content - Part 5: Evaluate Super Agent

print("\n" + "="*70)
print("STEP 5: EVALUATING SUPER AGENT")
print("="*70)

# Evaluate on all sets
super_train_metrics = evaluate_agent(super_model, super_train_env, "Super Agent - Training")
super_val_metrics = evaluate_agent(super_model, super_val_env, "Super Agent - Validation")
super_test_metrics = evaluate_agent(super_model, super_test_env, "Super Agent - Test")

# Compare to base agents
print("\n" + "="*70)
print("SUPER AGENT VS BASE AGENTS COMPARISON")
print("="*70)

comparison = {
    'Sentiment (Best)': sentiment_best['test_metrics']['sharpe_ratio'],
    'Technical (Best)': technical_best['test_metrics']['sharpe_ratio'],
    'Super Agent': super_test_metrics['sharpe_ratio']
}

for name, sharpe in comparison.items():
    print(f"{name:20s}: {sharpe:.3f}")

best_base = max(sentiment_best['test_metrics']['sharpe_ratio'],
                technical_best['test_metrics']['sharpe_ratio'])
improvement = (super_test_metrics['sharpe_ratio'] - best_base) / best_base * 100

print(f"\nSuper Agent improvement over best base: {improvement:+.1f}%")

if improvement > 0:
    print("SUCCESS: Hierarchical combination adds value!")
elif improvement > -5:
    print("NEUTRAL: Hierarchical combination maintains performance")
else:
    print("WARNING: Hierarchical combination underperforms - check configuration")

# Save Super Agent
super_model.save('./models/super_agent_ppo')
print("\nSuper Agent saved: ./models/super_agent_ppo.zip")

# Store results
results['super_ppo'] = {
    'model': super_model,
    'train_metrics': super_train_metrics,
    'val_metrics': super_val_metrics,
    'test_metrics': super_test_metrics,
    'train_env': super_train_env,
    'val_env': super_val_env,
    'test_env': super_test_env,
    'agent_type': 'hierarchical',
    'level': 'super'
}
```

---

## PART 3: META AGENT - COMPLETE IMPLEMENTATION

### What Meta Agent Does:

**Role:** Top-level coordinator with access to ALL information.

**Input:**
- All technical features (110 dimensions)
- All sentiment features (53 dimensions)
- Sentiment agent weights (10 dimensions)
- Technical agent weights (10 dimensions)
- Super agent weights (10 dimensions)
- Regime indicators (10 dimensions) [if enabled]
- Total observation: 193 or 203 dimensions

**Output:**
- Final portfolio allocation (10 dimensions)

**Learning Objective:**
- Maximize portfolio-level risk-adjusted returns
- Consider all available information
- Override lower-level agents when beneficial

---

### Complete Meta Agent Implementation:

```python
# Cell 12 Content - Complete Meta Agent Training

```markdown
## Hierarchical Training: Meta Agent

Meta Agent is the top-level coordinator that makes final portfolio decisions
```

```python
from meta_agent_enviroment import MetaAgentEnv

print("="*70)
print("META AGENT TRAINING")
print("="*70)

print("\n" + "="*70)
print("STEP 1: WRAPPING SUPER AGENT")
print("="*70)

# Wrap the trained super agent
super_wrapped = HierarchicalAgentWrapper(
    super_model,
    super_train_env,
    'super'
)

print("Super Agent wrapped for Meta Agent training")

# Create Meta Agent environments
print("\n" + "="*70)
print("STEP 2: CREATING META AGENT ENVIRONMENTS")
print("="*70)

meta_train_env = MetaAgentEnv(
    price_data=train_prices,
    features=train_technical,
    sentiment_features=train_sentiment,
    sentiment_agent=sentiment_wrapped,
    technical_agent=technical_wrapped,
    super_agent=super_wrapped,
    regime_indicators=train_regime,
    initial_capital=100000,
    **meta_agent_config
)

meta_val_env = MetaAgentEnv(
    price_data=val_prices,
    features=val_technical,
    sentiment_features=val_sentiment,
    sentiment_agent=sentiment_wrapped,
    technical_agent=technical_wrapped,
    super_agent=super_wrapped,
    regime_indicators=val_regime,
    initial_capital=100000,
    **meta_agent_config
)

meta_test_env = MetaAgentEnv(
    price_data=test_prices,
    features=test_technical,
    sentiment_features=test_sentiment,
    sentiment_agent=sentiment_wrapped,
    technical_agent=technical_wrapped,
    super_agent=super_wrapped,
    regime_indicators=test_regime,
    initial_capital=100000,
    **meta_agent_config
)

print("Meta Agent environments created:")
print(f"  Train observation dim: {meta_train_env.observation_space.shape[0]}")
print(f"  Val:   {len(val_prices)} months")
print(f"  Test:  {len(test_prices)} months")

# Train Meta Agent
print("\n" + "="*70)
print("STEP 3: TRAINING META AGENT")
print("="*70)

# Meta agent hyperparameters (even more conservative)
meta_ppo_config = {
    'learning_rate': 5e-5,      # Very low LR (top-level, careful updates)
    'n_steps': 1024,            # Same as Super
    'batch_size': 128,          # Large batch for stability
    'n_epochs': 3,              # Few epochs (complex observation space)
    'ent_coef': 0.15,           # High entropy (explore comprehensive strategies)
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'clip_range': 0.2,
}

print("\nTraining Meta Agent with PPO...")
meta_model = PPO("MlpPolicy", meta_train_env, **meta_ppo_config, verbose=1)

meta_callback = EarlyStoppingCallback(
    meta_val_env,
    eval_freq=2000,
    patience=5,
    verbose=1
)

# Meta agent needs fewer timesteps (highest-level abstractions)
meta_timesteps = 40000

meta_model.learn(
    total_timesteps=meta_timesteps,
    callback=meta_callback
)

print("\nMeta Agent training completed")

# Evaluate Meta Agent
print("\n" + "="*70)
print("STEP 4: EVALUATING META AGENT")
print("="*70)

meta_train_metrics = evaluate_agent(meta_model, meta_train_env, "Meta Agent - Training")
meta_val_metrics = evaluate_agent(meta_model, meta_val_env, "Meta Agent - Validation")
meta_test_metrics = evaluate_agent(meta_model, meta_test_env, "Meta Agent - Test")

# Complete hierarchy comparison
print("\n" + "="*70)
print("COMPLETE HIERARCHY PERFORMANCE")
print("="*70)

full_comparison = {
    'Sentiment (Best Base)': sentiment_best['test_metrics']['sharpe_ratio'],
    'Technical (Best Base)': technical_best['test_metrics']['sharpe_ratio'],
    'Super Agent': super_test_metrics['sharpe_ratio'],
    'Meta Agent': meta_test_metrics['sharpe_ratio']
}

for name, sharpe in full_comparison.items():
    improvement = (sharpe - best_base) / best_base * 100
    print(f"{name:25s}: {sharpe:.3f} ({improvement:+.1f}% vs best base)")

# Save Meta Agent
meta_model.save('./models/meta_agent_ppo')
print("\nMeta Agent saved: ./models/meta_agent_ppo.zip")

# Store results
results['meta_ppo'] = {
    'model': meta_model,
    'train_metrics': meta_train_metrics,
    'val_metrics': meta_val_metrics,
    'test_metrics': meta_test_metrics,
    'train_env': meta_train_env,
    'val_env': meta_val_env,
    'test_env': meta_test_env,
    'agent_type': 'hierarchical',
    'level': 'meta'
}
```

---

## PART 4: HYPERPARAMETER TUNING FOR HIERARCHICAL AGENTS

### Key Differences from Base Agents:

| Aspect | Base Agents | Hierarchical Agents |
|--------|-------------|---------------------|
| **Observation Space** | 53-110 dims | 20-203 dims |
| **Learning Rate** | 3e-4 | 1e-4 to 5e-5 |
| **n_steps** | 2048 | 1024 |
| **n_epochs** | 5 | 3 |
| **ent_coef** | 0.05 | 0.1-0.15 |
| **timesteps** | 100000 | 40000-50000 |

---

### Why These Differences?

**1. Lower Learning Rate:**
```python
# Hierarchical agents make more abstract decisions
# Need careful, gradual updates
# Too fast → unstable blending strategies

Base:         learning_rate = 3e-4
Super:        learning_rate = 1e-4  (3x slower)
Meta:         learning_rate = 5e-5  (6x slower)
```

**2. Smaller n_steps:**
```python
# Shorter episodes in hierarchical envs
# Less data per agent type
# More frequent updates needed

Base:         n_steps = 2048  (27 episodes worth)
Hierarchical: n_steps = 1024  (13 episodes worth)
```

**3. Fewer n_epochs:**
```python
# Complex observation spaces
# Easy to overfit to current blending strategy
# Keep it simple

Base:         n_epochs = 5
Hierarchical: n_epochs = 3
```

**4. Higher Entropy:**
```python
# Need to explore different blending strategies
# Avoid committing too early
# Financial markets need adaptability

Base:         ent_coef = 0.05
Super:        ent_coef = 0.10
Meta:         ent_coef = 0.15
```

**5. Fewer Timesteps:**
```python
# Less data (only 77 training samples)
# Higher-level abstractions learn faster
# Avoid overfitting to specific agent combinations

Base:         timesteps = 100000
Super:        timesteps = 50000
Meta:         timesteps = 40000
```

---

### Recommended Configurations:

```python
# SUPER AGENT (Conservative - Recommended)
super_config = {
    'learning_rate': 1e-4,
    'n_steps': 1024,
    'batch_size': 128,
    'n_epochs': 3,
    'ent_coef': 0.1,
    'timesteps': 50000,
}

# META AGENT (Very Conservative - Recommended)
meta_config = {
    'learning_rate': 5e-5,
    'n_steps': 1024,
    'batch_size': 128,
    'n_epochs': 3,
    'ent_coef': 0.15,
    'timesteps': 40000,
}

# Alternative: Aggressive (if seeing slow convergence)
aggressive_config = {
    'learning_rate': 3e-4,  # Higher
    'n_steps': 2048,        # More data
    'n_epochs': 5,          # More learning
    'ent_coef': 0.05,       # Less exploration
    'timesteps': 75000,     # More training
}
```

---

## PART 5: TROUBLESHOOTING COMMON ISSUES

### Issue 1: Super/Meta Agent Worse Than Base Agents

**Symptoms:**
- Hierarchical Sharpe < Best base agent Sharpe
- No improvement from hierarchy

**Causes:**
1. Base agents too similar (no diversity to blend)
2. Hierarchical agent not learning (too conservative hyperparameters)
3. Reward function mismatch

**Solutions:**
```python
# Check base agent diversity
sent_weights = sentiment_wrapped.weights
tech_weights = technical_wrapped.weights
correlation = np.corrcoef(sent_weights, tech_weights)[0, 1]

print(f"Base agent correlation: {correlation:.3f}")
if correlation > 0.8:
    print("WARNING: Base agents too similar!")
    print("Solution: Retrain with different features or algorithms")

# Increase hierarchical learning
super_config['ent_coef'] = 0.15  # More exploration
super_config['learning_rate'] = 3e-4  # Faster learning
```

---

### Issue 2: Hierarchical Agent Crashes During Training

**Error:** `AttributeError: 'NoneType' object has no attribute 'weights'`

**Cause:** Agent wrappers not properly initialized

**Solution:**
```python
# Ensure wrappers have all required attributes
class HierarchicalAgentWrapper:
    def __init__(self, model, env, agent_type):
        self.model = model
        self.env = env
        self.agent_type = agent_type
        # CRITICAL: Initialize weights
        self.n_assets = len(env.price_data.columns)
        self.weights = np.ones(self.n_assets) / self.n_assets
    
    def reset(self, seed=None):
        # CRITICAL: Reset weights
        self.weights = np.ones(self.n_assets) / self.n_assets
        return None
```

---

### Issue 3: Training Very Slow

**Symptoms:**
- Each timestep takes >1 second
- Training estimated >5 hours

**Causes:**
1. Nested environment calls (hierarchical overhead)
2. Too many features in Meta agent observation

**Solutions:**
```python
# Reduce timesteps
super_timesteps = 30000  # Instead of 50000
meta_timesteps = 25000   # Instead of 40000

# Increase n_steps (less frequent updates)
super_config['n_steps'] = 2048  # Instead of 1024

# Or train with SAC (more sample efficient)
super_model = SAC("MlpPolicy", super_train_env, 
                  learning_rate=1e-4,
                  buffer_size=10000,
                  batch_size=256)
```

---

### Issue 4: Overfitting in Hierarchical Agents

**Symptoms:**
- Train Sharpe much higher than test Sharpe
- Validation Sharpe decreases after initial improvement

**Causes:**
1. Too many epochs
2. Too many timesteps
3. Learning rate too high

**Solutions:**
```python
# More conservative configuration
super_config = {
    'learning_rate': 5e-5,  # Even lower
    'n_epochs': 2,          # Fewer epochs
    'timesteps': 30000,     # Fewer timesteps
    'ent_coef': 0.2,        # More exploration
}

# Add stronger early stopping
callback = EarlyStoppingCallback(
    val_env,
    eval_freq=1000,  # More frequent checks
    patience=3,      # Stop earlier
)
```

---

## PART 6: EXPECTED RESULTS AND BENCHMARKS

### Realistic Performance Expectations:

```python
# Base Agents (Best):
Sentiment: 0.8-1.2 Sharpe
Technical: 1.2-1.6 Sharpe

# Super Agent:
Expected: 1.4-1.8 Sharpe (+10-20% over best base)

# Meta Agent:
Expected: 1.5-1.9 Sharpe (+15-25% over best base)

# Why improvement?
- Diversification (ensemble effect)
- Regime adaptation (switches strategies)
- Risk management (reduces drawdowns)
```

### Benchmark Scenarios:

**Scenario 1: Strong Improvement (Success)**
```
Base Agents:
  Sentiment: 1.0
  Technical: 1.3
  
Super Agent: 1.6 (+23%)
Meta Agent: 1.7 (+31%)

Interpretation: Hierarchy adds significant value
```

**Scenario 2: Moderate Improvement (Good)**
```
Base Agents:
  Sentiment: 1.1
  Technical: 1.4
  
Super Agent: 1.5 (+7%)
Meta Agent: 1.6 (+14%)

Interpretation: Hierarchy provides modest enhancement
```

**Scenario 3: No Improvement (Needs Tuning)**
```
Base Agents:
  Sentiment: 1.0
  Technical: 1.5
  
Super Agent: 1.4 (-7%)
Meta Agent: 1.5 (0%)

Interpretation: Base agents too correlated OR
                hierarchical hyperparameters need adjustment
```

---

## SUMMARY CHECKLIST

### Before Training Hierarchical Agents:

- [ ] Base agents trained and saved ✓
- [ ] Best base agents selected
- [ ] Agent wrappers created with proper interface
- [ ] Data splits prepared (train/val/test)
- [ ] Regime indicators calculated (if using)
- [ ] Hyperparameters configured conservatively
- [ ] Early stopping callback prepared

### During Training:

- [ ] Monitor validation Sharpe (should improve)
- [ ] Check for crashes (agent wrapper issues)
- [ ] Verify training speed (not too slow)
- [ ] Watch for early stopping trigger

### After Training:

- [ ] Evaluate on all sets (train/val/test)
- [ ] Compare to base agents
- [ ] Check for overfitting (train vs test gap)
- [ ] Save models
- [ ] Visualize portfolio allocations
- [ ] Analyze regime-specific performance

---

## NEXT STEPS

1. **Run Cell 11** to train Super Agent
2. **Verify** Super Agent improves over base agents
3. **Run Cell 12** to train Meta Agent
4. **Compare** all agents in Cell 14
5. **Analyze** which hierarchical level adds most value
6. **Fine-tune** hyperparameters if needed

**Expected Total Time:** 2-3 hours for both hierarchical agents

---

## TROUBLESHOOTING QUICK REFERENCE

| Problem | Quick Fix |
|---------|-----------|
| Worse than base | Increase ent_coef to 0.15 |
| Training too slow | Reduce timesteps by 50% |
| Crashes | Check agent wrapper initialization |
| Overfitting | Reduce n_epochs to 2 |
| Not learning | Increase learning_rate to 3e-4 |
| Unstable | Decrease learning_rate to 5e-5 |

---

**Ready to implement? Copy the code from Part 2 into Cell 11 and Part 3 into Cell 12!**

