# Part 2: Hierarchical Super Agent

## Overview
Train a hierarchical reinforcement learning agent that combines recommendations from base agents (technical and sentiment) to make final portfolio decisions.

---

## 🎯 Concept

### Hierarchical Structure

```
┌─────────────────────────────────────┐
│         Super Agent                 │
│   (Meta-Level Controller)           │
│                                     │
│  • Receives base agent outputs      │
│  • Decides agent weights/selection  │
│  • Makes final portfolio decision   │
└─────────────────────────────────────┘
            ▲         ▲
            │         │
    ┌───────┘         └───────┐
    │                         │
┌───▼────────┐        ┌───────▼─────┐
│ Technical  │        │  Sentiment  │
│   Agent    │        │    Agent    │
│            │        │             │
│ • Tech     │        │ • Sentiment │
│   features │        │   features  │
│ • Portfolio│        │ • Portfolio │
│   weights  │        │   weights   │
└────────────┘        └─────────────┘
```

### Why Hierarchical?

1. **Specialization**: Base agents focus on their domains
2. **Ensemble Benefits**: Combine diverse strategies
3. **Adaptive Blending**: Learn when to trust each agent
4. **Market Regime Detection**: Switch strategies based on conditions

---

## 📁 Files

### `super_agent_enviroment.py`
**Purpose:** Gymnasium environment for the super agent

**Key Features:**
- Loads pre-trained base agents
- Gets portfolio recommendations from both agents
- Super agent observes:
  - Base agent recommendations
  - Market features
  - Recent performance
- Super agent outputs meta-policy:
  - Agent selection weights
  - Final portfolio allocation

**Observation Space:**
- Base agent recommendations (2 × n_assets)
- Market state features
- Agent performance history

**Action Space:**
- Agent blending weights
- Or discrete selection (which agent to follow)

### `train_super_agent.ipynb`
**Purpose:** Train the hierarchical super agent

**Training Process:**
1. Load best base agents from `../models_part1/`
2. Create super agent environment
3. Train meta-controller
4. Evaluate on test set
5. Compare vs individual agents

---

## 🎯 Approaches

### Approach 1: Agent Selection
```python
# Discrete action: select one agent
action ∈ {0, 1}  # 0=technical, 1=sentiment

if action == 0:
    portfolio = technical_agent.predict(obs)
else:
    portfolio = sentiment_agent.predict(obs)
```

**Pros:** Simple, interpretable
**Cons:** Can't combine agents

### Approach 2: Agent Blending
```python
# Continuous weights for blending
weights = [w_technical, w_sentiment]  # sum to 1

portfolio = (w_technical × technical_portfolio + 
             w_sentiment × sentiment_portfolio)
```

**Pros:** Flexible, smooth transitions
**Cons:** More complex

### Approach 3: Direct Meta-Policy
```python
# Super agent makes decisions directly
# But uses base agent features as input

obs = concat([technical_features, sentiment_features, 
              technical_rec, sentiment_rec])
portfolio = super_agent.predict(obs)
```

**Pros:** Maximum flexibility
**Cons:** May ignore base agents

---

## 🔧 Implementation

### Load Base Agents
```python
from stable_baselines3 import PPO, SAC

tech_agent = PPO.load('../models_part1/best_technical_ppo.zip')
sent_agent = SAC.load('../models_part1/best_sentiment_sac.zip')
```

### Create Super Environment
```python
from super_agent_enviroment import SuperAgentEnv

super_env = SuperAgentEnv(
    data_dir='../data_hierarchical',
    split='train',
    tech_model=tech_agent,
    sent_model=sent_agent,
    blending_mode='continuous'  # or 'discrete'
)
```

### Train Super Agent
```python
super_model = PPO(
    'MlpPolicy',
    super_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512
)

super_model.learn(total_timesteps=200_000)
```

---

## 📊 Expected Results

### Individual Agents (Baseline)
| Agent | Test Sharpe |
|-------|-------------|
| Technical | 1.78 |
| Sentiment | 1.92 |

### Super Agent (Target)
| Approach | Expected Test Sharpe |
|----------|---------------------|
| Best Individual | 1.92 |
| Simple Average | ~1.85 |
| **Super Agent (optimal blending)** | **2.0-2.2** 🎯 |

**Goal:** Beat best individual agent by combining strengths.

---

## 🎓 Key Considerations

### 1. Exploration-Exploitation
- Super agent must learn *when* to trust each base agent
- Market conditions matter (trending vs mean-reverting)

### 2. Base Agent Quality
- Super agent can only be as good as base agents
- Garbage in, garbage out

### 3. Overfitting Risk
- Don't overfit to validation set
- Use held-out test set for final evaluation

### 4. Market Regimes
- Technical agent may excel in trending markets
- Sentiment agent may excel in news-driven markets
- Super agent learns the patterns

---

## 🔬 Evaluation Metrics

### Performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside deviation
- **Max Drawdown**: Worst peak-to-trough
- **Calmar Ratio**: Return / max drawdown

### Blending Analysis
- **Agent selection distribution**: How often each agent is chosen
- **Blending weights over time**: Evolution of trust
- **Regime detection**: Performance by market condition

### Comparison
- **vs Best Individual**: Improvement over best base agent
- **vs Equal Weight**: Improvement over 50/50 blend
- **vs Benchmark**: Outperformance vs buy-and-hold

---

## 📈 Visualizations

Create comprehensive analysis:
- Agent selection over time
- Blending weights evolution
- Performance by market regime
- Cumulative returns comparison
- Risk-return scatter

---

## 🚀 Advanced Features

### Market Regime Detection
```python
# Add regime features to observation
regime_features = [
    volatility_regime,  # High/low vol
    trend_strength,     # Trending/ranging
    sentiment_signal,   # Positive/negative
]
```

### Dynamic Risk Management
```python
# Adjust based on recent performance
if recent_drawdown > threshold:
    reduce_risk()  # Shift to more conservative agent
```

### Online Learning
```python
# Update super agent with new data
super_model.learn(new_episodes, reset_num_timesteps=False)
```

---

## 🔄 Workflow

1. **Prepare Base Agents**
   - Train/load best base agents
   - Verify performance on validation

2. **Design Super Environment**
   - Define observation space
   - Choose blending approach
   - Implement reward function

3. **Train Super Agent**
   - Multiple random seeds
   - Track validation performance
   - Early stopping

4. **Evaluate**
   - Test set performance
   - Ablation studies
   - Error analysis

5. **Deploy**
   - Live monitoring
   - Performance tracking
   - Periodic retraining

---

## 📚 References

1. Sutton, R. S., Precup, D., & Singh, S. (1999). *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning*. Artificial intelligence, 112(1-2), 181-211.

2. Dayan, P., & Hinton, G. E. (1992). *Feudal reinforcement learning*. In Advances in neural information processing systems (pp. 271-278).

3. Bacon, P. L., Harb, J., & Precup, D. (2017). *The option-critic architecture*. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 31, No. 1).

---

## 🎯 Success Criteria

✅ **Minimum Viable:**
- Test Sharpe > 1.92 (best individual agent)
- Stable across market conditions
- Low correlation with base agents

🌟 **Excellent:**
- Test Sharpe > 2.0
- < 15% max drawdown
- Consistent outperformance

---

## Next Steps

1. ⏳ Implement super agent environment
2. ⏳ Train with multiple approaches
3. ⏳ Comprehensive evaluation
4. ⏳ Deploy to production
5. ⏳ Monitor and retrain

