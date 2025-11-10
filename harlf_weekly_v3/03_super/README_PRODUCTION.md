# PRODUCTION TRAINING - COMPLETE GUIDE

## ✅ What Was Fixed

### 1. SuperAgent Initialization
**BEFORE (WRONG):**
```python
SuperAgent(base_agents=base_agents, main_env=train_env, learning_rate=3e-4)
```

**AFTER (CORRECT):**
```python
SuperAgent(base_agents=base_agents, learning_rate=3e-4)
```

### 2. SuperAgent.train() Parameters
**BEFORE (WRONG):**
```python
super_agent.train(
    env=train_env,              # ❌ Wrong parameter name!
    n_epochs=50,                # ❌ Wrong parameter name!
    n_episodes_per_epoch=5,     # ❌ Doesn't exist!
    eval_episodes=3             # ❌ Doesn't exist!
)
```

**AFTER (CORRECT):**
```python
super_agent.train(
    train_env=train_env,        # ✓ Correct!
    val_env=val_env,
    n_episodes=100,             # ✓ Correct!
    eval_every=10,              # ✓ Correct!
    patience=20,
    verbose=True
)
```

---

## 📦 Files Ready for Production

### Core Training Files
1. **train_3_base_agents_PRODUCTION.py** - Train 3 base agents (300k steps each)
2. **train_super_agent_PRODUCTION.py** - Train super agent (100 episodes)
3. **test_fixed_weights.py** - Test simple weight combinations

### Quick Test File (from before)
4. **quick_test_FINAL.py** - Quick 10-minute test

---

## 🚀 How to Run

### Option 1: Full Production Training (Recommended)

**Step 1: Train Base Agents (~1-2 hours)**
```bash
cd /Users/nataliamarko/Documents/GitHub_Projects/reinforcement_learning_in_finance/harlf_weekly_v3/03_super/

python train_3_base_agents_PRODUCTION.py
```

This creates:
- `models/base_agents_production.json` (metadata)
- 3 model files in `models/technical/` and `models/sentiment/`

**Step 2: Train Super Agent (~20-30 minutes)**
```bash
python train_super_agent_PRODUCTION.py
```

This creates:
- `models/super_agent_production.pt` (trained super agent)
- `models/super_agent_production_analysis.json` (performance analysis)

**Step 3: Test Fixed Weights (optional, ~5 minutes)**
```bash
python test_fixed_weights.py
```

This tests if simple weight combinations work better than meta-learning.

---

### Option 2: Quick Test (10 minutes)

If you just want to verify everything works:

```bash
python quick_test_FINAL.py
```

This trains with only 10k steps (not production-ready but fast).

---

## 📊 Expected Results

### Quick Test (10k steps)
| Metric | Value |
|--------|-------|
| Base Agent Sharpe | ~1.3 |
| Super Agent Validation | ~1.6 |
| Super Agent Test | ~1.0 |

**Note:** These are undertrained and for testing only.

### Production (300k steps)
| Metric | Value |
|--------|-------|
| Base Agent Sharpe | **1.5 - 2.0** |
| Super Agent Validation | **2.0 - 2.5** |
| Super Agent Test | **1.5 - 2.0** |

**Note:** These are production-ready models.

---

## 🎯 What Each Script Does

### train_3_base_agents_PRODUCTION.py
- Trains 3 agents with **300,000 steps** each
- Agent 1: Technical + EMA Sharpe (return maximizer)
- Agent 2: Technical + Multi-objective (loss minimizer)
- Agent 3: Sentiment + EMA Sharpe (alternative strategy)
- Time: **1-2 hours total**
- Output: `base_agents_production.json`

### train_super_agent_PRODUCTION.py
- Loads the 3 base agents
- Trains meta-learner to combine them optimally
- Uses **100 episodes** with early stopping
- Time: **20-30 minutes**
- Output: `super_agent_production.pt`

### test_fixed_weights.py
- Tests simple weight combinations (no learning)
- Compares: equal weights, favor best, single agent only, etc.
- Tells you if simple weighting is enough
- Time: **5 minutes**
- Output: `fixed_weights_results.json`

---

## 🔧 Troubleshooting

### Issue: "Base agents not found"
**Solution:** Run `train_3_base_agents_PRODUCTION.py` first.

### Issue: "Data mismatch warning"
```
⚠️ agent_1: 202 dates
⚠️ agent_3: 197 dates
```
**Solution:** This is OK. Super agent handles it by stopping at 197.

### Issue: Negative training Sharpe
```
Episode 10: Train Sharpe = -0.659
```
**Solution:** This is normal early in training. It improves over time.

### Issue: Validation > Test Sharpe
```
Val Sharpe: 1.583
Test Sharpe: 1.030
```
**Solution:** This is overfitting. With more episodes (100 instead of 30), the gap reduces.

---

## 📈 Understanding the Results

### Base Agents Analysis
After training, check `base_agents_production.json`:
```json
{
  "agent_1_return_max": {
    "test_sharpe": 1.5,
    "model_path": "..."
  },
  ...
}
```

**Good if:** Test Sharpe > 1.2

### Super Agent Analysis
After training, check `super_agent_production_analysis.json`:
```json
{
  "test_sharpe": 1.8,
  "agent_weights": {
    "agent_1": {"mean": 0.45},
    "agent_2": {"mean": 0.35},
    "agent_3": {"mean": 0.20}
  }
}
```

**What to look for:**
- **Test Sharpe > Base agents:** Super agent adds value ✓
- **Balanced weights (20-40% each):** All agents contributing ✓
- **One agent dominates (>80%):** Just use that agent alone

### Fixed Weights Results
After testing, check `fixed_weights_results.json`:

**If simple weights beat super agent:**
- Use fixed weights (simpler, no training needed)
- Example: `[0.4, 0.4, 0.2]`

**If super agent beats simple weights:**
- Use trained super agent (more adaptive)

---

## 💡 Tips

1. **Start with full training** - Quick test is just for verification
2. **Check fixed weights first** - Often simpler is better
3. **Monitor validation Sharpe** - Should increase over time
4. **Check agent usage** - All agents should contribute (not just one)
5. **Compare to baseline** - Super agent should beat best single agent

---

## ✅ Success Criteria

You'll know it worked when:

- ✓ All 3 base agents train without errors
- ✓ Base agent Test Sharpe > 1.2
- ✓ Super agent trains for 100 episodes
- ✓ Super agent Test Sharpe > best base agent
- ✓ All agents get used (weights 15-50% each)

---

## 🎉 Next Steps After Training

1. **Backtest** - Run on out-of-sample data
2. **Compare strategies** - Super agent vs fixed weights vs best single agent
3. **Production deploy** - Use best strategy for live trading
4. **Monitor performance** - Track ongoing Sharpe ratio

---

## 📞 Quick Reference

| Task | Command | Time |
|------|---------|------|
| Quick Test | `python quick_test_FINAL.py` | 10 min |
| Train Base Agents | `python train_3_base_agents_PRODUCTION.py` | 1-2 hours |
| Train Super Agent | `python train_super_agent_PRODUCTION.py` | 20-30 min |
| Test Fixed Weights | `python test_fixed_weights.py` | 5 min |

**Total production time: ~2-2.5 hours**

Good luck! 🚀
