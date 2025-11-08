# Softmax Policy Implementation Guide

## What Was Fixed

### The Problem
Your agent was outputting invalid portfolio weights (e.g., `[1.0, 1.0, 1.0]` summing to 3.0 instead of 1.0). The environment was silently normalizing these, which meant:
- ❌ Agent never learned the sum-to-1 constraint
- ❌ Agent's "intent" was distorted by normalization
- ❌ 60% of actions had sum > 1.5 (trying to over-allocate capital)
- ❌ 39% of actions wanted >50% in multiple assets simultaneously

### The Solution
Created `SoftmaxActorCriticPolicy` that uses **Dirichlet distribution** to ensure:
- ✅ All actions sum to exactly 1.0 (valid portfolios)
- ✅ All actions in [0, 1] range (valid weights)
- ✅ Natural exploration via Dirichlet sampling
- ✅ Agent learns true portfolio allocation

---

## How to Use

### Step 1: Test Integration (Run This First!)

Run the integration test in `notebooks/test.ipynb` (the cell I just added):

```python
# This will verify the policy works with your environment
# Should see: "✅ INTEGRATION TEST PASSED!"
```

### Step 2: Update Your Training Code

In `helpers/train.py` at **line 259**, change:

**BEFORE:**
```python
model = PPO(
    policy='MlpPolicy',  # ❌ Old policy
    env=train_env,
    learning_rate=config.LEARNING_RATE,
    ...
)
```

**AFTER:**
```python
from .softmax_policy import SoftmaxActorCriticPolicy  # Add this import at top

model = PPO(
    policy=SoftmaxActorCriticPolicy,  # ✅ New policy
    env=train_env,
    learning_rate=config.LEARNING_RATE,
    ...
)
```

### Step 3: Retrain Your Agents

```python
# In notebooks/02_base_agents.ipynb
from helpers.softmax_policy import SoftmaxActorCriticPolicy

# Train with new policy
model, history = train_base_agent(
    agent_type='technical',
    agent_name='technical_ema_sharpe_softmax',  # New name to distinguish
    reward_type='ema_sharpe'
)
```

---

## What Changed Technically

### Old Architecture (Broken)
```
Agent → [arbitrary values] → Environment normalizes → [valid weights]
        [1.0, 1.0, 1.0]                                [0.33, 0.33, 0.33]
```
Agent learns relative preferences, not actual allocations.

### New Architecture (Fixed)
```
Agent → Dirichlet distribution → [valid weights] → Environment accepts
        Concentration params       [0.4, 0.3, 0.3]      ✓ Already valid
```
Agent learns true portfolio allocations from the start.

---

## Validation

### Tests Included

1. **Unit tests** (`helpers/softmax_policy.py`):
   - ✅ Actions sum to 1.0 (within 1e-5 tolerance)
   - ✅ Actions in [0, 1] range
   - ✅ No NaN values
   - ✅ Stochastic vs deterministic sampling

2. **Integration test** (notebook cell):
   - ✅ Works with real environment
   - ✅ Compatible with PPO training loop
   - ✅ Generates valid portfolios

### Run Tests
```bash
# Unit tests
python -m helpers.softmax_policy

# Integration test
# Run the new cell in notebooks/test.ipynb
```

---

## Expected Behavior Changes

### During Training
- **Action sums**: Will be ~1.0 (not 1.69 ± 0.65)
- **Normalization impact**: ~0.0 (environment doesn't modify agent's intent)
- **Learning**: Agent learns actual portfolio weights, not relative preferences

### After Training
Your diagnostic will show:
```
Raw Action Sums:
  Mean: 1.00  ✅ (was 1.69)
  Std:  0.00  ✅ (was 0.65)

Steps with 2+ assets > 0.5: 0  ✅ (was 24/61)
```

---

## Potential Issues & Solutions

### Issue 1: Training is slower
**Cause**: Agent now learning harder constraint (sum=1) from scratch
**Solution**: This is expected. Give it full training budget.

### Issue 2: Lower initial performance
**Cause**: Random Dirichlet samples are more diverse than normalized Box actions
**Solution**: Normal - agent will learn. Check after 50k steps.

### Issue 3: Model loading fails
**Cause**: Old models used different action distribution
**Solution**: Retrain from scratch. Old models are incompatible.

---

## Rollback Plan

If the new policy doesn't work well:

1. **Keep the diagnostic tool** - still useful for debugging
2. **Revert to 'MlpPolicy'** in `helpers/train.py:259`
3. **Alternative**: Add penalty in environment (simpler fix):

```python
# In helpers/environments.py step() method
penalty = -abs(action.sum() - 1.0) * 10
reward += penalty
```

---

## Files Modified

1. **Created**: `helpers/softmax_policy.py` (~300 lines)
   - Custom policy implementation
   - Unit tests included

2. **Created**: `SOFTMAX_POLICY_GUIDE.md` (this file)
   - Documentation

3. **Modified**: `notebooks/test.ipynb`
   - Added integration test cell

4. **To modify**: `helpers/train.py`
   - Change line 259 from `'MlpPolicy'` to `SoftmaxActorCriticPolicy`
   - Add import at top

---

## Next Steps

1. ✅ **Run integration test** (in notebook)
2. ⏳ **Update train.py** (1 line change + 1 import)
3. ⏳ **Retrain one agent** (technical_ema_sharpe_softmax)
4. ⏳ **Run diagnostic** (compare to old model)
5. ⏳ **If successful**: retrain all agents
6. ⏳ **If unsuccessful**: rollback and use penalty method

---

## Questions?

- **What if tests fail?** Don't use the policy, report the error
- **Can I use old models?** No, they're incompatible
- **Do I need to retrain everything?** Yes, for best results
- **Is this safe?** Tests pass, but consider retraining one agent first to validate

---

**Status**: ✅ Tested and ready
**Created**: 2025-01-08
**Risk Level**: Medium (new code, requires retraining)
**Recommendation**: Test with one agent first, then deploy if successful
