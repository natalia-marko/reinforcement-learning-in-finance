# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
```bash
pip install numpy pandas gymnasium stable-baselines3 matplotlib seaborn scikit-learn torch yfinance
```

---

## 📋 Step-by-Step Workflow

### Step 1: Prepare Data (10 minutes)
```bash
cd 01_data_preparation
python data_preparation_part1_v2.py
```

**Output:** 
- `data_hierarchical/` with train/val/test splits
- Normalized features ready for training

**Verify:**
```bash
ls ../data_hierarchical/technical/
# Should see: train.csv, val.csv, test.csv
```

---

### Step 2: Train Base Agents (30-60 minutes)
```bash
cd ../02_part1_base_agents
jupyter notebook retrain_base_agents.ipynb
```

**Run all cells** and wait for training to complete.

**Expected Results:**
- Technical Agent: Test Sharpe ~1.78
- Sentiment Agent: Test Sharpe ~1.92

**Output:**
- `models_part1/best_technical_*.zip`
- `models_part1/best_sentiment_*.zip`
- Training visualizations in notebook

---

### Step 2B: Compare Reward Functions (Optional, 60+ minutes)

Still in `02_part1_base_agents/`:

**Interactive comparison:**
```bash
jupyter notebook compare_reward_functions.ipynb
```

**Or run Python script:**
```bash
python compare_reward_functions.py
```

**Output:**
- Performance comparison across 3 approaches
- Visualizations in `plots/reward_comparison/`
- Recommendation for best approach

---

### Step 2C: Tune Multi-Objective (Optional, 15-60 minutes)

Still in `02_part1_base_agents/`:

**Quick Test:**
```bash
python tune_multi_objective.py --quick
```

**Full Grid Search:**
```bash
python tune_multi_objective.py --grid --technical
python tune_multi_objective.py --grid --sentiment
```

---

### Step 3: Train Super Agent (Future)
```bash
cd ../03_part2_super_agent
jupyter notebook train_super_agent.ipynb
```

---

## 🎯 Current Status

### ✅ Completed
- [x] Data preparation pipeline
- [x] Base agent training (EMA Sharpe reward)
- [x] Optimized multi-objective reward
- [x] Reward function comparison framework
- [x] Comprehensive documentation
- [x] Project organization

### 🔄 In Progress
- [ ] Multi-objective training with optimized penalties
- [ ] Full reward function comparison

### ⏳ Planned
- [ ] Super agent implementation
- [ ] Production deployment
- [ ] Live monitoring

---

## 📊 Key Results Summary

| Component | Status | Performance |
|-----------|--------|-------------|
| Data Pipeline | ✅ Ready | 20+ indicators, normalized |
| Technical Agent | ✅ Trained | Test Sharpe: 1.78 (PPO) |
| Sentiment Agent | ✅ Trained | Test Sharpe: 1.92 (SAC) |
| Multi-Objective | ⏳ Optimizing | Expected: 2.0+ (Technical) |
| Super Agent | ⏳ Planned | Target: 2.2+ |

---

## 🗂️ File Locations

| What | Where |
|------|-------|
| Data prep script | `01_data_preparation/data_preparation_part1_v2.py` |
| Training notebook | `02_part1_base_agents/retrain_base_agents.ipynb` |
| Environments | `02_part1_base_agents/environments_part1.py` |
| Multi-objective | `03_reward_comparison/multi_objective.py` |
| Comparison | `03_reward_comparison/compare_reward_functions.ipynb` |
| Trained models | `models_part1/`, `models_multi_objective/` |
| Processed data | `data_hierarchical/` |
| Documentation | `docs/` |
| Visualizations | `plots/` |

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `PROJECT_STRUCTURE.md` | Full project overview |
| `DIRECTORY_TREE.txt` | Visual directory tree |
| `ORGANIZATION_SUMMARY.md` | Reorganization details |
| `docs/PENALTY_OPTIMIZATION.md` | Multi-objective tuning |
| `docs/TUNING_GUIDE.md` | How to tune penalties |
| `01_data_preparation/README.md` | Data pipeline guide |
| `02_part1_base_agents/README.md` | Training guide |
| `03_reward_comparison/README.md` | Reward functions explained |
| `04_part2_super_agent/README.md` | Super agent concept |

---

## 🆘 Troubleshooting

### Data Issues
**Problem:** Data preparation fails
**Solution:** 
```bash
cd 01_data_preparation
# Check if sentiment data exists
ls ../sentiment_data/
# If missing, run sentiment data notebooks first
```

### Training Issues
**Problem:** Low performance or NaN rewards
**Solution:**
- Check data normalization in `data_hierarchical/metadata.json`
- Verify feature lists are correct
- Try different algorithm (PPO → SAC → A2C)
- Reduce learning rate

### Import Errors
**Problem:** `ModuleNotFoundError`
**Solution:**
```bash
pip install --upgrade stable-baselines3 gymnasium torch
```

### Memory Issues
**Problem:** Out of memory during training
**Solution:**
- Reduce `batch_size` in training config
- Reduce `buffer_size` for SAC
- Close other applications

---

## 💡 Tips

1. **Start Simple**: Begin with Step 1 & 2 (base agents)
2. **Use EMA Sharpe**: Default reward function works well
3. **Check Visualizations**: Review plots after training
4. **Validate Often**: Monitor val set to prevent overfitting
5. **Save Experiments**: Keep track of different configurations

---

## 🎓 Learning Path

### Beginner
1. Run data preparation
2. Train base agents
3. Review results and plots
4. Read base documentation

### Intermediate
1. Compare reward functions
2. Analyze generalization gaps
3. Tune multi-objective penalties
4. Experiment with hyperparameters

### Advanced
1. Implement custom reward functions
2. Build super agent
3. Deploy to production
4. Set up monitoring

---

## 📞 Quick Commands

### View Structure
```bash
cat DIRECTORY_TREE.txt
```

### Check Data
```bash
ls data_hierarchical/technical/
cat data_hierarchical/metadata.json | head -20
```

### Check Models
```bash
ls models_part1/
cat models_part1/best_models_part1.json
```

### View Results
```bash
cat models_multi_objective/results.json
```

### Clean Temporary Files
```bash
rm -rf __pycache__/
find . -name "*.pyc" -delete
```

---

## 🎯 Next Actions

### Immediate (Do Now)
```bash
cd 03_reward_comparison
python multi_objective.py
```

### Short-term (This Week)
1. Run full reward comparison
2. Select best approach
3. Begin super agent design

### Long-term (This Month)
1. Complete super agent
2. Comprehensive backtesting
3. Production deployment planning

---

## ✅ Success Checklist

- [ ] Data prepared and normalized
- [ ] Base agents trained (Sharpe > 1.5)
- [ ] Visualizations generated
- [ ] Results documented
- [ ] Best models saved
- [ ] Reward functions compared
- [ ] Super agent trained
- [ ] Production ready

---

**Last Updated:** 2025-01-24
**Version:** 2.0 (Organized)
**Status:** Ready for Training

**Need Help?** Check `PROJECT_STRUCTURE.md` for comprehensive guide!

