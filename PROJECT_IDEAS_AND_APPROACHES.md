## Reinforcement Learning in Finance — Ideas, Approaches, and Project Overview

### What this project is
This repository explores **reinforcement learning (RL)** for **portfolio allocation / rebalancing** under realistic constraints:

- **Multi-asset allocation** with an explicit **cash** component
- Transaction costs and turnover awareness
- Robust evaluation via **rolling / walk-forward validation**
- A production-oriented "decision layer" ("Board") that selects among specialized agents

The primary workflow is notebook-first, with Python modules used for reusable utilities.

---

## Core system design

### 1) Feature engineering (minimal + causal)
The baseline feature set is intentionally simple and causal (no lookahead):

- **Log returns** per asset
- **Rolling volatility** (normalized / z-scored)
- **RSI** (scaled around 0)
- **Correlation** of each asset to benchmark returns

Output shapes commonly used:
- `features`: `(time, n_assets, n_features)`
- `prices`: `(time, n_assets)`
- `benchmark`: `(time,)`

---

### 2) Environment: Portfolio rebalancing with a "Smart Valve"
The environment implements realistic portfolio mechanics:

- Portfolio consists of **equities + cash**.
- The agent outputs **logits**; the env applies **softmax** to obtain portfolio weights.
- A **Smart Valve** reduces noise trading:
  - Compute drift between the portfolio’s natural (drifted) weights and target weights
  - **Only rebalance** when drift exceeds a threshold (`config.REBALANCE_THRESHOLD`)
- Transaction cost is proportional to **trade volume** (`config.TX_COST_BPS`).

#### Key implementation points
- **Timing** (clean RL convention):
  - observation at time `t`
  - choose action at `t` and rebalance at `price[t]`
  - reward realized from `NAV(t) -> NAV(t+1)` using `price[t+1]`
- **No negative cash**: if estimated costs exceed intended cash, equity allocation is scaled so cash never goes below zero.

---

### 3) Three specialized agents (reward shaping)
The system uses a small ensemble of role-specialized agents:

- **Bull**: incentivized to outperform benchmark (alpha) and tolerate concentration.
- **Bear**: incentivized to minimize equity exposure (defensive posture) and avoid losses.
- **Sniper**: incentivized for smoother returns (penalizes large daily moves), meant for choppy regimes.

These are trained using PPO on the same environment dynamics but different reward functions.

---

### 4) The "Board" (AdaptiveBoard): regime selection layer
Instead of one monolithic policy, a lightweight rule-based "Board" selects which specialist agent to use.

Inputs:
- **Market volatility** (rolling std of benchmark returns)
- **Market trend** (benchmark vs SMA50 sign)

Outputs:
- Selected agent action (logits)
- A regime label (e.g., `CRASH`, `CHOPPY`, `GROWTH`, `CAUTION`, etc.)

#### Key idea
**Volatility is not always bad.** High volatility in an **uptrend** can be a "violent rally" regime rather than a crash. The board can treat **high-vol uptrend** differently than **high-vol downtrend**.

The project experiments with policies such as:
- **baseline**: `vol > panic` always → Bear
- **panic_up_sniper**: `vol > panic` + uptrend → Sniper, downtrend → Bear
- **trend_adjusted**: trend modifies effective thresholds (uptrend tolerates higher vol before switching to panic/choppy)

---

## Validation protocol

### Rolling walk-forward
Typical protocol:
- **Train** 3 years
- **Validate** 1 year (optimize board thresholds)
- **Test** 1 year
- Repeat for successive test years and **stitch** test results

### Threshold optimization
On the validation year, the board’s thresholds are grid searched to maximize an objective like **Sharpe**.

### Stitched evaluation
Results are stitched across the true test years:
- NAV is commonly reconstructed from `nav_pct_change` (returns-based stitching)
- For any sanity checks using raw NAV, you must stitch year-by-year because the env resets NAV at each test year

---

## Reproducibility notes (important)

### 1) Config import gotcha (nested repo)
This repo contains a nested `daily_adviser/.git`, which can cause notebooks executed from that directory to import a different `config.py` than intended.

Best practice:
- In notebooks, explicitly load the **project root** `config.py` and print its path.

### 2) PPO is not perfectly deterministic
Even with a fixed seed, PPO training may vary across runs due to backend nondeterminism.

Best-effort steps:
- set global seeds (`random`, `numpy`, `torch`)
- force single-thread execution where possible
- enable deterministic torch algorithms when available

### 3) Universe changes change everything
Changing `TICKERS` changes:
- action dimension
- learned policies
- portfolio dynamics

For apples-to-apples comparisons, keep the same universe or explicitly override it.

---

## How to run the rolling validation notebook

The notebook `daily_adviser/notebooks/02_rolling_validation_v1_fixed.ipynb` is the clean baseline for experiments.

Suggested run order:
1. Run **Cell 1** and confirm it prints the correct `config.py` path.
2. Run data + features.
3. Run training/validation/test loop.
4. Review stitched performance, cash allocation, and drawdown plots.

To compare board policies:
- Run the A/B setup that logs `variant` for each policy and prints a side-by-side summary.

---

## What to explore next (ideas)

- Replace SMA trend with more robust trend signal (e.g., slope of SMA, MA crossover, or volatility-adjusted trend)
- Add regime features like drawdown, VIX proxy, or realized skew/kurtosis
- Reduce training variance using more timesteps, evaluation callbacks, or multiple seeds
- Add risk metrics beyond max drawdown: **Ulcer Index**, **time-under-water**, downside deviation
- Introduce constraints: max position size, sector caps, minimum cash floors

