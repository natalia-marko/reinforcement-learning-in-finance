# Daily AI Portfolio Advisor

An automated AI-powered portfolio advisor that uses reinforcement learning to generate daily rebalancing recommendations for a 7-stock tech portfolio.

## Overview

This system uses an **ensemble of three specialized RL agents** (Bull, Bear, Sniper) trained with PPO to provide intelligent portfolio allocation advice. The system adapts to market conditions by mixing agent predictions based on real-time volatility signals.

### Key Features

- **Multi-Agent Ensemble**: Bull agent for growth, Bear for crash protection, Sniper for precision timing
- **Regime Detection**: Automatically switches strategy based on market volatility
- **Circuit Breakers**: Emergency cash allocation when panic thresholds are triggered
- **Inertia Management**: Avoids excessive trading by requiring minimum conviction threshold
- **Transaction Costs**: Accounts for 10bps trading costs
- **Automated Notifications**: Telegram alerts with trading recommendations

## Architecture

```
Market Data (YFinance)
         ↓
Feature Engineering (Volatility, RSI, Correlations)
         ↓
Regime Detection (Fast/Slow Volatility)
         ↓
Ensemble Voting (Bull/Bear/Sniper)
         ↓
Portfolio Decision (Trade/Hold with Inertia)
         ↓
Telegram Notification
```

### Trading Logic

| Market Condition | Regime | Agent Mix |
|-----------------|--------|-----------|
| Vol > 1.77% (fast) OR Vol > 3.78% (slow) | **CRASH** | 100% Cash (Circuit Breaker) |
| Vol > 1.2% (stable) | **CHOPPY** | 50% Sniper + 25% Bear + 25% Bull |
| Vol < 1.2% | **GROWTH** | 70% Bull + 30% Sniper |

## Setup

### 1. Install Dependencies

```bash
cd daily_adviser
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file (use `.env.example` as template):

```bash
cp .env.example .env
```

Edit `.env`:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
TELEGRAM_CHAT_ID=your_chat_id_from_get_chat_id_script
```

**Getting Telegram Credentials:**

1. Create a bot via [@BotFather](https://t.me/botfather) on Telegram
2. Copy the bot token
3. Run `python get_chat_id.py` to get your chat ID
4. Update `.env` file

### 3. Configure GitHub Actions (Optional)

For automated daily runs, set repository secrets:

- Go to GitHub repo → Settings → Secrets → Actions
- Add `TELEGRAM_BOT_TOKEN`
- Add `TELEGRAM_CHAT_ID`

## Usage

### Local Execution

1. **Update your current portfolio weights** in `daily_advisor.py`:

```python
CURRENT_WEIGHTS = np.array([
    0.14, # NVDA
    0.14, # MU
    0.14, # AAPL
    0.14, # AMD
    0.14, # ASML
    0.14, # MSFT
    0.14, # GOOG
    0.02  # CASH
])
```

2. **Run the advisor:**

```bash
python daily_advisor.py
```

3. **Check output:**
   - Console: Real-time decisions and reasoning
   - Telegram: Formatted recommendation with action items
   - Log file: `advisory_log_YYYYMMDD.jsonl` with structured data

### GitHub Actions

The system runs automatically **Monday-Friday at 9:30 AM ET** (market open).

**Manual trigger:**
- Go to Actions tab
- Select "Daily Financial Adviser"
- Click "Run workflow"

## Configuration

### Portfolio Assets

Edit `TICKERS` in `daily_advisor.py` (line 49):
```python
TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']
```

⚠️ **Warning**: If you change assets, you MUST retrain the models.

### Thresholds

Adjust sensitivity in `daily_advisor.py`:
- `BEST_FAST_VOL = 0.0177` - Fast volatility trigger (5-day)
- `BEST_SLOW_VOL = 0.0378` - Slow volatility trigger (20-day)  
- `BEST_THRESH = 0.0286` - Minimum turnover to execute trade (2.86%)

These were optimized via Optuna in `archive/legacy_versions/simple/03_tuning.ipynb`.

## Model Files

The system requires three trained PPO models:
- `agent_bull.zip` (2.1 MB)
- `agent_bear.zip` (2.1 MB)
- `agent_sniper.zip` (2.1 MB)

**Location**: Same directory as `daily_advisor.py`

📖 See [MODELS_README.md](MODELS_README.md) for training details.

## Understanding the Output

### Console Output

```
=================================================
      AI PORTFOLIO ADVISOR (PRODUCTION)   
=================================================
Fetching live data for 7 assets + QQQ...

--- MARKET WEATHER (2025-12-05) ---
Fast Volatility (5d):  1.23%  [Trigger: >1.77%]
Slow Volatility (20d): 0.98%  [Trigger: >3.78%]
AI Regime Decision:    GROWTH (Bull Allocation)
Required Turnover:     3.45%
Inertia Threshold:     2.86%
--------------------------------------------------
📢 DECISION: REBALANCE REQUIRED
--------------------------------------------------
ASSET    CURRENT    TARGET     DELTA
--------------------------------------------------
NVDA      14.0%      18.2%      +4.2% <<< BIG MOVE
MU        14.0%      13.1%      -0.9% 
...
```

### Telegram Notification

```
🤖 AI PORTFOLIO ADVISOR

📅 Date: 2025-12-05
📊 Market Weather:
  • Fast Vol (5d): 1.23%
  • Slow Vol (20d): 0.98%

🎯 Regime: GROWTH (Bull Allocation)

📢 DECISION: REBALANCE REQUIRED
Turnover: 3.5%

ASSET    NOW  TARGET   DELTA
----------------------------
NVDA    14.0%  18.2%  +4.2%⬆️
MU      14.0%  13.1%  -0.9%⬇️
...
```

## Feature Engineering

The system processes the following features per asset:

1. **Normalized Volatility**: Z-score of 20-day rolling std (252-day rolling normalization)
2. **RSI (14-period)**: Relative Strength Index scaled to [-0.5, 0.5]
3. **Market Correlation**: 60-day rolling correlation with QQQ

✅ **Look-Ahead Bias**: All calculations use only past data (verified against training code).

## Troubleshooting

### "Model file not found"

**Solution**: Copy model files from `archive/legacy_versions/simple/` to `daily_adviser/`

```bash
cp archive/legacy_versions/simple/agent_*.zip daily_adviser/
```

### "CURRENT_WEIGHTS must sum to 1.0"

**Solution**: Edit `CURRENT_WEIGHTS` in `daily_advisor.py` to ensure they sum to exactly 1.0:

```python
weights = [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.02]
assert abs(sum(weights) - 1.0) < 0.001
```

### "No data returned"

**Causes:**
- Market is closed (weekends/holidays)
- Yahoo Finance API temporary outage
- Network connectivity issues

**Solution**: Run again during market hours or check internet connection.

### "pandas_ta module not found"

**Solution**: Install missing dependency:

```bash
pip install pandas-ta>=0.3.14
```

## Development

### Running Tests

```bash
# Test weight validation
python -c "
import numpy as np
weights = np.array([0.14]*7 + [0.02])
assert np.isclose(weights.sum(), 1.0, atol=0.001)
print('✓ Weight validation passed')
"
```

### Viewing Logs

Logs are saved as JSON Lines format:

```bash
# View today's log
cat advisory_log_$(date +%Y%m%d).jsonl | jq .

# Extract decisions
cat advisory_log_*.jsonl | jq 'select(.event=="advisor_run_complete") | {date:.timestamp, decision:.decision}'
```

## Files Structure

```
daily_adviser/
├── daily_advisor.py          # Main advisor script
├── get_chat_id.py            # Telegram setup utility
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
├── daily_adviser_guide.md    # User manual (legacy)
├── README.md                 # This file
├── MODELS_README.md          # Model training documentation
├── agent_bull.zip            # Bull agent (trained model)
├── agent_bear.zip            # Bear agent (trained model)
├── agent_sniper.zip          # Sniper agent (trained model)
└── .github/
    └── workflows/
        └── schedule.yml      # GitHub Actions automation
```

## Safety Notes

⚠️ **This is advisory software, not execution software**. The system provides recommendations, but YOU must:

1. Verify market conditions manually (check VIX, news, QQQ chart)
2. Review the agent's logic before executing trades
3. Understand that past performance ≠ future results
4. Start with paper trading to validate the system

## License

Private research project. Not for distribution.

## Support

For issues or questions, review:
1. [daily_adviser_guide.md](daily_adviser_guide.md) - Original usage guide
2. [MODELS_README.md](MODELS_README.md) - Model training details
3. Training notebooks in `archive/legacy_versions/simple/`
