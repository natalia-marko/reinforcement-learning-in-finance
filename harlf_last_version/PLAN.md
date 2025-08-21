Here’s a baseline framework for recreating the HARLF approach on your a portfolio using Yahoo price data and Reddit/Yahoo news sentiment.

### 1 Collect and preprocess price data

1. **Choose a long time window** – train on 2003‑2017 and evaluate on 2018‑2024.  

2. **Download adjusted closing prices** for each asset in the portfolio.  Load teh assets from portfolio and a benchmark asset for comparison ('QQQ')and obtain prices from Yahoo via `yfinance`.  Adjusted prices remove splits/dividends so returns are comparable across time.

3. **Compute daily log returns**:

   $$
   r_{t}=\ln\left(\frac{P_{t}}{P_{t-1}}\right)
   $$

   Log returns add rather than multiply across days and allow the model to handle large price changes more gracefully.  The normalized price series in log space make extreme moves easier to visualize.

4. **Derive monthly indicators** such as:

   * **Volatility vector**: monthly standard deviation of daily returns.
   * **Sharpe ratio** (risk‑adjusted return) and **Sortino ratio** (downside‑risk adjusted return).
   * **Max drawdown** and **Calmar ratio** (return divided by maximum loss in the month).
   * **Correlation matrix**: compute the correlation of daily returns across all assets and flatten it into a vector.

   These features give the RL agent context about risk and market structure beyond raw returns.

5. **Normalize features**: use min‑max scaling to bring all numeric features into \[0,1] ranges.  without normalization, agents will overweight high‑priced assets.

### 2 Scrape and validate sentiment data

1. **Define search keywords** for each asset (e.g., “Apple”, “AAPL”, “Apple stock”) and a date range.  The paper’s pseudo‑code (Algorithm 1) loops over assets and months, fetches top news articles per keyword

2. **Use Yahoo Finance and Reddit** as news sources.  For each month and asset:

   * On Yahoo Finance, scrape 10 recent news headlines.  Keep titles and short descriptions.
   * From Reddit, use the official API or Pushshift (free historical access) to fetch posts from relevant subreddits (`r/stocks`, `r/investing`, etc.) containing your keywords.  Extract post titles and body text.

3. **Compute sentiment scores**: run each headline/post through a financial‐domain classifier (FinBERT or similar).  For each document, record $p_{\text{pos}}$, $p_{\text{neg}}$ and set a polarity score $(p_{\text{pos}}-p_{\text{neg}})$.  Aggregate by month and asset:

   $$
   S_{a,t} = \frac{1}{N_{a,t}}\sum_{i=1}^{N_{a,t}} (p_{i,\text{pos}} - p_{i,\text{neg}})
   $$

   where $N_{a,t}$ is the number of articles for asset $a$ in month $t$.

4. **Validate the sentiment signal**: compute the Pearson correlation between your monthly sentiment score $S_{a,t}$ and subsequent monthly log returns $r_{a,t+1}$.  A significant positive/negative correlation indicates that sentiment is predictive.  Use this diagnostic when tuning your NLP model; e.g., adjust the classifier threshold or weighting if the correlation is weak.

### 3 Prepare the RL environment

1. **State representation**: concatenate the financial indicators (returns, volatility, risk ratios, correlation vector) with the sentiment score for each asset.  This produces a compact monthly state vector for the agent.  Including volatility and correlation gives the agent extra context about trend persistence and diversification.

2. **Action space**: allocate continuous portfolio weights for each asset.  The weights must be non‑negative and sum to 1 (no leverage or short‑selling).

3. **Reward function**: define a linear combination of multiple objectives:

   * **Return on investment (ROI)** to encourage higher returns.
   * **Maximum drawdown (MDD)** to penalize large losses.
   * **Sortino or Calmar ratios** to focus on downside risk.
     assign real‑valued weights to each component to balance performance.

4. **Hierarchical agent architecture**:

   * **Base agents** specialize in different modalities—one uses only price indicators; another uses only sentiment.  Each base agent outputs portfolio weights.
   * **Meta‑agents** aggregate base‑agent decisions (e.g., weighted average) and stabilize allocations across modalities.
   * **Super‑agent** merges meta‑agent outputs into the final portfolio, optimizing for long‑term objectives.
     Hierarchical structures improve interpretability and scalability by breaking a complex task into simpler decisions.

5. **Algorithms and training**: implement base and meta agents using RL algorithms such as PPO, SAC, DDPG or TD3; these continuous‑control methods handle portfolio allocation well.  Train agents on the long historical window (e.g., 2003‑2017) and test on the out‑of‑sample period (2018‑2024) to assess generalization.  Fix random seeds to ensure reproducibility.

### 4 Backtesting and performance evaluation

1. **Benchmarks**: compare your agents against an equal‑weighted portfolio and a market index (e.g., S\&P 500).  The paper uses these baselines to contextualize performance.

2. **Metrics**: compute annualized ROI, Sharpe ratio, Sortino ratio, maximum drawdown, and volatility.  Evaluate on the test period to see whether sentiment‑augmented agents outperform price‑only agents and the benchmarks.

3. **Sensitivity analysis**: test different weightings in the reward function, alternative RL algorithms, or feature sets.  Examine how performance changes when sentiment scores are removed or replaced with random noise—this helps quantify their contribution.

4. **Transaction costs (optional)**: the paper suggests adding transaction cost modelling and stress testing as future work.  You can incorporate a small cost per trade to make the simulation more realistic.

### 5 Next steps

* **Expand the sentiment corpus**: once the Yahoo/Reddit baseline works, you can experiment with additional sources (e.g., Google News, SEC filings, X posts).  Larger text corpora provide richer signals but require careful rate‑limit handling.

* **Refine the NLP model**: adapt FinBERT with your own labelled data or test other domain‑specific LLMs.  The paper highlights the promise of lightweight LLMs for sentiment.

* **Improve risk modelling**: incorporate correlation‑based diversification more explicitly, or explore hierarchical RL with options (sub‑policies) to handle different market regimes.

By following these steps—collecting long‑term price data, creating validated sentiment scores, constructing a hierarchical RL environment with log returns and correlation features, and evaluating on a held‑out period—you’ll build a robust HARLF‑style tool for your portfolio.
