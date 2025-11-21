## Feature Categories Overview

Below is an organized overview of the primary feature groups typically used:

### Categories and Example Features

- **Momentum & Trend** (18 features)  
  Examples: `return_1w`, `return_4w`, `rsi_14d`, `roc_4w`, `macd_histogram`, `stochastic_14w`, `price_to_sma_4w`, `price_to_ema_8w`

- **Volatility & Risk** (16 features)  
  Examples: `volatility_4w`, `atr_14d`, `bb_width`, `bb_position`, `downside_volatility_4w`, `upside_volatility_4w`

- **Volume & Liquidity** (30 features)  
  Examples: `obv`, `obv_roc_4w`, `relative_volume`, `chaikin_money_flow`, `volume_oscillator`, `mfi_14d`, `vwap_distance`

- **Risk-Adjusted Performance** (15 features)  
  Examples: `sharpe_4w`, `sortino_13w`, `calmar_26w`, `rolling_calmar_26w`

- **Drawdown & Path Risk** (17 features)  
  Examples: `current_drawdown`, `max_drawdown_13w`, `ulcer_index`, `recovery_factor`, `pain_index`

- **Statistical/Distribution** (6 features)  
  Examples: `skew_13w`, `kurt_26w`
  
- **Market Relationships** (7 features)  
  Examples: `beta_to_market`, `correlation_to_qqq`, `relative_return`, `bench_corr_12w`

- **Volatility Regime (Macro)** (6 features)  
  Examples: `vix_level`, `vix_percentile_52w`, `realized_vs_implied_vol`

- **Interest Rates & Yield** (8 features)  
  Examples: `treasury_10y_yield`, `yield_curve_slope`, `ted_spread`, `credit_spread_hy`

- **Inflation & Growth** (9 features)  
  Examples: `inflation_yoy`, `core_inflation_yoy`, `gdp_growth_estimate`, `ism_manufacturing`

- **Market Breadth** (8 features)  
  Examples: `advance_decline_line`, `new_highs_lows_diff`, `pct_above_sma_200`, `mcclellan_oscillator`

- **Equity Conditions** (6 features)  
  Examples: `sp500_distance_from_ath`, `market_pe_ratio`, `equity_risk_premium`

- **Liquidity & Flows** (7 features)  
  Examples: `fed_balance_sheet_change`, `m2_money_supply_growth`, `etf_flow_equity`

- **Cross-Asset & FX** (6 features)  
  Examples: `dxy_dollar_index`, `gold_price_change`, `oil_price_wti`, `stock_bond_correlation_60d`

- **Regime Probabilities** (3 features)  
  Examples: `regime_probability_bull`, `regime_probability_bear`, `risk_parity_regime`

- **Portfolio State** (3 features)  
  Examples: `current_portfolio_weights`, `cash_balance`, `shares_held`

- **Technical/Raw** (2 features)  
  Examples: `log_returns_60d`, `ohlc_normalized`

- **Latent & Advanced** (4 features)  
  Examples: `autoencoder_latent`, `svd_principal_components`, `news_embeddings`, `asset_covariance_matrix`

---

### Additional Calendar Features

These features help capture calendar and cyclical market effects:
- **sin_month, cos_month:**  
  Capture monthly seasonality such as the January effect, summer doldrums, and year-end rallies.
- **sin_dow, cos_dow:**  
  Encode weekday effects (e.g., Monday effect, impact of options expiry, weekend gap risk).
- **sin_dom, cos_dom:**  
  Highlight turn-of-month patterns, window dressing, and payroll/date-related flows.

---

## Implementation Notes for RL Agents

**Multi-Timeframe Focus:**  
About 40% of indicators are available in different timeframes (suffixes like `_4w`, `_13w`). This helps agents learn both short-term and long-term patterns (e.g., `return_1w` vs. `calmar_52w`).

**Asset vs. Macro Split:**  
Roughly 65% of features are asset-specific (unique per stock), and 35% are macro/global (same for all assets, e.g., `vix_level`), supporting regime adaptation.

**Deduplication Logic:**  
Only exact string duplicates are removed. For example, `max_dd_4w` and `max_drawdown_4w` remain as separate features (though they're similar).

**RL Utility:**  
This diverse feature set enables expressive RL state spaces (PPO, DDPG, SAC). It supports interaction between drawdown/Sortino for reward shaping, volume for confirmation, and macro data for regime tracking.  
**Tip:** Start with the 50–100 most important features for your strategy to avoid dimensionality issues, then reduce or add as needed.

**Normalization Recommendation:**  
Normalize all features using z-score or min-max scaling over a 252-day rolling window. This improves neural network stability.

---

## RL State Space Implementation Tips

- **Broadcast calendar features**: Use the same calendar feature vector (e.g., sin/cos representations) for all tickers each day.
- **Normalize with other features**: Integrate calendar and price features in your rolling 252-day normalization pipeline.
- **Feature importance**: After training, use SHAP or integrated gradients to assess feature importance. Calendar features often rank among the top 20.
- **Avoid overfitting**: Only add features that clearly benefit your RL agent. Extra calendar variables (e.g., hour-of-day for daily bars) can add noise.
- **Pairing**: Combining calendar effects with volume or drawdown often yields robust policies (e.g., “high volume + end-of-month + low drawdown → increase weight”).

---