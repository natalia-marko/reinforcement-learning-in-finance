# Data Exploration Notebook Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Workflow Overview](#workflow-overview)
4. [Detailed Methodology](#detailed-methodology)
5. [Output Files](#output-files)
6. [Interpretation Guide](#interpretation-guide)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Introduction

### Purpose

The `data_exploration.ipynb` notebook performs comprehensive feature analysis and selection for the reinforcement learning portfolio management system. The primary goal is to identify the most informative features while reducing dimensionality to:

- Improve model training efficiency
- Reduce overfitting risk
- Maintain predictive performance
- Ensure feature diversity across categories

### Key Objectives

1. **Feature Quality Assessment**: Identify features with missing data, zero variance, or low information content
2. **Redundancy Detection**: Find highly correlated features that provide redundant information
3. **Dimensionality Analysis**: Determine intrinsic dimensionality using PCA
4. **Importance Ranking**: Rank features by predictive power using SHAP values
5. **Feature Selection**: Select 30-40 core features ensuring diversity and predictive power
6. **Performance Validation**: Verify selected features maintain predictive performance

## Prerequisites

### Required Libraries

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
shap  # Optional but recommended
```

### System Components

- `portfolio_system.py`: Contains `SystemConfig` and `DataPipeline` classes
- `data/weekly_market_data_cache.parquet`: Cached weekly market data (auto-generated)
- `data/selected_features.json`: Previously selected features (optional, auto-loaded if available)

### Data Requirements

- Weekly OHLCV data for configured tickers (default: NVDA, MU, AAPL, AMD, ASML, MSFT, GOOG)
- Date range: 2015-01-01 to present (configurable in `SystemConfig`)
- Minimum data: At least 2 years of weekly data for meaningful analysis

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Loading & Feature Engineering                       │
│    - Load weekly market data                                │
│    - Engineer technical indicators                           │
│    - Create rolling statistics                              │
│    - Add regime features                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Feature Statistics & Quality Assessment                 │
│    - Export features to CSV                                │
│    - Calculate missing/zero percentages                    │
│    - Rank features by variance                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Correlation Analysis                                     │
│    - Compute correlation matrix                            │
│    - Identify highly correlated pairs (|r| > 0.95)        │
│    - Remove redundant features (keep higher variance)       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Principal Component Analysis (PCA)                      │
│    - Standardize features                                  │
│    - Fit PCA on all components                             │
│    - Analyze explained variance (90%, 95%, 99%)           │
│    - Examine component loadings                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. SHAP Analysis for Feature Importance                     │
│    - Prepare target: future 1-week log returns             │
│    - Train Random Forest surrogate model                  │
│    - Compute SHAP values                                   │
│    - Rank features by mean |SHAP|                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Feature Selection                                        │
│    - Select top features by SHAP importance               │
│    - Categorize features for diversity                     │
│    - Ensure representation across categories               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Export Selected Features                                 │
│    - Export to CSV (with importance scores)                │
│    - Export to JSON (for system integration)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. Performance Validation                                   │
│    - Compare full vs selected feature sets                │
│    - Measure performance retention                         │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Methodology

### 1. Data Loading and Feature Engineering

#### Process

The `DataPipeline` class handles:
- **Weekly Aggregation**: Converts daily OHLCV data to weekly frequency (Friday close)
- **Feature Engineering**: Creates 35+ features including:
  - **Returns**: 1w, 2w, 4w, 8w, 13w, 26w historical returns
  - **Volatility**: Annualized volatility across multiple windows
  - **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
  - **Rolling Statistics**: Sharpe ratios, skewness, kurtosis, max drawdown
  - **Regime Features**: Market volatility regimes, trend indicators, beta to market
  - **Relative Features**: Cross-sectional relative returns and ranks

#### Data Quality Handling

- **NaN Values**: Forward fill within ticker groups, then backward fill, finally fill with 0
- **Infinite Values**: Replace with NaN, then fill with 0
- **Outliers**: Clip to 1st and 99th percentiles
- **Zero Variance**: Remove features with zero variance

#### Feature Filtering

If `selected_features.json` exists, the pipeline automatically filters to only those features. This ensures consistency between exploration and training phases.

### 2. Feature Statistics Analysis

#### Quality Metrics

- **Missing Percentage**: `(NaN count / total rows) × 100`
  - Threshold: >10% missing may need special handling
- **Zero Percentage**: `(zero count / total rows) × 100`
  - Threshold: >50% zeros may indicate sparse/binary features
- **Variance**: Measure of feature spread
  - Higher variance = more information content
  - Zero variance = constant feature (should be removed)

#### Interpretation

- Features with high variance are generally more informative
- Missing data >10% may indicate data quality issues
- Zero variance features provide no information and should be removed

### 3. Correlation Analysis

#### Methodology

1. **Data Preparation**: Use training split only to prevent data leakage
2. **Correlation Computation**: Pearson correlation coefficient for all feature pairs
3. **Redundancy Detection**: Identify pairs with |correlation| > 0.95
4. **Feature Removal**: For correlated pairs, remove feature with lower variance

#### Correlation Threshold Rationale

**Threshold: 0.95**

- **Lower threshold (e.g., 0.90)**: Would remove more features but risk losing complementary information
- **Higher threshold (e.g., 0.98)**: Too conservative, would miss obvious redundancies
- **0.95**: Balances redundancy removal with information preservation

**Why Absolute Value?**
Both positive and negative high correlations indicate redundancy. For example, if `returns_4w` and `returns_8w` have correlation -0.96, they provide redundant information (one is essentially the inverse of the other).

#### Feature Removal Strategy

When two features are highly correlated:
- Keep the feature with **higher variance** (more informative)
- Remove the feature with **lower variance** (less informative)

Rationale: Higher variance features contain more information, so we prefer to keep them.

### 4. Principal Component Analysis (PCA)

#### Methodology

1. **Data Preparation**:
   - Remove highly correlated features (from correlation analysis)
   - Remove zero-variance features
   - Standardize features (mean=0, std=1) using `StandardScaler`

2. **PCA Computation**:
   - Fit PCA on all components to analyze full variance structure
   - Calculate explained variance ratio for each component
   - Compute cumulative explained variance

3. **Variance Thresholds**:
   - **90% variance**: Good balance between reduction and information retention
   - **95% variance**: Conservative reduction, retains most information
   - **99% variance**: Near-complete information capture

#### Why Standardize?

Features with larger scales would dominate PCA. Standardization ensures all features contribute equally to principal components.

#### Component Analysis

Each principal component is a linear combination of original features:
- **Loadings**: Weights showing feature contributions to each component
- **High loadings** (positive or negative): Strong contribution
- **Components ordered**: First component explains most variance

#### Use Case

We don't use PCA-transformed features directly. Instead:
- Use PCA insights to understand intrinsic dimensionality
- Inform feature selection by ensuring we keep features that contribute to high-variance components
- Validate that our feature selection captures the main variance structure

### 5. SHAP Analysis

#### Target Variable

**Future 1-week log returns**:
- Calculated as: `log(next_week_close / current_week_close)`
- Shifted by -1 to get forward-looking returns
- Grouped by ticker to ensure proper time ordering

**Why log returns?**
- Log returns are symmetric and additive over time
- System configuration uses log returns (`use_log_returns=True`)
- Better statistical properties for financial modeling

**Note**: The `log_returns` column is not available in `featured_data` after engineering, so we calculate it directly from close prices.

#### Surrogate Model: Random Forest

**Why Random Forest?**
1. Handles non-linear relationships well
2. Works efficiently with SHAP TreeExplainer
3. Provides feature importance as fallback if SHAP unavailable
4. Robust to outliers and missing values

**Hyperparameters**:
- `n_estimators=100`: Number of trees (more = better but slower)
- `max_depth=10`: Limit tree depth to prevent overfitting
- `min_samples_split=20`: Minimum samples to split (prevents overfitting)
- `random_state=42`: For reproducibility

**Training Sample**: Up to 5000 samples (for computational efficiency)

#### SHAP Computation

**TreeExplainer**: Optimized for tree-based models, faster than KernelExplainer

**Sample Size**: 500 samples
- SHAP complexity: O(n_samples × n_features × n_trees)
- Trade-off: Larger samples = better estimates but longer computation time
- 500 samples balances accuracy with computation time

**SHAP Values**:
- **SHAP Value**: Contribution of a feature to the prediction for a specific sample
- **Mean |SHAP|**: Average absolute contribution across samples
- **Higher mean |SHAP|**: Feature has stronger predictive power

#### Fallback Method

If SHAP is not available, falls back to Random Forest feature importances (Gini importance), which measures how much each feature contributes to reducing impurity in the trees.

### 6. Feature Selection

#### Selection Strategy

1. **Start with Top SHAP Features**: Select top N features by SHAP importance (target: 35 features)

2. **Categorize Features**: Group features into categories to ensure diversity:
   - **Returns**: Historical return features (1w, 2w, 4w, 8w, 26w)
   - **Volatility**: Volatility measures across different windows
   - **SMA**: Simple moving average features
   - **Price Ratios**: Price-to-SMA ratios, high-low ratios, close-open ratios
   - **Volume**: Volume change and volume ratios
   - **Technical**: Technical indicators (RSI, MACD, Bollinger Bands, ATR)
   - **Sharpe**: Risk-adjusted return measures
   - **Regime**: Market regime indicators (trend, volatility regimes, beta)
   - **Relative**: Cross-sectional relative features (relative returns, ranks)
   - **Other**: Additional features (drawdowns, skewness, kurtosis)

3. **Diversity Check**: Ensure representation across categories (prevents over-reliance on one feature type)

4. **Final Selection**: If fewer than target features, add more from importance ranking

#### Rationale for 30-40 Features

- **Too few (<20)**: May miss important predictive signals
- **Too many (>50)**: Increases overfitting risk, slows training, adds noise
- **30-40**: Balance between information content and model complexity

### 7. Performance Validation

#### Methodology

1. **Data Preparation**:
   - Use same target variable (future returns) for both models
   - Align indices to ensure consistent comparison
   - Split into train/validation sets (80/20)

2. **Model Training**:
   - Train Random Forest on full feature set
   - Train Random Forest on selected feature set
   - Same hyperparameters for fair comparison

3. **Performance Metrics**:
   - R² score on validation set
   - Performance retention: `(selected_score / full_score) × 100%`

#### Interpretation

- **Performance Retention >95%**: Excellent - selected features retain almost all predictive power
- **Performance Retention 90-95%**: Good - acceptable trade-off for reduced complexity
- **Performance Retention <90%**: Warning - may have removed important features

#### Important Notes

- **Negative R² scores** are common in financial prediction tasks due to market efficiency
- The comparison is relative: if both models have negative R², we compare which is less negative
- The goal is feature selection, not necessarily achieving positive R²
- Negative R² indicates the model performs worse than predicting the mean

## Output Files

### 1. `data/all_features_export.csv`

**Format**: CSV file with all features
**Columns**: `date`, `ticker`, and all feature columns
**Usage**: External analysis, spreadsheet tools, other languages
**Size**: ~(n_samples × n_features) rows

### 2. `data/selected_features.csv`

**Format**: CSV file with selected features and importance scores
**Columns**:
- `feature`: Feature name
- `importance`: SHAP importance score (or RF importance if SHAP unavailable)
- `rank`: Feature rank (1 = most important)

**Usage**: Analysis, reference, documentation

### 3. `data/selected_features.json`

**Format**: JSON array of feature names
**Example**:
```json
[
  "market_volatility",
  "trend_strength",
  "returns_26w",
  ...
]
```

**Usage**: 
- Imported by `DataPipeline` to automatically filter features
- Easy to use in Python: `json.load(open('selected_features.json'))`
- Ensures consistency between exploration and training phases

## Interpretation Guide

### Correlation Analysis Results

**High Correlation Count = 0**:
- Good: No highly redundant features detected
- All features provide unique information

**High Correlation Count > 0**:
- Review the correlated pairs
- Check if removal makes sense (consider domain knowledge)
- Verify that removed features have lower variance

### PCA Results

**90% Variance Components**:
- If low (e.g., <20): Features are relatively independent, good diversity
- If high (e.g., >30): Features are highly correlated, consider more aggressive reduction

**Component Loadings**:
- Features with high loadings (|loading| > 0.3) contribute strongly to that component
- If many features load on first component: May indicate a dominant factor (e.g., market regime)

### SHAP Importance

**Top Features**:
- Features with highest mean |SHAP| are most important for prediction
- Typically includes: market regime features, long-term returns, volatility measures

**Low Importance Features**:
- Features with very low SHAP values (< 0.0001) may be candidates for removal
- However, consider feature diversity - some low-importance features may still be valuable

### Performance Comparison

**Performance Retention >95%**:
- Excellent: Feature selection successful, minimal information loss
- Proceed with selected features

**Performance Retention 90-95%**:
- Good: Acceptable trade-off for reduced complexity
- Monitor model performance during training

**Performance Retention <90%**:
- Warning: May have removed important features
- Consider:
  - Reviewing removed features
  - Increasing target feature count
  - Checking if important features were accidentally removed

## Troubleshooting

### Issue: "Column not found: log_returns"

**Cause**: The `log_returns` column is not available in `featured_data` after feature engineering.

**Solution**: The notebook calculates log returns directly from close prices:
```python
train_data_sorted['next_close'] = train_data_sorted.groupby('ticker')['close'].shift(-1)
train_data_sorted['future_return'] = np.log(train_data_sorted['next_close'] / train_data_sorted['close'])
```

### Issue: SHAP computation is very slow

**Cause**: SHAP is computationally expensive (O(n_samples × n_features × n_trees)).

**Solutions**:
- Reduce `shap_sample_size` (default: 500)
- Use fewer trees in Random Forest (reduce `n_estimators`)
- Use fallback to RF feature importances (set `SHAP_AVAILABLE = False`)

### Issue: All features have high correlation

**Cause**: Features may be derived from similar underlying data (e.g., multiple return windows).

**Solution**:
- This is expected for related features (e.g., returns_4w and returns_8w)
- The correlation analysis will identify and remove redundant ones
- Consider domain knowledge when interpreting results

### Issue: Negative R² scores

**Cause**: Common in financial prediction due to market efficiency.

**Solution**:
- This is expected and not necessarily a problem
- Focus on relative performance (selected vs full feature set)
- The goal is feature selection, not achieving positive R²

### Issue: Selected features have high correlation

**Cause**: Feature selection prioritizes importance over correlation.

**Solution**:
- Review the correlation validation output (Section 6)
- If many high correlations (>0.90), consider:
  - Lowering correlation threshold in selection process
  - Manually removing highly correlated features from selected set
  - Increasing target feature count to allow more diversity

### Issue: Empty training split

**Cause**: All data may have been dropped during feature engineering.

**Solution**:
- Check data quality (missing values, date ranges)
- Verify feature engineering logic
- Check if selected_features.json is filtering out all features

## Best Practices

### Running the Notebook

1. **Run cells sequentially**: Each section builds on previous results
2. **Check outputs**: Verify each section produces expected results
3. **Review warnings**: Pay attention to data quality warnings
4. **Save results**: Export selected features before closing notebook

### Modifying Parameters

**Correlation Threshold**:
- Default: 0.95
- More aggressive: Lower to 0.90 (removes more features)
- More conservative: Raise to 0.98 (removes fewer features)

**PCA Variance Thresholds**:
- Default: 90%, 95%, 99%
- Adjust based on desired information retention
- Lower thresholds (e.g., 85%, 90%, 95%) for more aggressive reduction

**Feature Selection Target**:
- Default: 35 features
- Increase (e.g., 40-50) if performance retention is low
- Decrease (e.g., 25-30) if model is overfitting

**SHAP Sample Size**:
- Default: 500 samples
- Increase for better estimates (slower computation)
- Decrease for faster computation (less accurate estimates)

### Regular Updates

1. **Re-run periodically**: As new data becomes available, re-run feature selection
2. **Monitor performance**: Track how selected features perform in production
3. **Iterate**: Refine feature selection based on model performance
4. **Document changes**: Keep track of feature selection changes and their impact

### Integration with Portfolio System

1. **Use selected_features.json**: The `DataPipeline` automatically loads this file
2. **Verify consistency**: Ensure exploration and training use same features
3. **Monitor training**: Check if reduced feature set improves training stability
4. **Compare performance**: Track validation/test performance with selected features

### Feature Engineering Considerations

When adding new features:
1. Re-run correlation analysis to check for redundancy
2. Verify new features don't have zero variance
3. Check SHAP importance to see if new features are selected
4. Validate that new features improve model performance

### Performance Optimization

1. **Use caching**: The pipeline caches weekly data (7-day validity)
2. **Sample sizes**: Adjust SHAP and RF sample sizes based on available compute
3. **Parallel processing**: Random Forest uses `n_jobs=-1` for parallelization
4. **Memory management**: Large datasets may require chunking or sampling

