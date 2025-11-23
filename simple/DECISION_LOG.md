# Decision Log: Transition to MLP-Only Pipeline

**Date:** 2025-11-23
**Decision:** Archive LSTM model and focus exclusively on MLP.

## Rationale

We have compared the performance of the MLP (Multi-Layer Perceptron) and LSTM (Long Short-Term Memory) models using the Expanded Feature Set (175 features) and Lean Feature Selection.

### 1. Performance Comparison
The training stability analysis revealed nearly identical performance characteristics between the two architectures:

| Metric | LSTM (Fold 3) | MLP (Fold 3) |
| :--- | :--- | :--- |
| **Train Final Reward** | 43.19 | 43.62 |
| **Val Final Reward** | 3.09 | 3.23 |
| **Overfit Gap** | 40.10 | 40.39 |

Both models exhibited significant overfitting in later folds, indicating that the limiting factor is likely the data regime or feature stationarity rather than the model architecture itself. The LSTM did not provide a generalization advantage to justify its added complexity.

### 2. Complexity & Efficiency
- **Training Speed:** MLP trains significantly faster than LSTM.
- **Hyperparameters:** MLP has fewer hyperparameters to tune (no sequence length, hidden states, etc.).
- **Maintenance:** A single model pipeline is easier to maintain, debug, and improve.

### 3. Conclusion
Given that the LSTM offers **no performance benefit** over the MLP for this specific dataset and feature set, we are choosing the **Principle of Parsimony (Occam's Razor)**. We will proceed with the simpler, faster, and equally effective MLP model.

## Action Plan
1.  Archive `lstm/` directory to `archive/lstm/`.
2.  Promote `mlp/` to be the primary `src/` directory.
3.  Rename scripts for clarity (removing `_mlp` suffixes).
4.  Update notebooks and imports to reflect the new structure.
