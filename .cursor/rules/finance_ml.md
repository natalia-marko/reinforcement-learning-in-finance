# Finance ML Research: Cursor Rules (KISS Policy & Notebook-First)

## File Organization
- Core research, analysis, and results must be in Jupyter Notebook (.ipynb) files.
- Use .py files only for reusable utility functions or classes to be imported into notebooks.
- Notebook cells should include all code, visualizations, and essential explanation with markdown.

## Coding Style (KISS Principle)
- Write concise, readable code—avoid unnecessary complexity.
- Use clear names and modular functions for better readability.
- Prefer standard libraries: pandas, numpy, scikit-learn, yfinance.
- Skip over-engineered solutions; always seek a straightforward approach.
- Always validate input and handle exceptions simply.

## Visualizations
- Prefer simple visualizations (e.g., matplotlib/seaborn line/bar/box plots) over verbose printed reports.
- Each analysis step should have a clear, minimal visualization when possible. For example, plot returns, feature importances, residuals.
- Limit visualization to what is meaningful; avoid clutter, 3D plots, or excessive styling.
- Place concise markdown explanations before each chart.
- Do not print verbose DataFrames or text summaries for large datasets—summarize visually instead.

## Project Workflow
- Structure code as small, reusable functions in .py modules.
- Import utility functions into notebooks for use.
- Set random seeds for reproducibility.
- Organize notebooks by workflow step: data preparation, feature engineering, modeling, evaluation.
- Use sklearn pipelines and prefer reusable, clean code.

## Documentation & Testing
- Use docstrings for all custom functions/classes—keep them short and informative.
- Add comments only for non-obvious logic (favor clear code over excessive commenting).
- Simple, assert-style tests for main utility functions in .py modules.

## Output Preferences
- All findings and intermediate steps should be shown as simple plots in notebook cells.
- Avoid long textual output; use concise markdown summaries and visual aids.
- Emphasize actionable insights, not raw data.
