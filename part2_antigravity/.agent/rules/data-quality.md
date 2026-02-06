# Data Quality Rules - Wine Classification

These rules apply to the Wine multiclass classification pipeline.

## Required checks
- Always report dataset shape, schema, and null counts.
- Always check duplicate rows and report the count.
- Always check class balance (counts per class for 3 classes).
- Always perform outlier detection (IQR-based) and report outlier counts per feature.
- Always generate and save:
  - distribution plots for numeric features
  - a correlation heatmap

## Modeling data integrity
- Use stratified splits for train/test.
- Apply StandardScaler using train-only fit.
- Do not leak test data into training or scaling.

## Artifacts
- Save plots, metrics, and reports under `output/`.
- If an artifact cannot be generated, write a clear note in the report explaining what is missing and why.

## Quality steps
- Run `uv run ruff check --fix` and `uv run python -m py_compile` on new/changed Python files as explicit steps.
