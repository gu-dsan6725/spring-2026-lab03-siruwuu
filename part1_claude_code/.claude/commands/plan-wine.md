---
name: plan-wine
description: Create an implementation plan for the UCI Wine multiclass classification pipeline (EDA, feature engineering, XGBoost CV, evaluation report).
argument-hint: [optional notes or constraints]
disable-model-invocation: true
---

Create a detailed implementation plan for a Wine multiclass classification pipeline using `sklearn.datasets.load_wine()`.

Incorporate any extra constraints from: $ARGUMENTS

Derive a short kebab-case feature name from the task description (e.g., "wine-classification-pipeline" or "wine-xgboost-cv").

Write the plan to `.scratchpad/<feature-name>/plan.md` with the following structure:

## Plan: Wine Classification Pipeline

### Objective
Build a complete pipeline to classify wines into 3 classes, including EDA, feature engineering, XGBoost training with 5-fold stratified CV, and a comprehensive evaluation report.

### Steps
Numbered list of implementation steps. Each step must include:
- What file(s) will be created or modified
- What the code will do
- What artifacts will be saved to `output/`

Required steps to include:
1. EDA script
   - Load Wine dataset
   - Summary statistics, missing values, class balance
   - Distribution plots
   - Correlation heatmap
   - Outlier detection
2. Feature engineering script
   - Create at least 3 derived features
   - Standard scaling (fit on train only)
   - Stratified train/test split
   - Save processed datasets for modeling
3. Model training script
   - Train XGBoost classifier for multiclass
   - 5-fold stratified cross-validation
   - Save best model and metrics
4. Evaluation + reporting
   - Accuracy, macro precision/recall/F1
   - Per-class metrics
   - Confusion matrix + heatmap
   - Feature importance (top features + plot)
   - Write `output/evaluation_report.md` and `output/full_report.md`

### Technical Decisions
- Use `polars` for data manipulation (no pandas)
- Use `matplotlib` for plots (avoid seaborn unless already used elsewhere)
- Use stratified splitting for multiclass classification
- Use consistent logging format per CLAUDE.md
- Keep functions small and constants at top of file

### Testing Strategy
- Add minimal pytest tests where appropriate:
  - Basic smoke tests for data loading and feature engineering outputs
  - Ensure output artifacts are created
  - Ensure scripts run without errors

### Expected Output
List expected artifacts under `output/`, such as:
- EDA plots (distributions, correlation heatmap, outlier summaries)
- Processed datasets (parquet or equivalent)
- Saved model artifact
- Confusion matrix plot
- Feature importance plot
- `evaluation_report.md`
- `full_report.md`

---

After writing the plan, tell the user to review it and provide feedback before proceeding with implementation. Do NOT start building until the user approves the plan.
