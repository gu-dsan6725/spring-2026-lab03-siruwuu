# /train-wine-classifier

Train and evaluate an XGBoost multiclass classifier on the UCI Wine dataset.

## Rules to follow
- Follow `.gemini/GEMINI.md`
- Follow `.agent/rules/code-style-guide.md`
- Follow `.agent/rules/data-quality.md`

## Steps
1. Confirm feature-engineered data exists.
2. Perform stratified train/test split.
3. Apply StandardScaler (fit on train only).
4. Train XGBoost classifier for multiclass classification.
5. Use 5-fold stratified cross-validation.
6. Evaluate model with:
   - accuracy
   - macro precision
   - macro recall
   - macro F1-score
   - per-class precision/recall/F1
7. Generate confusion matrix and save heatmap plot.
8. Generate feature importance ranking and plot.
9. Write `output/evaluation_report.md` with metrics and recommendations.

## Quality checks
- Run:
  - `uv run ruff check --fix part2_antigravity/src/03_xgboost_model.py`
  - `uv run python -m py_compile part2_antigravity/src/03_xgboost_model.py`

## Expected outputs
- Saved model artifact in `output/`
- `output/evaluation_report.md`
- Confusion matrix plot
- Feature importance plot
