# /run-wine-eda

Run exploratory data analysis for the UCI Wine dataset (multiclass classification).

## Rules to follow
- Follow `.gemini/GEMINI.md`
- Follow `.agent/rules/code-style-guide.md`
- Follow `.agent/rules/data-quality.md`

## Steps
1. Confirm working directory is `part2_antigravity/`.
2. Load Wine dataset using `sklearn.datasets.load_wine()`.
3. Convert dataset to a `polars.DataFrame`.
4. Perform EDA:
   - summary statistics
   - missing value counts
   - duplicate row count
   - class balance (3 classes)
   - distribution plots
   - correlation heatmap
   - IQR-based outlier counts
5. Save all plots to `output/`.
6. Log progress and findings.

## Quality checks
- Run:
  - `uv run ruff check --fix part2_antigravity/src/01_eda.py`
  - `uv run python -m py_compile part2_antigravity/src/01_eda.py`

## Expected outputs
- Distribution plots in `output/`
- Correlation heatmap in `output/`
- Logged EDA summary
