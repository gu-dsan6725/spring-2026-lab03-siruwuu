# wine-profiler

Profile the UCI Wine dataset for multiclass classification and produce class-aware insights.

## When to use
Use this skill when asked to profile the Wine dataset, identify discriminative features, or summarize class differences.

## Steps
1. Load the dataset via `sklearn.datasets.load_wine()` and convert to a `polars.DataFrame`.
2. Compute overall summary statistics for all numeric features.
3. Compute class-specific statistics:
   - per-class mean and std for each feature
   - per-class counts and proportions (class balance)
4. Visualize feature distributions per class for at least 3 informative features.
   - save plots to `output/`
5. Discriminative feature analysis:
   - rank features by how different they are across classes
   - acceptable approaches:
     - ANOVA F-score ranking
     - mutual information ranking
     - simple heuristic: max(mean_by_class) - min(mean_by_class)
   - report the top 5 most discriminative features and interpret them briefly
6. Save a short markdown summary to `output/wine_profile.md`.
7. Follow the project logging and code style rules.
