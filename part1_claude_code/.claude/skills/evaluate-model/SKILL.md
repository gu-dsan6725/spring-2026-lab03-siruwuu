---
name: evaluate-model
description: Evaluate a trained Wine multiclass classification model and generate a performance report.
argument-hint: [model path or description]
---

When evaluating a trained Wine classification model, follow these steps:

1. **Load the trained model** from the `output/` directory.
2. **Load the test dataset** and generate predictions.
3. **Compute classification metrics**:
   - Accuracy
   - Macro precision, macro recall, macro F1-score
   - Per-class precision, recall, and F1-score
4. **Generate a confusion matrix** and save a confusion matrix heatmap plot.
5. **Create a feature importance chart** if the model supports it (e.g., XGBoost).
6. **Write an evaluation report** to `output/evaluation_report.md` containing:
   - A metrics summary table
   - Confusion matrix summary
   - Feature importance summary
   - Key findings and observations
   - Recommendations for improvement
7. **Save all plots** to the `output/` directory.

Use polars for any data handling.  
Log all metrics using the project's logging format.  
Follow the coding standards in CLAUDE.md.

If $ARGUMENTS specifies a model path, use that. Otherwise, look for models in the `output/` directory.
