---
name: analyze-data
description: Perform exploratory data analysis on the UCI Wine dataset (multiclass classification). Use when asked to explore or profile the Wine data.
argument-hint: [dataset or file path]
---

When performing exploratory data analysis for the Wine classification task, follow these steps:

1. **Load the Wine dataset** using `sklearn.datasets.load_wine()` or from the provided file path.
2. **Convert the data** into a polars DataFrame. Identify feature columns and the target class column.
3. **Compute summary statistics** including mean, median, std, min, and max for each numeric feature.
4. **Check for missing values** and report the count per column.
5. **Check class balance** and report counts for each of the 3 wine classes.
6. **Generate distribution plots** for numeric features using matplotlib histograms.
7. **Create a correlation matrix heatmap** to examine relationships between features.
8. **Identify potential outliers** using the IQR method and report counts per feature.
9. **Log a summary** of key findings using the project's logging format.
10. **Save all plots** to the `output/` directory.

Use polars (not pandas) for all data manipulation.  
Follow the coding standards in CLAUDE.md.

If $ARGUMENTS specifies a dataset or file path, use that. Otherwise, analyze the default Wine dataset.
