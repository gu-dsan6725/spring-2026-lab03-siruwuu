# Wine Classification ML Project

## Project Overview
This project performs exploratory data analysis and builds an XGBoost classification model on the UCI Wine dataset from scikit-learn. The goal is to classify wines into 3 classes.

Dataset:
- Source: `sklearn.datasets.load_wine()`
- Samples: 178
- Features: 13 numeric features
- Target: 3 wine classes (multiclass classification)

## Coding Standards

### Language and Tools
- Use Python 3.11+
- Use `uv` for package management (never pip)
- Use `polars` for data manipulation (not pandas)
- Use `ruff` for linting and formatting
- Use `pytest` for testing

### Code Style
- Use type annotations for all function parameters (one parameter per line)
- All private functions must start with underscore (`_`) and be placed at the top of the file
- Public functions follow after private functions
- Functions should be no more than 30-50 lines
- Two blank lines between function definitions
- Use multi-line imports

### Logging
Always use this logging configuration:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
```

### Constants

* Do not hard-code constants inside functions
* Declare constants at the top of the file with type annotations

### Modeling Requirements (Wine Classification)

* Use stratified train/test split for the 3 classes
* Apply StandardScaler (fit on train only, transform train/test)
* Train an XGBoost classifier for multiclass classification
* Use 5-fold stratified cross-validation during training and/or tuning
* Report evaluation metrics:

  * Accuracy
  * Macro precision, macro recall, macro F1
  * Per-class precision/recall/F1 if available
  * Confusion matrix (include a heatmap plot if plots are generated)
* Include feature importance:

  * Provide top features ranking
  * Save a feature importance plot if applicable

### After Writing Python Files

* Always run `uv run ruff check --fix <filename>` after writing Python files
* Always run `uv run python -m py_compile <filename>` to verify syntax

### Output

* Save plots to the `output/` directory
* Use `logging.info()` for progress messages
* Pretty-print dictionaries in log messages using `json.dumps(data, indent=2, default=str)`
