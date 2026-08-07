# Week 6: Pima Indians Diabetes Classification

Model tuning and validation project for the AnalystLab Africa Machine Learning Internship. This project builds a baseline Logistic Regression classifier on the Pima Indians Diabetes dataset, applies hyperparameter tuning via GridSearchCV with 5-fold cross-validation, and compares model performance before and after tuning.

## Problem Statement

Predict whether a patient has diabetes (`Outcome`: 0 = No, 1 = Yes) based on diagnostic health measurements, and evaluate whether hyperparameter tuning meaningfully improves model performance over a default-parameter baseline.

## Dataset

**Pima Indians Diabetes Database**
Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
768 records, 8 feature columns, 1 binary target column (`Outcome`).

Class distribution: ~65.1% No Diabetes, ~34.9% Diabetes.

## Files in This Repository

| File | Description |
|---|---|
| `diabetes.csv` | Raw dataset (Pima Indians Diabetes Database) |
| `Pima Indians Diabetes Classification.ipynb` | Full notebook: preprocessing, baseline model, hyperparameter tuning, cross-validation, evaluation |
| `Week 6 Analyst Lab Report.docx` | Validation report: methodology, results, and key findings |

## Data Preprocessing

Five columns — `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` — contain biologically implausible zero values representing missing data (a known characteristic of this dataset). `Outcome` and `Pregnancies` were excluded from this treatment, as zero is a valid, meaningful value in both.

Zeros in the five affected columns were converted to `NaN`, then imputed based on each column's skewness:
- **Mean imputation** — `Glucose`, `BloodPressure` (skew within ±0.5, roughly symmetric)
- **Median imputation** — `SkinThickness`, `Insulin`, `BMI` (skew above 0.5, right-skewed; median is robust to outliers, mean is not)

**Known limitation:** `Insulin` required imputation for ~49% of records. Median imputation preserves central tendency but compresses the interquartile range, which may understate true variance in downstream cross-validation folds.

## Modeling Approach

- **Algorithm:** Logistic Regression
- **Feature scaling:** `StandardScaler`, fit on training data only, applied to both train and test sets (to prevent data leakage)
- **Train/test split:** 80/20, `random_state=42`
- **Baseline:** Default hyperparameters (`C=1`)
- **Tuning:** `GridSearchCV`, 5-fold cross-validation, grid over `C = [0.01, 0.1, 1, 10, 100]`

## Results

| Metric | Baseline (default C=1) | Tuned (GridSearchCV) |
|---|---|---|
| Accuracy | 0.7532 | 0.7532 |
| Precision | 0.6667 | 0.6667 |
| Recall | 0.6182 | 0.6182 |
| F1-Score | 0.6415 | 0.6415 |
| ROC-AUC | 0.8211 | 0.8231 |

**Best hyperparameters found:** `C=1` (matches scikit-learn's default)

## Key Findings

- GridSearchCV selected `C=1`, the library default — meaning the default regularization strength was already optimal for this model on this data.
- Test accuracy was identical before and after tuning (0.7532), because the tuned model's hyperparameters were functionally the same as the baseline's.
- This is a legitimate, informative result: for Logistic Regression's relatively small hyperparameter space, tuning provided no measurable improvement on this dataset. It does not indicate the tuning process failed.

## Limitations

- Insulin imputation affected ~49% of records; see Data Preprocessing section above.
- Logistic Regression's limited hyperparameter space may partly explain the null tuning result; models with richer hyperparameter spaces (e.g. Gradient Boosting, Random Forest) may show a more pronounced before/after delta.

## How to Run

```bash
pip install pandas numpy scikit-learn
jupyter notebook "Pima Indians Diabetes Classification.ipynb"
```

## Ibinabo Direh

AnalystLab Africa ML Internship
