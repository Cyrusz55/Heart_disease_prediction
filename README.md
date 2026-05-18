# Heart Disease Prediction

## Project overview

This repository contains a notebook-driven machine learning workflow for heart disease prediction.
The main analysis lives in `book1.ipynb`, where several processed heart disease datasets are loaded from `heart+disease/`, combined, cleaned, preprocessed, and used to train multiple classification models.

The notebook works with the original multi-class target values `0` to `4`, which the code comments describe as increasing disease severity.
Although the notebook introduction mentions a binary classification goal, the implemented pipeline evaluates a five-class problem.

## Motivation

The project explores whether routine clinical variables can support automated prediction of heart disease severity.
The emphasis is on a reproducible modeling workflow: consistent preprocessing, leakage-aware validation, class-imbalance handling, and comparison of several baseline and tree-based classifiers.

## Features

- Combines multiple processed heart disease datasets into a single analysis table.
- Converts all columns to numeric form and inspects missing values, duplicates, and class balance.
- Uses a scikit-learn / imbalanced-learn preprocessing pipeline with:
  - median imputation for numeric features,
  - most-frequent imputation for categorical features,
  - standard scaling for numeric features,
  - one-hot encoding for categorical features.
- Applies SMOTE inside the training pipeline.
- Benchmarks multiple classifiers under stratified cross-validation.
- Reports accuracy, weighted precision, weighted recall, weighted F1, and multiclass ROC-AUC.
- Saves the final fitted pipeline as `best_pipeline.pkl` and the model comparison table as `model_results_cv.csv`.

## Data pipeline

The data flow implemented in `book1.ipynb` is:

1. Load processed heart disease records from `heart+disease/`.
2. Assign a common column schema.
3. Concatenate the datasets into one dataframe.
4. Coerce all fields to numeric values.
5. Review duplicates and missing values.
6. Drop duplicate rows.
7. Split features and target.
8. Create train/test splits with stratification.
9. Apply preprocessing, imputation, encoding, scaling, and SMOTE inside the model pipeline.
10. Cross-validate and fit candidate models.
11. Evaluate the held-out test set.
12. Persist the best pipeline and summary metrics.

## Feature engineering

The notebook uses a small, explicit feature-engineering layer rather than heavy transformation.

### Feature selection

- `num_major_vessels` is dropped before modeling.
- The remaining variables are split into numerical and categorical groups.

### Numerical preprocessing

- `age`
- `resting_blood_pressure`
- `cholesterol`
- `max_heart_rate`
- `oldpeak`

These features are imputed with the median and standardized.

### Categorical preprocessing

- `sex`
- `chest_pains`
- `fasting_blood_sugar`
- `rest_ecg`
- `exercise_induced_angina`
- `slope`
- `thalassemia`

These features are imputed with the most frequent value and one-hot encoded with `drop='first'`.

### Class imbalance handling

SMOTE is applied within the training pipeline to reduce class imbalance during model fitting and cross-validation.

## Models and evaluation

The notebook evaluates the following models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost
- Support Vector Machine
- K-Nearest Neighbors
- Gaussian Naive Bayes

Validation uses `StratifiedKFold` with 5 folds.
Reported metrics include:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1
- Multiclass ROC-AUC (`ovr`, weighted)

The notebook also includes optional grid search tuning for the CV winner.

## Results

The saved comparison table is `model_results_cv.csv`. Based on that file, the best untuned model by cross-validated weighted F1 is **Gradient Boosting**.

| Model | CV F1 Mean | Test F1 | CV ROC-AUC Mean | Test ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Gradient Boosting | 0.5193 | 0.5208 | 0.7728 | 0.7845 |
| Logistic Regression | 0.5011 | 0.4996 | 0.7613 | 0.7990 |
| Random Forest | 0.4981 | 0.5006 | 0.7686 | 0.8062 |
| XGBoost | 0.4913 | 0.4759 | 0.7618 | 0.7854 |
| SVM | 0.4860 | 0.5096 | 0.7571 | 0.7915 |
| AdaBoost | 0.4776 | 0.4832 | 0.7183 | 0.7327 |
| Decision Tree | 0.4667 | 0.4802 | 0.6711 | 0.6799 |
| KNN | 0.4575 | 0.5018 | 0.6958 | 0.7386 |
| Naive Bayes | 0.4432 | 0.4755 | 0.7466 | 0.7809 |

Interpretation should remain cautious: the best scores are modest, and weighted averages can hide poor performance on minority classes.

## Limitations

- The workflow is notebook-first; there is no standalone training script or inference API.
- The repository does not document an external deployment target.
- The target is treated as a 5-class problem in the implementation, even though the notebook introduction discusses a binary framing.
- The evaluation relies on a single stratified train/test split plus cross-validation on the training set.
- Weighted metrics may obscure class-specific errors.
- The repository snapshot includes empty `data/` and `models/` directories; the active artifacts currently live at the repository root.

## Future work

- Convert the notebook workflow into a reproducible Python module or command-line training script.
- Add a dedicated inference entrypoint for `best_pipeline.pkl`.
- Report macro-averaged metrics and per-class performance alongside weighted metrics.
- Save the generated figures alongside the documentation assets.
- Explore feature selection and additional hyperparameter tuning.
- Review whether the project should remain a 5-class severity task or be reformulated as binary classification.

## Installation

The repository does not include a pinned dependency file, so install the packages used by the notebook manually.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib jupyter
```

If `xgboost` is unavailable in your environment, the rest of the notebook still runs for the non-XGBoost models.

## Usage

1. Open `book1.ipynb` in Jupyter or PyCharm.
2. Run the notebook cells from top to bottom.
3. The notebook will generate:
   - `heart_disease_combined.csv`
   - `model_results_cv.csv`
   - `best_pipeline.pkl`
4. Place exported plots in `assets/figures/`.

## Project structure

```text
Heart_disease_prediction/
├── best_pipeline.pkl
├── book1.ipynb
├── heart_disease_combined.csv
├── model_results_cv.csv
├── README.md
├── assets/
│   └── figures/
│       └── README.md
├── data/
├── heart+disease/
│   ├── processed.cleveland.data
│   ├── reprocessed.hungarian.data
│   ├── processed.switzerland.data
│   ├── processed.va.data
│   └── other source files and documentation
└── models/
```

## Figures

Export notebook charts to `assets/figures/` and reference them from this README when needed.


