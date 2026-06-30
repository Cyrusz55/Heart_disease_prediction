import os
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from scripts.clean import CLEAN_PATH


MODEL_PATH = "models/heart_disease_model.joblib"
TARGET_COL = 'target'

cat_col = [
    "sex",
    "chest_pains",
    "fasting_blood_sugar",
    "rest_ecg",
    "exercise_induced_angina",
    "slope",
    "thalassemia",
]

num_col = [
    "age",
    "resting_blood_pressure",
    "cholesterol",
    "max_heart_rate",
    "oldpeak",
]

def get_X_y(df: pd.DataFrame):
    X = df[num_col + cat_col]
    y = df[TARGET_COL]
    return X, y

def build_pipeline():
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_col),
        ('cat', categorical_pipeline, cat_col)
    ])

    model = Pipeline(
        steps = [
            ('preprocessor', preprocessor),
            ('SVM', SVC(
                kernel="rbf",
                probability=True,
                random_state=42,

            ))
        ]
    )
    return model
def train_model(df:pd.DataFrame):
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    model = build_pipeline()


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        make_scorer,
    )
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(
            precision_score, average="weighted", zero_division=0
        ),
        "recall": make_scorer(
            recall_score, average="weighted", zero_division=0
        ),
        "f1": make_scorer(
            f1_score, average="weighted", zero_division=0
        ),
        "roc_auc": "roc_auc",
    }

    cv_scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring= scoring,
        n_jobs=-1,
        return_train_score=False
    )
    model.fit(X_train, y_train)
    best_f1_score = np.max(cv_scores['test_f1'])
    mean_f1_score = np.mean(cv_scores['test_f1'])

    print(f"Best Fold F1 Score: {best_f1_score}")
    print(f"Mean Cross-Validated F1 Score: {mean_f1_score}")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)


    metadata = {
        "model": model,
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    }

    print(f"\nTest Set Results (Best Estimator):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    os.makedirs('models', exist_ok = True)
    joblib.dump(metadata, MODEL_PATH)
    print(f"Model saves to {MODEL_PATH}")

    return model, X_test, y_test
def predict_heart_disease(model, new_data: pd.DataFrame):
    return model.predict(new_data)

def load_model():
    from pathlib import Path
    # tujaribu local
    local_path = Path("models/heart_disease_model.joblib")
    if local_path.exists():
        return joblib.load(local_path)

    # huggingface cache
    cache_path = Path("models/models--cyrusnx--heart_disease_model/snapshots")
    if cache_path.exists():
        snapshots = list(cache_path.glob("*/heart_disease_model.joblib"))
        if snapshots:
            return joblib.load(snapshots[0])
    raise FileNotFoundError("Model file not found")

def predict_single(input_data: dict):
    model = load_model()["model"]
    df = pd.DataFrame([input_data])
    actual_state = model.predict(df)
    return int(actual_state)

if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    train_model(df)