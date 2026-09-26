"""Train, tune, evaluate, and save the Wellness Tourism prediction model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "artifacts" / "data"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "deployment" / "wellness_tourism_model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "model_building" / "model_metrics.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "model_building" / "classification_report.json"
TARGET_COLUMN = "ProdTaken"
RANDOM_STATE = 42


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [column for column in X.columns if column not in categorical_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    classifier = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, dict]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    report = {
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    return metrics, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tourism model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=str(PROJECT_ROOT / "mlruns"),
        help="MLflow tracking URI. Defaults to a local folder.",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.data_dir / "train.csv")
    test_df = pd.read_csv(args.data_dir / "test.csv")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_pipeline(X_train)
    param_grid = {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [8, 16, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__class_weight": ["balanced", None],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=1,
        verbose=1,
    )

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("wellness_tourism_package_prediction")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="random_forest_grid_search"):
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        metrics, report = evaluate_model(best_model, X_test, y_test)
        joblib.dump(best_model, args.model_path, compress=3)

        mlflow.log_dict(param_grid, "param_grid.json")
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_f1", search.best_score_)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_artifact(str(args.model_path), artifact_path="model")

    metrics_payload = {
        "best_cv_f1": search.best_score_,
        "best_params": search.best_params_,
        "test_metrics": metrics,
    }
    args.metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Model training completed successfully.")
    print(f"Best CV F1: {search.best_score_:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")
    print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Saved model to: {args.model_path}")
    print(f"Saved metrics to: {args.metrics_path}")


if __name__ == "__main__":
    main()
