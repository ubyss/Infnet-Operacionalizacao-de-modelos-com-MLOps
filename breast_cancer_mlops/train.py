from __future__ import annotations

import json
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_prep import ensure_or_create_splits
from .paths import repo_root


def _root() -> Path:
    return repo_root()


def build_pipelines():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grids = []
    base_rf = {"clf__n_estimators": [100, 200], "clf__max_depth": [8, 12, None]}
    pipe_baseline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ]
    )
    grids.append(
        {
            "name": "rf_standard_scaler",
            "pipe": pipe_baseline,
            "grid": {**base_rf},
            "cv": cv,
        }
    )
    pipe_pca = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=42)),
            ("clf", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ]
    )
    grids.append(
        {
            "name": "rf_pca",
            "pipe": pipe_pca,
            "grid": {
                "pca__n_components": [0.85, 0.95],
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [8, 12, None],
            },
            "cv": cv,
        }
    )
    pipe_lda = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis()),
            ("clf", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ]
    )
    grids.append(
        {
            "name": "rf_lda",
            "pipe": pipe_lda,
            "grid": {
                "lda__solver": ["svd", "eigen"],
                **base_rf,
            },
            "cv": cv,
        }
    )
    return grids


def score_val(pipe: Pipeline, X_val, y_val) -> dict:
    proba = pipe.predict_proba(X_val)[:, 1]
    pred = pipe.predict(X_val)
    return {
        "val_accuracy": float(accuracy_score(y_val, pred)),
        "val_precision": float(precision_score(y_val, pred, zero_division=0)),
        "val_recall": float(recall_score(y_val, pred, zero_division=0)),
        "val_f1": float(f1_score(y_val, pred, zero_division=0)),
        "val_roc_auc": float(roc_auc_score(y_val, proba)),
    }


def main():
    root = _root()
    mlflow.set_tracking_uri(f"file:{root / 'mlruns'}")
    mlflow.set_experiment("breast_cancer_eng_ml")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = ensure_or_create_splits(root)
    X_train = X_train[feature_names]
    X_val = X_val[feature_names]
    X_test = X_test[feature_names]
    best_val_f1 = -1.0
    best_run_id = None
    best_name = None
    timing_rows = []
    for spec in build_pipelines():
        with mlflow.start_run(run_name=spec["name"]):
            t0 = time.perf_counter()
            gs = GridSearchCV(
                spec["pipe"],
                spec["grid"],
                cv=spec["cv"],
                scoring="f1",
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X_train, y_train)
            train_s = time.perf_counter() - t0
            best: Pipeline = gs.best_estimator_
            cv_mean = gs.cv_results_["mean_test_score"][gs.best_index_]
            cv_std = gs.cv_results_["std_test_score"][gs.best_index_]
            val_scores = score_val(best, X_val, y_val)
            n_comp = None
            var_exp = None
            if "pca" in best.named_steps:
                pca: PCA = best.named_steps["pca"]
                n_comp = int(pca.n_components_) if hasattr(pca, "n_components_") else None
                var_exp = float(sum(pca.explained_variance_ratio_)) if n_comp else None
            mlflow.log_params(gs.best_params_)
            mlflow.log_param("approach", spec["name"])
            mlflow.log_metric("cv_f1_mean", float(cv_mean))
            mlflow.log_metric("cv_f1_std", float(cv_std))
            mlflow.log_metric("train_grid_seconds", float(train_s))
            for k, v in val_scores.items():
                mlflow.log_metric(k, v)
            if n_comp is not None:
                mlflow.log_metric("pca_n_components", float(n_comp))
            if var_exp is not None:
                mlflow.log_metric("pca_variance_retained", var_exp)
            mlflow.sklearn.log_model(best, artifact_path="model")
            run = mlflow.active_run()
            rid = run.info.run_id
            timing_rows.append(
                {
                    "approach": spec["name"],
                    "train_cv_seconds": train_s,
                    "val_f1": val_scores["val_f1"],
                }
            )
            if val_scores["val_f1"] > best_val_f1:
                best_val_f1 = val_scores["val_f1"]
                best_run_id = rid
                best_name = spec["name"]
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, "BreastCancerClassifier")
        reg = {
            "selected_run_id": best_run_id,
            "selected_approach": best_name,
            "selection_metric": "val_f1",
            "timing": timing_rows,
        }
        with open(root / "artifacts" / "registry_selection.json", "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)
    print(json.dumps({"best_run_id": best_run_id, "best_approach": best_name, "best_val_f1": best_val_f1}, indent=2))


if __name__ == "__main__":
    main()
