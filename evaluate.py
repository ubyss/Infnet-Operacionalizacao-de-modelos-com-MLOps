from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_prep import TARGET_COL
from model_io import load_model_bundle


def _root() -> Path:
    return Path(__file__).resolve().parent


def load_model_and_test():
    root = _root()
    model, feature_names, run_id = load_model_bundle(root)
    split_dir = root / "artifacts" / "splits"
    X_test = pd.read_csv(split_dir / "X_test.csv")
    y_test = pd.read_csv(split_dir / "y_test.csv")[TARGET_COL]
    X_test = X_test[feature_names]
    return model, X_test, y_test, run_id


def main():
    model, X_test, y_test, run_id = load_model_and_test()
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    metrics = {
        "run_id": run_id,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
    }
    cm = confusion_matrix(y_test, pred).tolist()
    report = classification_report(
        y_test,
        pred,
        target_names=["Benigno", "Maligno"],
        digits=4,
    )
    out = _root() / "artifacts" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({**metrics, "confusion_matrix": cm}, f, indent=2)
    with open(out / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(json.dumps(metrics, indent=2))
    print(report)


if __name__ == "__main__":
    main()
