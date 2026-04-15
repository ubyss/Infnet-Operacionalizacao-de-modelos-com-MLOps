import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TARGET_COL = "diagnosis"
DATASET_SLUG = "yasserh/breast-cancer-dataset"
CSV_NAME = "breast-cancer.csv"

_SKLEARN_TO_KAGGLE = {
    "mean radius": "radius_mean",
    "mean texture": "texture_mean",
    "mean perimeter": "perimeter_mean",
    "mean area": "area_mean",
    "mean smoothness": "smoothness_mean",
    "mean compactness": "compactness_mean",
    "mean concavity": "concavity_mean",
    "mean concave points": "concave points_mean",
    "mean symmetry": "symmetry_mean",
    "mean fractal dimension": "fractal_dimension_mean",
    "radius error": "radius_se",
    "texture error": "texture_se",
    "perimeter error": "perimeter_se",
    "area error": "area_se",
    "smoothness error": "smoothness_se",
    "compactness error": "compactness_se",
    "concavity error": "concavity_se",
    "concave points error": "concave points_se",
    "symmetry error": "symmetry_se",
    "fractal dimension error": "fractal_dimension_se",
    "worst radius": "radius_worst",
    "worst texture": "texture_worst",
    "worst perimeter": "perimeter_worst",
    "worst area": "area_worst",
    "worst smoothness": "smoothness_worst",
    "worst compactness": "compactness_worst",
    "worst concavity": "concavity_worst",
    "worst concave points": "concave points_worst",
    "worst symmetry": "symmetry_worst",
    "worst fractal dimension": "fractal_dimension_worst",
}


def _from_sklearn_breast_cancer() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer()
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    X = X.rename(columns=_SKLEARN_TO_KAGGLE)
    y = 1 - bunch.target
    out = X.copy()
    out[TARGET_COL] = y
    return out


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return _project_root()


def read_saved_splits(root: Path | None = None):
    root = root or _project_root()
    split_dir = root / "artifacts" / "splits"
    if not (split_dir / "X_train.csv").is_file():
        return None
    X_train = pd.read_csv(split_dir / "X_train.csv")
    X_val = pd.read_csv(split_dir / "X_val.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")
    y_train = pd.read_csv(split_dir / "y_train.csv")[TARGET_COL]
    y_val = pd.read_csv(split_dir / "y_val.csv")[TARGET_COL]
    y_test = pd.read_csv(split_dir / "y_test.csv")[TARGET_COL]
    with open(split_dir / "feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def ensure_or_create_splits(root: Path | None = None):
    root = root or _project_root()
    got = read_saved_splits(root)
    if got is not None:
        return got
    df = load_dataframe()
    X, y = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_stratified(X, y)
    feature_names = list(X.columns)
    split_dir = root / "artifacts" / "splits"
    save_splits_artifacts(
        split_dir,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names,
    )
    prof = root / "artifacts" / "data_profile"
    prof.mkdir(parents=True, exist_ok=True)
    rep = {
        "quality": data_quality_report(X, y),
        "bias": bias_class_imbalance_summary(y),
    }
    with open(prof / "data_profile.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def data_drift_ks_report(
    ref: pd.DataFrame,
    new_df: pd.DataFrame,
    alpha: float = 0.01,
) -> dict:
    from scipy.stats import ks_2samp

    report = {"features": {}, "summary": {"n_features_checked": 0, "data_drift_flags": 0}}
    common = [
        c
        for c in ref.columns
        if c in new_df.columns and pd.api.types.is_numeric_dtype(ref[c])
    ]
    report["summary"]["n_features_checked"] = len(common)
    for col in common:
        a = ref[col].dropna().values
        b = new_df[col].dropna().values
        if len(a) < 10 or len(b) < 10:
            report["features"][col] = {"note": "amostra insuficiente"}
            continue
        stat, p = ks_2samp(a, b)
        drift = bool(p < alpha)
        if drift:
            report["summary"]["data_drift_flags"] += 1
        report["features"][col] = {
            "ks_statistic": float(stat),
            "p_value": float(p),
            "data_drift_p_lt_alpha": drift,
            "ref_mean": float(np.mean(a)),
            "new_mean": float(np.mean(b)),
        }
    return report


def load_dataframe(data_csv: str | None = None) -> pd.DataFrame:
    if data_csv and Path(data_csv).is_file():
        df = pd.read_csv(data_csv)
        return df
    env_path = os.environ.get("BREAST_CANCER_CSV")
    if env_path and Path(env_path).is_file():
        return pd.read_csv(env_path)
    local = _project_root() / "data" / CSV_NAME
    if local.is_file():
        return pd.read_csv(local)
    try:
        import kagglehub

        path = kagglehub.dataset_download(DATASET_SLUG)
        file_path = Path(path) / CSV_NAME
        return pd.read_csv(file_path)
    except Exception:
        return _from_sklearn_breast_cancer()


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map({"M": 1, "B": 0})
    if df[TARGET_COL].isna().any():
        raise ValueError("Valores de diagnosis inválidos.")
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def data_quality_report(X: pd.DataFrame, y: pd.Series) -> dict:
    n = len(X)
    missing = X.isna().sum().sum()
    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outlier_mask = (X < low) | (X > high)
    corr = X.corr(numeric_only=True)
    triu = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_vals = corr.where(triu).stack()
    high_corr_pairs = int((corr_vals.abs() > 0.9).sum())
    return {
        "n_samples": int(n),
        "n_features": int(X.shape[1]),
        "total_missing_cells": int(missing),
        "features_with_any_missing": int((X.isna().any()).sum()),
        "outlier_cells_iqr_rule": int(outlier_mask.sum().sum()),
        "high_correlation_pairs_abs_gt_0_9": high_corr_pairs,
        "class_distribution": y.value_counts(normalize=True).round(4).to_dict(),
        "class_counts": y.value_counts().to_dict(),
    }


def bias_class_imbalance_summary(y: pd.Series) -> dict:
    props = y.value_counts(normalize=True)
    minority_ratio = float(props.min())
    majority_ratio = float(props.max())
    return {
        "minority_class_ratio": minority_ratio,
        "majority_class_ratio": majority_ratio,
        "imbalance_ratio": majority_ratio / minority_ratio if minority_ratio > 0 else None,
        "stratify_recommended": minority_ratio < 0.2 or majority_ratio / minority_ratio > 3,
    }


def train_val_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size_of_trainval: float = 0.15 / 0.85,
) -> tuple:
    X_tv, X_test, y_tv, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv,
        y_tv,
        test_size=val_size_of_trainval,
        random_state=RANDOM_STATE,
        stratify=y_tv,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits_artifacts(
    out_dir: Path,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    feature_names: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_val.to_csv(out_dir / "X_val.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    pd.DataFrame({TARGET_COL: y_train}).to_csv(out_dir / "y_train.csv", index=False)
    pd.DataFrame({TARGET_COL: y_val}).to_csv(out_dir / "y_val.csv", index=False)
    pd.DataFrame({TARGET_COL: y_test}).to_csv(out_dir / "y_test.csv", index=False)
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)


def main():
    root = _project_root()
    df = load_dataframe()
    X, y = preprocess(df)
    q = data_quality_report(X, y)
    b = bias_class_imbalance_summary(y)
    rep = {"quality": q, "bias": b}
    art = root / "artifacts" / "data_profile"
    art.mkdir(parents=True, exist_ok=True)
    with open(art / "data_profile.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_stratified(X, y)
    save_splits_artifacts(
        root / "artifacts" / "splits",
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        list(X.columns),
    )
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
