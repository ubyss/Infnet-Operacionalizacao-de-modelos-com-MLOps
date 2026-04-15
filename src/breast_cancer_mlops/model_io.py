from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn

from .paths import repo_root


def project_root() -> Path:
    return repo_root()


def load_model_bundle(root: Path | None = None):
    root = root or repo_root()
    mlflow.set_tracking_uri(f"file:{root / 'mlruns'}")
    sel = root / "artifacts" / "registry_selection.json"
    if not sel.is_file():
        raise FileNotFoundError("Execute o treino (train ou notebook) antes de carregar o modelo.")
    with open(sel, encoding="utf-8") as f:
        meta = json.load(f)
    run_id = meta["selected_run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    split_dir = root / "artifacts" / "splits"
    with open(split_dir / "feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)
    return model, feature_names, run_id
