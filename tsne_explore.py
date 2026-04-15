from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from data_prep import TARGET_COL, ensure_or_create_splits, project_root, read_saved_splits


def _root() -> Path:
    return project_root()


def main():
    root = _root()
    pack = read_saved_splits(root)
    if pack is None:
        pack = ensure_or_create_splits(root)
    X_train, _, _, y_train, _, _, feature_names = pack
    X_train = X_train[feature_names]
    Xs = StandardScaler().fit_transform(X_train)
    n = len(X_train)
    perp = min(30, max(5, (n - 1) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(Xs)
    out_dir = root / "artifacts" / "exploration"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(
        {
            "tsne_1": emb[:, 0],
            "tsne_2": emb[:, 1],
            TARGET_COL: y_train.values,
        }
    )
    df_out.to_csv(out_dir / "tsne_train_2d.csv", index=False)
    meta = {
        "kl_divergence": float(tsne.kl_divergence_),
        "n_iter_": int(tsne.n_iter_),
        "perplexity": float(perp),
        "n_samples": int(n),
    }
    with open(out_dir / "tsne_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
