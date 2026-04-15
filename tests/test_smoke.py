from __future__ import annotations

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from breast_cancer_mlops.train import build_pipelines


def test_build_pipelines_three_approaches():
    assert len(build_pipelines()) == 3


def test_tsne_fit_small_matrix():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((45, 8))
    z = StandardScaler().fit_transform(x)
    y = TSNE(n_components=2, random_state=0, perplexity=10, init="random").fit_transform(z)
    assert y.shape == (45, 2)
