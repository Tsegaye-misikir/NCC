# -*- coding: utf-8 -*-
"""Diagnostics that explain *why* a layer transfers well or badly.

The probing tables say which layer wins.  These two measurements say what is
in the layer that makes it win:

Language-identification probe
    Train a linear classifier to predict which language a sentence is in,
    from the same pooled vector the emotion probe sees.  High accuracy means
    the layer keeps language identity linearly available -- the classic
    signature of a "language-specific" layer.  A layer that is easy to
    identify the language from is a layer where a cross-lingual probe can
    latch onto language-specific directions and fail to transfer.

Linear CKA and centroid cosine
    Two views of how well two languages' spaces line up at a layer.  CKA
    asks whether the layer *organises* the two sets of sentences similarly;
    centroid cosine asks whether the two clouds sit in the same place.

    The distinction matters more than it looks.  CKA centres each side
    before comparing, so it is blind to a constant offset between two
    languages -- and a constant offset is exactly what breaks a linear probe
    carried from one language to another.  A layer can therefore show high
    CKA and still transfer badly.  Report both, and expect centroid cosine
    to be the better predictor of zero-shot performance.

None of these is a substitute for the transfer numbers; they are the
explanatory variables you regress those numbers against.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from layerprobe.features import FeatureStore, LayerFeatures


def _pooled_layer_matrix(store: FeatureStore, languages: Sequence[str], split: str, layer_id: int):
    """Stack one layer's features for several languages, with language tags."""

    blocks, tags = [], []
    for code, language in enumerate(languages):
        feats: LayerFeatures = store[language][split]
        matrix = feats.layer(layer_id)
        blocks.append(matrix)
        tags.append(np.full(matrix.shape[0], code, dtype=np.int64))
    return np.concatenate(blocks, axis=0), np.concatenate(tags, axis=0)


def language_identification(
    store: FeatureStore,
    languages: Sequence[str],
    layer_ids: Sequence[int],
    train_split: str = "train",
    test_split: str = "test",
    C: float = 1.0,
    max_iter: int = 1000,
    max_per_language: Optional[int] = 400,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Per-layer accuracy of a linear language-identification probe."""

    languages = list(languages)
    if len(languages) < 2:
        return []

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    for layer_id in layer_ids:
        X_train, y_train = _pooled_layer_matrix(store, languages, train_split, layer_id)
        X_test, y_test = _pooled_layer_matrix(store, languages, test_split, layer_id)
        if max_per_language is not None:
            cap = max_per_language * len(languages)
            if X_train.shape[0] > cap:
                idx = rng.choice(X_train.shape[0], size=cap, replace=False)
                X_train, y_train = X_train[idx], y_train[idx]

        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")
        clf.fit(scaler.transform(X_train), y_train)
        accuracy = float((clf.predict(scaler.transform(X_test)) == y_test).mean())
        chance = float(max(np.bincount(y_test, minlength=len(languages))) / len(y_test))
        rows.append(
            {
                "layer": int(layer_id),
                "language_id_accuracy": accuracy,
                "chance": chance,
                # Above-chance identifiability, normalised so 1.0 means the
                # language is perfectly recoverable and 0.0 means no better
                # than guessing the most frequent language.  Small negative
                # values are ordinary sampling noise around chance, and are
                # left unclamped so that noise level stays visible.
                "language_specificity": float((accuracy - chance) / max(1e-9, 1.0 - chance)),
            }
        )
    return rows


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two sets of representations.

    ``X`` and ``Y`` must have the same number of rows (paired or merely
    same-sized samples); their dimensionalities may differ.  Returns a value
    in [0, 1], 1 meaning the two spaces induce the same similarity structure.
    """

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"CKA needs equal sample counts, got {X.shape[0]} and {Y.shape[0]}")
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F) -- the feature-space form,
    # far cheaper than building n-by-n Gram matrices.
    cross = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    denom = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    if denom < 1e-12:
        return 0.0
    return float(cross / denom)


def cross_lingual_alignment(
    store: FeatureStore,
    source_languages: Sequence[str],
    target_languages: Sequence[str],
    layer_ids: Sequence[int],
    split: str = "test",
    max_samples: int = 500,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Per-layer CKA and centroid distance for every source/target pair."""

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    for source in source_languages:
        for target in target_languages:
            if source == target:
                continue
            for layer_id in layer_ids:
                A = store[source][split].layer(layer_id)
                B = store[target][split].layer(layer_id)
                n = min(len(A), len(B), max_samples)
                if n < 2:
                    continue
                # CKA compares similarity structure, not paired items, so an
                # independent random subsample of each side is valid.
                A_s = A[rng.choice(len(A), size=n, replace=False)]
                B_s = B[rng.choice(len(B), size=n, replace=False)]
                centroid_a = A.mean(axis=0)
                centroid_b = B.mean(axis=0)
                cosine = float(
                    centroid_a
                    @ centroid_b
                    / max(1e-9, np.linalg.norm(centroid_a) * np.linalg.norm(centroid_b))
                )
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "layer": int(layer_id),
                        "cka": linear_cka(A_s, B_s),
                        "centroid_cosine": cosine,
                    }
                )
    return rows


def correlate_alignment_with_transfer(
    alignment_rows: Sequence[Dict[str, float]],
    language_rows: Sequence[Dict[str, float]],
    transfer_rows: Sequence[Dict[str, object]],
    metric: str = "macro_f1_mean",
) -> Dict[str, float]:
    """Correlate the diagnostics against zero-shot performance, per layer.

    Answers the question the diagnostics exist for: does a layer being
    language-neutral actually predict that it transfers?  Returns Pearson
    correlations over the per-layer means, plus the sample size they rest on
    (with 13 layers these are indicative, not confirmatory).
    """

    single = {
        int(r["layers"][0]): float(r[metric])
        for r in transfer_rows
        if r.get("kind") == "single" and metric in r and len(r.get("layers", [])) == 1
    }
    if len(single) < 3:
        return {"n_layers": len(single)}

    layers = sorted(single)
    transfer = np.array([single[l] for l in layers])

    out: Dict[str, float] = {"n_layers": len(layers)}

    def _correlate(name: str, by_layer: Dict[int, float]) -> None:
        if not by_layer:
            return
        values = np.array([by_layer.get(l, np.nan) for l in layers])
        if not np.isfinite(values).all():
            return
        if np.std(values) < 1e-12 or np.std(transfer) < 1e-12:
            return
        out[f"pearson_{name}_vs_transfer"] = float(np.corrcoef(values, transfer)[0, 1])

    for metric in ("cka", "centroid_cosine"):
        grouped: Dict[int, List[float]] = {}
        for row in alignment_rows:
            if metric in row:
                grouped.setdefault(int(row["layer"]), []).append(float(row[metric]))
        _correlate(metric, {layer: float(np.mean(vals)) for layer, vals in grouped.items()})

    _correlate(
        "language_specificity",
        {int(r["layer"]): float(r["language_specificity"]) for r in language_rows},
    )
    return out
