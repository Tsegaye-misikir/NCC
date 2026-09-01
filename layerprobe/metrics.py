# -*- coding: utf-8 -*-
"""Scoring for multi-label and single-label emotion classification.

Macro-F1 is the headline number throughout: emotion corpora for
low-resource languages are small and heavily imbalanced, so accuracy and
micro-F1 both reward a probe for ignoring the rare emotions entirely.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def _as_int(array: np.ndarray) -> np.ndarray:
    return np.asarray(array).astype(int)


def multilabel_scores(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, emotions: Optional[Sequence[str]] = None
) -> Dict[str, float]:
    """Macro/micro F1 and per-emotion F1 at a fixed decision threshold."""

    y_true = _as_int(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    scores = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float((y_pred == y_true).all(axis=1).mean()),
        "threshold": float(threshold),
    }
    per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    names = list(emotions) if emotions is not None else [f"label_{i}" for i in range(len(per_label))]
    for name, value in zip(names, per_label):
        scores[f"f1_{name}"] = float(value)
    return scores


def singlelabel_scores(
    y_true: np.ndarray, y_prob: np.ndarray, emotions: Optional[Sequence[str]] = None
) -> Dict[str, float]:
    """Macro/micro F1 and per-class F1 for a single-label probe."""

    y_true = _as_int(y_true)
    y_pred = np.asarray(y_prob).argmax(axis=1)
    n_classes = np.asarray(y_prob).shape[1]
    labels = list(range(n_classes))
    scores = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        "accuracy": float((y_pred == y_true).mean()),
    }
    per_label = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    names = list(emotions) if emotions is not None else [f"class_{i}" for i in labels]
    for name, value in zip(names, per_label):
        scores[f"f1_{name}"] = float(value)
    return scores


def score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    task: str,
    threshold: float = 0.5,
    emotions: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    if task == "multilabel":
        return multilabel_scores(y_true, y_prob, threshold, emotions)
    return singlelabel_scores(y_true, y_prob, emotions)


def tune_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, grid: Optional[Sequence[float]] = None
) -> float:
    """Pick the multi-label threshold that maximises dev macro-F1.

    A single global threshold is tuned rather than one per emotion: with a
    few hundred dev examples per language, per-emotion thresholds overfit.
    """

    grid = list(grid) if grid is not None else [round(0.05 * i, 2) for i in range(1, 20)]
    y_true = _as_int(y_true)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (np.asarray(y_prob) >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray, task: str, emotions=None) -> Dict[str, float]:
    """Score of the trivial always-predict-the-majority classifier.

    Reported alongside every experiment: a cross-lingual probe that fails to
    clear this line has not transferred anything at all.
    """

    if task == "multilabel":
        positive = (np.asarray(y_train).mean(axis=0) >= 0.5).astype(float)
        y_prob = np.tile(positive, (len(y_test), 1))
        return multilabel_scores(y_test, y_prob, 0.5, emotions)
    n_classes = int(max(np.max(y_train), np.max(y_test)) + 1)
    majority = int(np.bincount(_as_int(y_train), minlength=n_classes).argmax())
    y_prob = np.zeros((len(y_test), n_classes))
    y_prob[:, majority] = 1.0
    return singlelabel_scores(y_test, y_prob, emotions)


def transfer_gap(zero_shot: float, in_language: float) -> float:
    """How much a zero-shot probe loses against a probe trained in-language.

    Negative values mean the cross-lingual probe is *better* than the
    language's own (which happens when the target's training set is tiny);
    large positive values are the negative-transfer regime this study is
    about.
    """

    return float(in_language - zero_shot)


def summarise(runs: Sequence[Dict[str, float]], keys: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Mean and standard deviation of a metric dict across seeds."""

    if not runs:
        return {}
    keys = list(keys) if keys is not None else sorted({k for r in runs for k in r})
    out: Dict[str, float] = {"n_seeds": len(runs)}
    for key in keys:
        values = [float(r[key]) for r in runs if key in r]
        if not values:
            continue
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out


def paired_bootstrap(
    scores_a: Sequence[float], scores_b: Sequence[float], n_resamples: int = 10000, seed: int = 0
) -> float:
    """Two-sided bootstrap p-value that two layer choices differ.

    Fed the per-seed scores of two configurations, it answers the question a
    layer-comparison table always raises: is that 1.2-point gap real, or
    seed noise?
    """

    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs the same number of scores on each side")
    if a.size == 0:
        return float("nan")
    observed = float(np.mean(a - b))
    diffs = a - b
    centred = diffs - diffs.mean()
    rng = np.random.default_rng(seed)
    draws = rng.choice(centred, size=(n_resamples, diffs.size), replace=True).mean(axis=1)
    return float((np.abs(draws) >= abs(observed)).mean())
