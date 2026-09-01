# -*- coding: utf-8 -*-
"""Scoring functions and the representation diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from layerprobe.analysis import (
    correlate_alignment_with_transfer,
    cross_lingual_alignment,
    language_identification,
    linear_cka,
)
from layerprobe.features import LayerFeatures
from layerprobe.metrics import (
    majority_baseline,
    multilabel_scores,
    paired_bootstrap,
    score,
    singlelabel_scores,
    summarise,
    transfer_gap,
    tune_threshold,
)


def test_perfect_multilabel_prediction_scores_one():
    y = np.array([[1, 0], [0, 1], [1, 1]])
    scores = multilabel_scores(y, y.astype(float), 0.5, ["a", "b"])
    assert scores["macro_f1"] == pytest.approx(1.0)
    assert scores["subset_accuracy"] == pytest.approx(1.0)
    assert scores["f1_a"] == pytest.approx(1.0)


def test_macro_f1_punishes_ignoring_a_rare_emotion():
    """Why macro-F1 is the headline: micro-F1 would hide this."""

    y = np.zeros((100, 2), dtype=int)
    y[:90, 0] = 1
    y[:5, 1] = 1
    always_common = np.zeros((100, 2))
    always_common[:, 0] = 1.0
    scores = multilabel_scores(y, always_common, 0.5, ["common", "rare"])
    assert scores["f1_rare"] == 0.0
    assert scores["macro_f1"] < scores["micro_f1"]


def test_singlelabel_scores_and_missing_class():
    y_true = np.array([0, 1, 2])
    y_prob = np.eye(3)
    scores = singlelabel_scores(y_true, y_prob, ["a", "b", "c"])
    assert scores["accuracy"] == pytest.approx(1.0)
    assert scores["macro_f1"] == pytest.approx(1.0)


def test_score_dispatches_on_task():
    y = np.array([[1, 0]])
    assert "subset_accuracy" in score(y, y.astype(float), "multilabel")
    assert "accuracy" in score(np.array([0]), np.array([[0.9, 0.1]]), "singlelabel")


def test_tune_threshold_picks_a_better_operating_point():
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    prob = np.array([[0.4, 0.1], [0.45, 0.05], [0.1, 0.4], [0.05, 0.44]])
    best = tune_threshold(y, prob)
    assert best < 0.5
    assert (
        multilabel_scores(y, prob, best)["macro_f1"] > multilabel_scores(y, prob, 0.5)["macro_f1"]
    )


def test_majority_baseline_is_a_real_floor():
    y_train = np.zeros((100, 2), dtype=np.float32)
    y_train[:80, 0] = 1
    y_test = np.zeros((20, 2), dtype=np.float32)
    y_test[:16, 0] = 1
    scores = majority_baseline(y_train, y_test, "multilabel", ["a", "b"])
    # It always predicts the frequent emotion and never the rare one.
    assert scores["f1_a"] > 0.8
    assert scores["f1_b"] == 0.0


def test_majority_baseline_single_label():
    y_train = np.array([0, 0, 0, 1])
    y_test = np.array([0, 0, 1])
    scores = majority_baseline(y_train, y_test, "singlelabel", ["a", "b"])
    assert scores["accuracy"] == pytest.approx(2 / 3)


def test_transfer_gap_sign():
    assert transfer_gap(0.4, 0.6) == pytest.approx(0.2)  # zero-shot loses ground
    assert transfer_gap(0.7, 0.6) == pytest.approx(-0.1)  # ... and can also win


def test_summarise_mean_and_std():
    out = summarise([{"macro_f1": 0.4}, {"macro_f1": 0.6}], ["macro_f1"])
    assert out["macro_f1_mean"] == pytest.approx(0.5)
    assert out["macro_f1_std"] == pytest.approx(np.std([0.4, 0.6], ddof=1))
    assert out["n_seeds"] == 2


def test_summarise_single_seed_has_zero_std():
    assert summarise([{"macro_f1": 0.4}], ["macro_f1"])["macro_f1_std"] == 0.0


def test_paired_bootstrap_separates_signal_from_noise():
    identical = [0.5, 0.51, 0.49, 0.5]
    assert paired_bootstrap(identical, identical) > 0.5
    clearly_better = [0.8, 0.82, 0.79, 0.81]
    assert paired_bootstrap(clearly_better, identical) < 0.2


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same number of scores"):
        paired_bootstrap([0.1, 0.2], [0.1])


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def test_cka_is_one_for_identical_and_rotated_spaces():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)
    # CKA is invariant to orthogonal transformation -- that is the property
    # that makes it usable for comparing two languages' representations.
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert linear_cka(X, X @ q) == pytest.approx(1.0, abs=1e-6)
    # ... and to isotropic scaling.
    assert linear_cka(X, 3.0 * X) == pytest.approx(1.0, abs=1e-6)


def test_cka_is_low_for_unrelated_spaces():
    rng = np.random.default_rng(1)
    assert linear_cka(rng.normal(size=(200, 8)), rng.normal(size=(200, 8))) < 0.3


def test_cka_rejects_mismatched_sample_counts():
    with pytest.raises(ValueError, match="equal sample counts"):
        linear_cka(np.zeros((4, 2)), np.zeros((5, 2)))


def _store_with_language_signal(n_layers: int = 3, n: int = 80, d: int = 6):
    """Two languages whose separability shrinks as depth increases."""

    rng = np.random.default_rng(0)
    store = {}
    for code, language in enumerate(["eng", "amh"]):
        store[language] = {}
        for split in ("train", "test"):
            values = rng.normal(size=(n_layers, n, d)).astype(np.float32)
            for layer in range(n_layers):
                offset = 8.0 * (1.0 - layer / max(1, n_layers - 1))
                values[layer, :, 0] += offset * (1 if code == 0 else -1)
            store[language][split] = LayerFeatures(values, list(range(n_layers)), language, split)
    return store


def test_language_probe_tracks_the_planted_language_signal():
    store = _store_with_language_signal()
    rows = language_identification(store, ["eng", "amh"], [0, 1, 2], max_per_language=None)
    assert [r["layer"] for r in rows] == [0, 1, 2]
    assert rows[0]["language_id_accuracy"] > 0.95  # bottom layer: fully separable
    assert rows[-1]["language_id_accuracy"] < rows[0]["language_id_accuracy"]
    assert rows[0]["language_specificity"] > 0.9
    # A layer with no language signal sits at chance, so its specificity is
    # ~0 and may dip slightly below it; it is deliberately not clamped.
    assert all(-0.2 <= r["language_specificity"] <= 1.0 for r in rows)


def test_language_probe_needs_two_languages():
    store = _store_with_language_signal()
    assert language_identification(store, ["eng"], [0]) == []


def test_cross_lingual_alignment_emits_a_row_per_pair_and_layer():
    store = _store_with_language_signal()
    rows = cross_lingual_alignment(store, ["eng"], ["amh"], [0, 1, 2], max_samples=40)
    assert len(rows) == 3
    assert all(0.0 <= r["cka"] <= 1.0 for r in rows)
    assert all(-1.0 <= r["centroid_cosine"] <= 1.0 for r in rows)
    # Bottom layer holds the planted language offset, so the two languages'
    # centroids point in opposite directions there.
    assert rows[0]["centroid_cosine"] < rows[-1]["centroid_cosine"]


def test_alignment_skips_a_language_paired_with_itself():
    store = _store_with_language_signal()
    assert cross_lingual_alignment(store, ["eng"], ["eng"], [0]) == []


def test_correlation_needs_enough_layers():
    assert correlate_alignment_with_transfer([], [], [])["n_layers"] == 0


def test_correlation_recovers_a_planted_relationship():
    transfer = [
        {"kind": "single", "layers": [l], "macro_f1_mean": 0.1 * l, "experiment": "zeroshot"}
        for l in range(5)
    ]
    alignment = [{"layer": l, "cka": 0.1 * l, "centroid_cosine": 0.0} for l in range(5)]
    language = [{"layer": l, "language_specificity": 1.0 - 0.1 * l} for l in range(5)]
    out = correlate_alignment_with_transfer(alignment, language, transfer)
    assert out["pearson_cka_vs_transfer"] == pytest.approx(1.0, abs=1e-6)
    assert out["pearson_language_specificity_vs_transfer"] == pytest.approx(-1.0, abs=1e-6)


def test_correlation_covers_centroid_cosine_separately_from_cka():
    """CKA and centroid cosine can disagree, so both must be reported."""

    transfer = [
        {"kind": "single", "layers": [l], "macro_f1_mean": 0.1 * l, "experiment": "zeroshot"}
        for l in range(5)
    ]
    # CKA flat-but-noisy while the centroids line up with transfer: the
    # situation the offline smoke run actually produces.
    alignment = [
        {"layer": l, "cka": 0.5 - 0.01 * l, "centroid_cosine": 0.2 * l} for l in range(5)
    ]
    out = correlate_alignment_with_transfer(alignment, [], transfer)
    assert out["pearson_centroid_cosine_vs_transfer"] == pytest.approx(1.0, abs=1e-6)
    assert out["pearson_cka_vs_transfer"] == pytest.approx(-1.0, abs=1e-6)


def test_correlation_skips_a_constant_diagnostic():
    transfer = [
        {"kind": "single", "layers": [l], "macro_f1_mean": 0.1 * l, "experiment": "zeroshot"}
        for l in range(5)
    ]
    alignment = [{"layer": l, "cka": 0.5, "centroid_cosine": 0.5} for l in range(5)]
    out = correlate_alignment_with_transfer(alignment, [], transfer)
    assert "pearson_cka_vs_transfer" not in out
    assert out["n_layers"] == 5
