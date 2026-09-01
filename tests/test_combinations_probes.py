# -*- coding: utf-8 -*-
"""Layer combinations and the probes trained on top of them."""

from __future__ import annotations

import numpy as np
import pytest

from layerprobe.combinations import (
    build_combinations,
    describe_combinations,
    layer_tensor,
    materialize,
)
from layerprobe.config import CombinationConfig, ProbeConfig, ScalarMixConfig
from layerprobe.features import LayerFeatures
from layerprobe.probes import LogisticProbe, MLPProbe, ScalarMixProbe, build_probe

LAYERS = [0, 1, 2, 3]


def _features(n: int = 40, d: int = 6, seed: int = 0) -> LayerFeatures:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(LAYERS), n, d)).astype(np.float32)
    return LayerFeatures(values, list(LAYERS), "eng", "train")


def _separable(n: int = 120, d: int = 8, n_layers: int = 4, signal_layer: int = 2, seed: int = 0):
    """Features where only one layer carries the label, plus its labels.

    The whole study rests on a probe being able to tell layers apart, so the
    tests check that on data where the answer is known by construction.
    """

    rng = np.random.default_rng(seed)
    y = (rng.random((n, 2)) < 0.5).astype(np.float32)
    values = rng.normal(scale=1.0, size=(n_layers, n, d)).astype(np.float32)
    values[signal_layer, :, :2] += 6.0 * (y - 0.5)
    return LayerFeatures(values, list(range(n_layers)), "eng", "train"), y


def test_build_combinations_covers_every_family():
    cfg = CombinationConfig(individual_layers=True, average_all=True, window_sizes=[2], scalar_mix=True)
    combos = build_combinations(cfg, LAYERS)
    kinds = {c.kind for c in combos}
    assert kinds == {"single", "average", "scalarmix"}
    names = [c.name for c in combos]
    assert "last" in names and "avg_all" in names and "scalar_mix" in names
    # The final layer appears once, under the readable baseline name.
    assert "layer3" not in names
    assert sum(1 for c in combos if c.kind == "single" and c.layers == (3,)) == 1


def test_last_is_present_even_without_individual_layers():
    combos = build_combinations(CombinationConfig(individual_layers=False, scalar_mix=False), LAYERS)
    assert [c.name for c in combos if c.kind == "single"] == ["last"]


def test_named_windows_default_to_thirds():
    combos = build_combinations(
        CombinationConfig(individual_layers=False, average_all=False, window_sizes=[], scalar_mix=False),
        list(range(13)),
    )
    names = {c.name for c in combos}
    assert {"avg_bottom", "avg_middle", "avg_top"} <= names


def test_explicit_named_windows_and_concat_groups():
    cfg = CombinationConfig(
        individual_layers=False,
        average_all=False,
        window_sizes=[],
        named_windows={"mid": [1, 2]},
        concat_groups={"ends": [0, 3]},
        scalar_mix=False,
    )
    combos = {c.name: c for c in build_combinations(cfg, LAYERS)}
    assert combos["mid"].kind == "average" and combos["mid"].layers == (1, 2)
    assert combos["ends"].kind == "concat"


def test_windows_larger_than_the_stack_are_skipped():
    cfg = CombinationConfig(
        individual_layers=False,
        average_all=False,
        window_sizes=[99],
        named_windows={"mid": [1, 2]},
        scalar_mix=False,
    )
    assert [c.name for c in build_combinations(cfg, LAYERS)] == ["last", "mid"]


def test_materialize_shapes_and_semantics():
    features = _features()
    single = next(c for c in build_combinations(CombinationConfig(), LAYERS) if c.name == "last")
    np.testing.assert_allclose(materialize(features, single), features.layer(3))

    avg = next(c for c in build_combinations(CombinationConfig(), LAYERS) if c.name == "avg_all")
    np.testing.assert_allclose(
        materialize(features, avg), features.values.mean(axis=0), rtol=1e-5
    )

    cfg = CombinationConfig(individual_layers=False, concat_groups={"c": [0, 1]}, scalar_mix=False)
    concat = next(c for c in build_combinations(cfg, LAYERS) if c.name == "c")
    assert materialize(features, concat).shape == (40, 12)


def test_scalar_mix_cannot_be_materialized_as_a_matrix():
    features = _features()
    mix = next(c for c in build_combinations(CombinationConfig(), LAYERS) if c.kind == "scalarmix")
    with pytest.raises(ValueError, match="materialised inside the probe"):
        materialize(features, mix)
    assert layer_tensor(features, mix).shape == (4, 40, 6)


def test_describe_combinations():
    rows = describe_combinations(build_combinations(CombinationConfig(), LAYERS))
    assert all({"name", "kind", "layers", "n_layers"} <= set(r) for r in rows)


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------


def test_logistic_probe_finds_the_informative_layer():
    features, y = _separable(signal_layer=2)
    probe = LogisticProbe(ProbeConfig(C=[1.0], max_iter=500), "multilabel", ["a", "b"])

    scores = {}
    for layer in range(4):
        X = features.layer(layer)
        outcome = probe.run(X[:80], y[:80], X[80:100], y[80:100], X[100:], y[100:])
        scores[layer] = outcome.test["macro_f1"]

    assert max(scores, key=scores.get) == 2
    assert scores[2] > 0.85
    assert scores[0] < 0.7


def test_logistic_probe_handles_a_constant_label_column():
    """A rare emotion can be absent from a low-resource training split."""

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = np.zeros((60, 2), dtype=np.float32)
    y[:, 0] = (rng.random(60) < 0.5).astype(np.float32)  # second column never fires
    probe = LogisticProbe(ProbeConfig(C=[1.0], max_iter=200), "multilabel", ["a", "b"])
    outcome = probe.run(X[:40], y[:40], X[40:50], y[40:50], X[50:], y[50:])
    assert np.isfinite(outcome.test["macro_f1"])


def test_logistic_probe_single_label_with_a_missing_class():
    """predict_proba must still be widened to the full emotion inventory."""

    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    y = rng.integers(0, 2, size=60)  # classes 2 and 3 never appear
    probe = LogisticProbe(ProbeConfig(C=[1.0], max_iter=200), "singlelabel", ["a", "b", "c", "d"])
    outcome = probe.run(X[:40], y[:40], X[40:50], y[40:50], X[50:], y[50:])
    assert "f1_d" in outcome.test
    assert np.isfinite(outcome.test["macro_f1"])


def test_probe_sweep_reports_the_chosen_hyperparameters():
    features, y = _separable()
    X = features.layer(2)
    probe = LogisticProbe(ProbeConfig(C=[0.01, 10.0], max_iter=300), "multilabel", ["a", "b"])
    outcome = probe.run(X[:80], y[:80], X[80:100], y[80:100], X[100:], y[100:])
    assert outcome.chosen["C"] in (0.01, 10.0)
    assert 0.0 < outcome.chosen["threshold"] < 1.0


def test_scalar_mix_learns_to_favour_the_informative_layer():
    features, y = _separable(n=200, signal_layer=1, seed=3)
    tensor = features.values
    probe = ScalarMixProbe(
        ProbeConfig(),
        ScalarMixConfig(epochs=120, learning_rate=0.05),
        "multilabel",
        ["a", "b"],
        seed=0,
    )
    outcome = probe.run(
        tensor[:, :140], y[:140], tensor[:, 140:170], y[140:170], tensor[:, 170:], y[170:]
    )
    weights = outcome.layer_weights
    assert weights is not None and len(weights) == 4
    assert np.isclose(sum(weights), 1.0, atol=1e-5)
    assert int(np.argmax(weights)) == 1
    assert outcome.test["macro_f1"] > 0.8


def test_mlp_probe_runs_and_scores():
    features, y = _separable(signal_layer=0)
    X = features.layer(0)
    probe = MLPProbe(
        ProbeConfig(epochs=200, hidden_size=16, learning_rate=0.01), "multilabel", ["a", "b"], seed=0
    )
    outcome = probe.run(X[:80], y[:80], X[80:100], y[80:100], X[100:], y[100:])
    assert outcome.test["macro_f1"] > 0.7


def test_build_probe_dispatch():
    assert isinstance(build_probe(ProbeConfig(kind="logreg"), "multilabel", ["a"]), LogisticProbe)
    assert isinstance(build_probe(ProbeConfig(kind="mlp"), "multilabel", ["a"]), MLPProbe)
    with pytest.raises(ValueError, match="unknown probe.kind"):
        build_probe(ProbeConfig(kind="svm"), "multilabel", ["a"])
