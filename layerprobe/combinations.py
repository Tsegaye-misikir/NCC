# -*- coding: utf-8 -*-
"""The layer selections and combinations that enter the comparison.

Four families, in increasing order of how much they are allowed to adapt to
the task:

``single``
    One layer on its own -- including ``last``, the default choice that this
    study exists to question.
``average``
    An unweighted mean over a set of layers (all of them, or a window).
    Costs nothing and needs no tuning.
``concat``
    Layers stacked side by side.  Strictly more expressive than an average
    but multiplies the probe's parameter count, which matters when the
    target language has a few hundred training sentences.
``scalarmix``
    Softmax-weighted mean whose weights are learned with the classifier
    (ELMo-style).  The weights are themselves a result: they say which
    layers the task actually wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from layerprobe.config import CombinationConfig
from layerprobe.features import LayerFeatures


@dataclass(frozen=True)
class LayerCombination:
    """A named recipe for turning layer-wise features into a design matrix."""

    name: str
    kind: str  # single | average | concat | scalarmix
    layers: tuple = field(default_factory=tuple)

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.name


def _window_name(layers: Sequence[int]) -> str:
    return f"avg{layers[0]}-{layers[-1]}"


def build_combinations(cfg: CombinationConfig, layer_ids: Sequence[int]) -> List[LayerCombination]:
    """Expand a :class:`CombinationConfig` into concrete combinations.

    Order matters only for readability of the output tables: individual
    layers first, then averages, then concatenations, then the learned mix.
    """

    layer_ids = list(layer_ids)
    if not layer_ids:
        raise ValueError("no layers to combine")
    combos: List[LayerCombination] = []
    seen: set = set()

    def add(combo: LayerCombination) -> None:
        key = (combo.kind, combo.layers)
        if key in seen:
            return
        seen.add(key)
        combos.append(combo)

    # The final layer is the baseline every other row is compared against, so
    # it is always present -- and it is added first so that it keeps the
    # readable name rather than being deduplicated behind "layer12".
    add(LayerCombination("last", "single", (layer_ids[-1],)))
    if cfg.individual_layers:
        for layer in layer_ids:
            add(LayerCombination(f"layer{layer}", "single", (layer,)))

    if cfg.average_all:
        add(LayerCombination("avg_all", "average", tuple(layer_ids)))

    for size in cfg.window_sizes:
        if size <= 1 or size > len(layer_ids):
            continue
        for start in range(0, len(layer_ids) - size + 1):
            window = layer_ids[start : start + size]
            add(LayerCombination(_window_name(window), "average", tuple(window)))

    named = dict(cfg.named_windows)
    if not named:
        # Sensible defaults expressed relative to the encoder's depth, so the
        # same config works for a 12-layer XLM-R base, a 24-layer large, and
        # a 28-layer Qwen without editing anything.
        n = len(layer_ids)
        named = {
            "avg_bottom": layer_ids[: max(1, n // 3)],
            "avg_middle": layer_ids[max(1, n // 3) : max(2, 2 * n // 3)],
            "avg_top": layer_ids[max(2, 2 * n // 3) :],
        }
    available = set(layer_ids)

    def _validated(name: str, layers: Sequence[int], source: str) -> List[int]:
        """Reject layer indices this encoder does not have.

        Hard-coded windows are the main trap when swapping models: a group
        written for a 13-layer XLM-R silently refers to nothing on a
        28-layer Qwen, so fail loudly at build time rather than deep inside
        the probe loop.
        """

        layers = [int(l) for l in layers]
        missing = sorted(set(layers) - available)
        if missing:
            raise ValueError(
                f"{source} {name!r} refers to layer(s) {missing}, but this encoder "
                f"only has {min(available)}..{max(available)}. Leave "
                "combinations.named_windows empty to get depth-relative thirds."
            )
        return layers

    for name, layers in named.items():
        layers = _validated(name, layers, "combinations.named_windows")
        if layers:
            add(LayerCombination(name, "average", tuple(layers)))

    for name, layers in cfg.concat_groups.items():
        layers = _validated(name, layers, "combinations.concat_groups")
        if layers:
            add(LayerCombination(name, "concat", tuple(layers)))

    if cfg.scalar_mix:
        add(LayerCombination("scalar_mix", "scalarmix", tuple(layer_ids)))

    return combos


def materialize(features: LayerFeatures, combo: LayerCombination) -> np.ndarray:
    """Build the ``(n, d)`` design matrix a probe sees for this combination.

    Scalar-mix combinations have no fixed matrix -- their weights are learned
    -- so they are handled by :mod:`layerprobe.probes` instead and raise here.
    """

    if combo.kind == "scalarmix":
        raise ValueError("scalar-mix combinations are materialised inside the probe")
    stacked = np.stack([features.layer(l) for l in combo.layers], axis=0)
    if combo.kind == "single":
        return stacked[0]
    if combo.kind == "average":
        return stacked.mean(axis=0)
    if combo.kind == "concat":
        return np.concatenate(list(stacked), axis=1)
    raise ValueError(f"unknown combination kind {combo.kind!r}")


def layer_tensor(features: LayerFeatures, combo: LayerCombination) -> np.ndarray:
    """The ``(n_layers, n, d)`` tensor a scalar-mix probe mixes over."""

    return np.stack([features.layer(l) for l in combo.layers], axis=0)


def describe_combinations(combos: Sequence[LayerCombination]) -> List[Dict[str, object]]:
    return [
        {"name": c.name, "kind": c.kind, "layers": list(c.layers), "n_layers": len(c.layers)}
        for c in combos
    ]
