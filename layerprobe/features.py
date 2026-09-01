# -*- coding: utf-8 -*-
"""On-disk cache of layer-wise features.

Encoding a corpus 13 times over -- once per experiment, per seed -- is the
single most expensive thing in this study and it is entirely deterministic,
so we do it once and memoise the result.  A cache entry is keyed by
everything that can change the numbers (model, pooling, max length,
normalisation, and a fingerprint of the texts themselves), which means a
stale cache cannot silently poison a run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from layerprobe.config import EncoderConfig
from layerprobe.data import Corpus, EmotionSplit
from layerprobe.encoders import BaseEncoder, build_encoder


@dataclass
class LayerFeatures:
    """Pooled representations of one split at every kept layer."""

    #: ``(n_layers, n_examples, hidden_size)``
    values: np.ndarray
    #: Encoder layer index of each row of ``values`` (0 == embeddings).
    layer_ids: List[int]
    language: str
    split: str

    @property
    def n_layers(self) -> int:
        return self.values.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.values.shape[2]

    def layer(self, layer_id: int) -> np.ndarray:
        """The ``(n, d)`` matrix for one encoder layer index."""

        try:
            row = self.layer_ids.index(layer_id)
        except ValueError as exc:
            raise KeyError(
                f"layer {layer_id} was not extracted; available layers: {self.layer_ids}"
            ) from exc
        return self.values[row]

    def subset(self, idx: Sequence[int]) -> "LayerFeatures":
        return LayerFeatures(self.values[:, list(idx)], list(self.layer_ids), self.language, self.split)


#: ``{language: {split: LayerFeatures}}``
FeatureStore = Dict[str, Dict[str, LayerFeatures]]


def _fingerprint(cfg: EncoderConfig, split: EmotionSplit) -> str:
    """A cache key covering both the encoder settings and the exact texts."""

    hasher = hashlib.blake2b(digest_size=16)
    payload = {
        "model": cfg.model_name,
        "pooling": cfg.pooling,
        "max_length": cfg.max_length,
        "normalize": cfg.normalize,
        "layers": cfg.layers,
        # dtype changes the numbers, so it must change the cache key too.
        "dtype": cfg.dtype,
        "half_precision": cfg.half_precision,
        "synthetic_hidden_size": cfg.synthetic_hidden_size,
        "synthetic_num_layers": cfg.synthetic_num_layers,
        "synthetic_noise": cfg.synthetic_noise,
        "language": split.language,
        "split": split.split,
        "n": len(split),
    }
    hasher.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    for text in split.texts:
        hasher.update(text.encode("utf-8", errors="replace"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.clip(norms, 1e-9, None)


def extract_split(
    split: EmotionSplit,
    cfg: EncoderConfig,
    encoder: BaseEncoder,
    cache_dir: Optional[str | Path] = None,
) -> LayerFeatures:
    """Encode one split, reading from / writing to the cache when given one."""

    layer_ids = list(cfg.layers) if cfg.layers is not None else encoder.layer_ids
    path: Optional[Path] = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{_fingerprint(cfg, split)}.npz"
        if path.exists():
            with np.load(path) as blob:
                return LayerFeatures(
                    blob["values"], [int(i) for i in blob["layer_ids"]], split.language, split.split
                )

    values = encoder.encode(split.texts)
    available = encoder.layer_ids
    missing = [lid for lid in layer_ids if lid not in available]
    if missing:
        raise ValueError(f"requested layers {missing} but the encoder only has {available}")
    values = values[[available.index(lid) for lid in layer_ids]]
    if cfg.normalize:
        values = _normalize(values)
    values = np.ascontiguousarray(values, dtype=np.float32)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written via a handle: np.savez_compressed would append a second
        # ".npz" to a path that does not already end in one.
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, values=values, layer_ids=np.asarray(layer_ids, dtype=np.int32))
        tmp.replace(path)
    return LayerFeatures(values, layer_ids, split.language, split.split)


def extract_corpus(
    corpus: Corpus,
    cfg: EncoderConfig,
    cache_dir: Optional[str | Path] = None,
    encoder: Optional[BaseEncoder] = None,
    verbose: bool = True,
) -> tuple[FeatureStore, BaseEncoder]:
    """Encode every (language, split) in the corpus.

    Returns the feature store and the encoder, so that callers can read the
    layer count off the same object that produced the features.
    """

    encoder = encoder or build_encoder(cfg)
    store: FeatureStore = {}
    for language, splits in corpus.items():
        store[language] = {}
        for split_name, split in splits.items():
            if verbose:
                print(f"  encoding {language}/{split_name} ({len(split)} examples)", flush=True)
            store[language][split_name] = extract_split(split, cfg, encoder, cache_dir)
    return store, encoder


def stack_features(features: Sequence[LayerFeatures]) -> LayerFeatures:
    """Concatenate several splits along the example axis (joint training)."""

    if not features:
        raise ValueError("nothing to stack")
    layer_ids = features[0].layer_ids
    for feat in features[1:]:
        if feat.layer_ids != layer_ids:
            raise ValueError("cannot stack features extracted from different layer sets")
    values = np.concatenate([f.values for f in features], axis=1)
    return LayerFeatures(values, list(layer_ids), "multi", features[0].split)
