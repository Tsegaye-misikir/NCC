# -*- coding: utf-8 -*-
"""Configuration objects for the layer-probing study.

Everything an experiment needs is declared here as a dataclass so that a run
is fully described by one YAML file (which is copied into the output
directory, next to the results it produced).
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Emotion inventory of the SemEval-2025 Task 11 / BRIGHTER corpora.  Datasets
# that use a different inventory just declare their own in the YAML.
DEFAULT_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


@dataclass
class DataConfig:
    """Where the emotion data comes from and how it is split.

    ``source`` selects the loader:

    ``hf``
        A Hugging Face dataset id (``hf_path``/``hf_name``), one config per
        language.  This is how BRIGHTER / SemEval-2025 Task 11 ships.
    ``local``
        CSV or TSV files under ``local_dir``, named
        ``{language}_{split}.csv``; a ``text`` column plus one 0/1 column per
        emotion (multi-label) or a single ``label`` column (single-label).
    ``synthetic``
        A deterministic toy corpus generated in-process.  It downloads
        nothing, so it is what the test-suite and the offline smoke run use.
    """

    source: str = "synthetic"
    languages: List[str] = field(default_factory=lambda: ["amh", "hau", "eng"])
    #: Languages the probes are trained on for the cross-lingual experiments.
    source_languages: List[str] = field(default_factory=lambda: ["eng"])
    #: Low-resource targets we care about; defaults to "everything else".
    target_languages: List[str] = field(default_factory=list)

    emotions: List[str] = field(default_factory=lambda: list(DEFAULT_EMOTIONS))
    #: ``multilabel`` (BRIGHTER-style) or ``singlelabel``.
    task: str = "multilabel"

    hf_path: Optional[str] = None
    hf_name_template: str = "{language}"
    hf_text_column: str = "text"
    local_dir: Optional[str] = None
    local_suffix: str = "csv"

    #: Cap on training examples per language; ``None`` keeps everything.  A cap
    #: makes languages comparable when their corpora differ wildly in size.
    max_train_per_language: Optional[int] = None
    max_eval_per_language: Optional[int] = None
    #: Fraction carved out of train when a corpus ships no dev split.
    dev_fraction: float = 0.1
    #: Fraction carved out of train when a corpus ships no test split either.
    test_fraction: float = 0.2

    synthetic_size: int = 240
    synthetic_vocab: int = 400

    def resolved_target_languages(self) -> List[str]:
        if self.target_languages:
            return list(self.target_languages)
        return [lg for lg in self.languages if lg not in self.source_languages]


@dataclass
class EncoderConfig:
    """The frozen multilingual encoder and how its layers are pooled."""

    #: A Hugging Face model id, or ``synthetic`` for the offline stand-in.
    model_name: str = "synthetic"
    #: ``mean`` (mask-aware mean over tokens), ``cls`` or ``max``.
    pooling: str = "mean"
    max_length: int = 128
    batch_size: int = 32
    device: Optional[str] = None  # ``None`` -> cuda when available, else cpu
    #: fp16 activations during extraction; ignored on CPU.
    half_precision: bool = False
    #: Layers to keep. ``None`` keeps all of them, embeddings (layer 0) included.
    layers: Optional[List[int]] = None
    #: L2-normalise each pooled vector.  Off by default so that layer norm
    #: differences between layers stay visible to the probe's scaler.
    normalize: bool = False
    #: Only used when ``model_name == "synthetic"``.
    synthetic_hidden_size: int = 64
    synthetic_num_layers: int = 12
    #: Noise added to the synthetic representations.  Without it the toy task
    #: saturates at macro-F1 1.0 and every layer looks identical, which
    #: defeats the point of a smoke run.
    synthetic_noise: float = 1.4


@dataclass
class ProbeConfig:
    """The classifier trained on top of a (combination of) layer(s)."""

    #: ``logreg`` (scikit-learn, one-vs-rest for multi-label) or ``mlp``.
    kind: str = "logreg"
    #: Inverse regularisation strength for ``logreg``; a list triggers a
    #: dev-set sweep and the best value is reported.
    C: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    max_iter: int = 2000
    #: Standardise features before fitting.  Important when comparing layers,
    #: whose activation scales differ by an order of magnitude.
    standardize: bool = True
    hidden_size: int = 256  # ``mlp`` only
    epochs: int = 30  # ``mlp`` / scalar-mix only
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    #: Decision threshold for multi-label probes, tuned on dev when ``None``.
    threshold: Optional[float] = None
    #: Fraction of the training set each seed draws (without replacement).
    #: A logistic probe is deterministic, so without this every seed returns
    #: the same number and the reported standard deviations are all zero.
    #: Resampling turns them into what they should measure: how sensitive a
    #: layer's advantage is to *which* few hundred sentences you trained on.
    #: Set to ``None`` to train on the full set at every seed.
    train_subsample: Optional[float] = 0.9


@dataclass
class ScalarMixConfig:
    """The learned weighted combination of layers (ELMo-style scalar mix)."""

    enabled: bool = True
    epochs: int = 60
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    #: Softmax temperature on the layer weights; >1 flattens the mix.
    temperature: float = 1.0
    #: Learn a global scale gamma on top of the mixture, as in ELMo.
    learn_gamma: bool = True
    #: Apply layer norm to each layer before mixing, so that layers with large
    #: activation norms cannot dominate the mixture for free.
    layer_norm: bool = True


@dataclass
class CombinationConfig:
    """Which layer combinations enter the comparison."""

    #: Probe every individual layer (0..L).  This is the backbone of the study.
    individual_layers: bool = True
    #: Uniform average over all layers.
    average_all: bool = True
    #: Uniform averages over sliding windows of these sizes.
    window_sizes: List[int] = field(default_factory=lambda: [4])
    #: Named windows, e.g. ``{"top4": [9, 10, 11, 12]}``; resolved after the
    #: encoder reports how many layers it actually has when left empty.
    named_windows: Dict[str, List[int]] = field(default_factory=dict)
    #: Concatenate these layer groups (dimensionality grows linearly).
    concat_groups: Dict[str, List[int]] = field(default_factory=dict)
    #: Include the learned scalar mix over all layers.
    scalar_mix: bool = True


@dataclass
class AnalysisConfig:
    """Diagnostics that explain the probing numbers."""

    #: Per-layer language-identification probe: how language-specific is a layer?
    language_probe: bool = True
    #: Linear CKA between languages, per layer: how aligned are the spaces?
    cka: bool = True
    #: Cap the sample used for CKA -- it is O(n^2) in memory.
    cka_max_samples: int = 500


@dataclass
class ExperimentConfig:
    """Top-level configuration: one YAML file == one of these."""

    name: str = "layerwise-emotion"
    output_dir: str = "results/layerwise-emotion"
    cache_dir: str = "cache/features"
    seeds: List[int] = field(default_factory=lambda: [13, 42, 1234])
    #: Which experiments to run.
    run_monolingual: bool = True
    run_zeroshot: bool = True
    run_multilingual: bool = True

    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    scalar_mix: ScalarMixConfig = field(default_factory=ScalarMixConfig)
    combinations: CombinationConfig = field(default_factory=CombinationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)


_SECTIONS = {
    "data": DataConfig,
    "encoder": EncoderConfig,
    "probe": ProbeConfig,
    "scalar_mix": ScalarMixConfig,
    "combinations": CombinationConfig,
    "analysis": AnalysisConfig,
}


def _build_section(cls, payload: Optional[Dict[str, Any]]):
    if payload is None:
        return cls()
    if not isinstance(payload, dict):
        raise TypeError(f"section for {cls.__name__} must be a mapping, got {type(payload)}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in section {cls.__name__}; "
            f"known keys are {sorted(known)}"
        )
    return cls(**payload)


def config_from_dict(payload: Dict[str, Any]) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a plain (YAML-shaped) mapping."""

    payload = copy.deepcopy(payload or {})
    sections = {name: _build_section(cls, payload.pop(name, None)) for name, cls in _SECTIONS.items()}
    known_top = {f.name for f in dataclasses.fields(ExperimentConfig)} - set(_SECTIONS)
    unknown = set(payload) - known_top
    if unknown:
        raise ValueError(
            f"unknown top-level key(s) {sorted(unknown)}; known keys are {sorted(known_top)}"
        )
    return ExperimentConfig(**payload, **sections)


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a YAML experiment description from disk."""

    with Path(path).open(encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    return config_from_dict(payload)
