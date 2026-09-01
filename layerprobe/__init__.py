# -*- coding: utf-8 -*-
"""
layerprobe
==========

Which transformer layers of a multilingual encoder carry the most useful
signal for emotion recognition in low-resource languages?

The package freezes a pretrained multilingual encoder (XLM-R by default),
extracts sentence representations from *every* hidden layer, and trains a
light-weight probe on top of individual layers, of the final layer, and of
simple combinations of layers (uniform averages over windows and a learned
weighted mix).  The same features drive a set of diagnostics -- a per-layer
language-identification probe and cross-lingual CKA -- which say *why* a
layer transfers well or badly.

Typical use is through the CLI::

    python run_experiments.py --config configs/brighter.yaml

but every stage is importable on its own; see ``docs/LAYERWISE_EMOTION.md``.
"""

from layerprobe.config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config"]
__version__ = "0.1.0"
