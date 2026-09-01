#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line entry point for the layer-wise emotion probing study.

Examples
--------
Offline smoke run (no downloads, finishes in about a minute)::

    python run_experiments.py --config configs/smoke.yaml

The real study on SemEval-2025 Task 11 / BRIGHTER with XLM-R::

    python run_experiments.py --config configs/brighter.yaml

Override anything from the command line::

    python run_experiments.py --config configs/brighter.yaml \\
        --model xlm-roberta-large --languages amh hau eng --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from layerprobe.config import ExperimentConfig, load_config
from layerprobe.pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare transformer layers as emotion representations across languages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML experiment description")
    parser.add_argument("--output-dir", type=str, default=None, help="where results are written")
    parser.add_argument("--cache-dir", type=str, default=None, help="feature cache directory")
    parser.add_argument("--model", type=str, default=None, help="encoder, e.g. xlm-roberta-base")
    parser.add_argument("--pooling", type=str, default=None, choices=["mean", "cls", "max"])
    parser.add_argument("--languages", nargs="+", default=None, help="languages to include")
    parser.add_argument("--source-languages", nargs="+", default=None, help="zero-shot training languages")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--probe", type=str, default=None, choices=["logreg", "mlp"])
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, ...")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None, help="cap training examples per language")
    parser.add_argument("--no-figures", action="store_true", help="skip matplotlib output")
    parser.add_argument("--quiet", action="store_true")
    return parser


def apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir
    if args.model:
        cfg.encoder.model_name = args.model
    if args.pooling:
        cfg.encoder.pooling = args.pooling
    if args.device:
        cfg.encoder.device = args.device
    if args.batch_size:
        cfg.encoder.batch_size = args.batch_size
    if args.languages:
        cfg.data.languages = list(args.languages)
    if args.source_languages:
        cfg.data.source_languages = list(args.source_languages)
    if args.seeds:
        cfg.seeds = list(args.seeds)
    if args.probe:
        cfg.probe.kind = args.probe
    if args.max_train is not None:
        cfg.data.max_train_per_language = args.max_train
    return cfg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else ExperimentConfig()
    cfg = apply_overrides(cfg, args)

    unknown_sources = set(cfg.data.source_languages) - set(cfg.data.languages)
    if unknown_sources:
        print(
            f"error: source language(s) {sorted(unknown_sources)} are not in "
            f"data.languages={cfg.data.languages}",
            file=sys.stderr,
        )
        return 2

    payload = run_experiment(cfg, make_figures=not args.no_figures, verbose=not args.quiet)
    if not args.quiet:
        print()
        print(Path(payload["summary_path"]).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
