# -*- coding: utf-8 -*-
"""End-to-end run: load data -> encode -> probe -> diagnose -> report."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from layerprobe.analysis import (
    correlate_alignment_with_transfer,
    cross_lingual_alignment,
    language_identification,
)
from layerprobe.config import ExperimentConfig
from layerprobe.data import describe, load_corpus
from layerprobe.experiments import run_all
from layerprobe.features import extract_corpus
from layerprobe.reporting import make_plots, save_summary, save_tables


def run_experiment(
    cfg: ExperimentConfig,
    output_dir: Optional[str | Path] = None,
    make_figures: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Execute the whole study described by ``cfg`` and write its outputs."""

    out = Path(output_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_seed = cfg.seeds[0] if cfg.seeds else 0

    if verbose:
        print(f"[1/5] loading corpus ({cfg.data.source})", flush=True)
    corpus = load_corpus(cfg.data, seed=base_seed)

    if verbose:
        print(f"[2/5] extracting layer features with {cfg.encoder.model_name}", flush=True)
    store, encoder = extract_corpus(corpus, cfg.encoder, cfg.cache_dir, verbose=verbose)
    layer_ids = cfg.encoder.layers if cfg.encoder.layers is not None else encoder.layer_ids

    if verbose:
        print("[3/5] probing layers and combinations", flush=True)
    payload: Dict[str, object] = dict(run_all(cfg, corpus, store, layer_ids, verbose=verbose))
    payload["config"] = cfg.to_dict()
    payload["data_summary"] = describe(corpus)

    if verbose:
        print("[4/5] running representation diagnostics", flush=True)
    languages = list(corpus)
    if cfg.analysis.language_probe:
        payload["language_probe"] = language_identification(
            store, languages, layer_ids, seed=base_seed
        )
    if cfg.analysis.cka:
        payload["alignment"] = cross_lingual_alignment(
            store,
            [lg for lg in cfg.data.source_languages if lg in corpus],
            [lg for lg in cfg.data.resolved_target_languages() if lg in corpus],
            layer_ids,
            max_samples=cfg.analysis.cka_max_samples,
            seed=base_seed,
        )
    payload["diagnostics_correlation"] = correlate_alignment_with_transfer(
        payload.get("alignment", []),
        payload.get("language_probe", []),
        [r for r in payload["results"] if r.get("experiment") == "zeroshot"],
    )

    if verbose:
        print("[5/5] writing tables and figures", flush=True)
    cfg.save(out / "config.yaml")
    payload["written"] = save_tables(payload, out)
    payload["summary_path"] = save_summary(payload, out)
    if make_figures:
        payload["figures"] = make_plots(payload, out)
    if verbose:
        print(f"done in {payload.get('runtime_seconds')}s -- results in {out}", flush=True)
    return payload
