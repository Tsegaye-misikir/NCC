#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare finished runs across encoders.

Each ``run_experiments.py`` invocation writes one ``results.json``.  This
script reads several of them and answers the questions that only appear once
more than one model is in play:

* Does the *shape* of the layer curve survive a change of model -- do all
  encoders peak in the middle, or is that an XLM-R artefact?
* Where does each model's best layer sit as a **fraction of its depth**?
  Comparing raw layer indices across a 13-layer and a 29-layer model is
  meaningless; relative depth is the only comparable axis.
* Is a model's advantage real, or is it just seeing more text? The fertility
  table exposes the truncation confound behind any cross-model claim.

Usage::

    python compare_models.py results/models/xlmr-base results/models/qwen3-0.6b
    python compare_models.py results/models/* --experiment zeroshot -o results/comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_run(directory: str | Path) -> Optional[dict]:
    """Read one run's ``results.json``, or ``None`` if it is not there."""

    path = Path(directory) / "results.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["_dir"] = str(directory)
    return payload


def _model_label(payload: dict) -> str:
    encoder = payload.get("encoder_summary") or {}
    if encoder.get("model_name"):
        return str(encoder["model_name"])
    config = payload.get("config") or {}
    return str((config.get("encoder") or {}).get("model_name", Path(payload["_dir"]).name))


def assign_labels(payloads: List[dict]) -> None:
    """Give every run a label that is unique within this comparison.

    Two runs can legitimately share a model name -- the same encoder with
    ``mean`` and ``last_token`` pooling, say, or a trimmed config against the
    full one.  Grouping those under one label would silently average two
    different experiments together, so collisions get disambiguated by
    whatever actually differs: the pooling, then the run's name, then its
    directory.
    """

    counts: Dict[str, int] = {}
    for payload in payloads:
        counts[_model_label(payload)] = counts.get(_model_label(payload), 0) + 1

    for payload in payloads:
        base = _model_label(payload)
        if counts[base] == 1:
            payload["_label"] = base
            continue
        encoder = payload.get("encoder_summary") or {}
        config = payload.get("config") or {}
        for suffix in (encoder.get("pooling"), config.get("name"), Path(payload["_dir"]).name):
            if suffix:
                candidate = f"{base} ({suffix})"
                if candidate not in {p.get("_label") for p in payloads}:
                    payload["_label"] = candidate
                    break
        else:
            payload["_label"] = f"{base} ({payload['_dir']})"


def _label(payload: dict) -> str:
    return payload.get("_label") or _model_label(payload)


def summary_table(payloads: List[dict], experiment: str = "zeroshot") -> pd.DataFrame:
    """Best combination per model and language, with depth expressed relatively."""

    rows: List[dict] = []
    for payload in payloads:
        model = _label(payload)
        encoder = payload.get("encoder_summary") or {}
        depth = max(1, int(encoder.get("num_layers", 0)) or 1)
        for entry in payload.get("best", []):
            if entry.get("experiment") != experiment:
                continue
            layers = entry.get("best_layers") or []
            row = {
                "model": model,
                "n_layers": depth,
                "language": entry["eval_language"],
                "best_combination": entry["best_combination"],
                "best_macro_f1": entry["best_macro_f1"],
                "last_layer_macro_f1": entry.get("last_layer_macro_f1"),
                "gain_over_last": entry.get("gain_over_last"),
            }
            if len(layers) == 1:
                # The comparable axis across models of different depth.
                row["best_depth_fraction"] = round(float(layers[0]) / depth, 3)
            rows.append(row)
    return pd.DataFrame(rows)


def layer_curves(payloads: List[dict], experiment: str = "zeroshot") -> pd.DataFrame:
    """Single-layer macro-F1 per model, with a relative-depth column."""

    rows: List[dict] = []
    for payload in payloads:
        model = _label(payload)
        depth = max(1, int((payload.get("encoder_summary") or {}).get("num_layers", 0)) or 1)
        seen = set()
        for result in payload.get("results", []):
            if result.get("experiment") != experiment or result.get("kind") != "single":
                continue
            layers = result.get("layers") or []
            if len(layers) != 1:
                continue
            key = (result["eval_language"], int(layers[0]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "model": model,
                    "language": result["eval_language"],
                    "layer": int(layers[0]),
                    "depth_fraction": round(float(layers[0]) / depth, 4),
                    "macro_f1": result.get("macro_f1_mean"),
                    "macro_f1_std": result.get("macro_f1_std", 0.0),
                }
            )
    return pd.DataFrame(rows)


def fertility_table(payloads: List[dict]) -> pd.DataFrame:
    """Tokens per character and truncation rate, per model and language."""

    rows: List[dict] = []
    for payload in payloads:
        model = _label(payload)
        for entry in payload.get("fertility", []) or []:
            rows.append({"model": model, **entry})
    return pd.DataFrame(rows)


def plot_relative_curves(curves: pd.DataFrame, output_dir: Path, experiment: str) -> Optional[str]:
    """Layer curves on a shared relative-depth axis, one panel per language."""

    if curves.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    languages = sorted(curves["language"].unique())
    fig, axes = plt.subplots(
        1, len(languages), figsize=(4.2 * len(languages), 4.2), sharey=True, squeeze=False
    )
    for ax, language in zip(axes[0], languages):
        subset = curves[curves.language == language]
        for model, group in subset.groupby("model"):
            group = group.sort_values("depth_fraction")
            ax.plot(group.depth_fraction, group.macro_f1, marker="o", ms=3, label=str(model))
        ax.set_title(language)
        ax.set_xlabel("relative depth (0 = embeddings, 1 = final)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("macro-F1")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(f"Layer curves across models -- {experiment}")
    fig.tight_layout()
    path = output_dir / f"model_comparison_{experiment}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare layer-probing runs across encoders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", help="directories written by run_experiments.py")
    parser.add_argument("--experiment", default="zeroshot", help="which regime to compare")
    parser.add_argument("-o", "--output-dir", default="results/comparison")
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    payloads, missing = [], []
    for directory in args.run_dirs:
        payload = load_run(directory)
        (payloads if payload else missing).append(payload or directory)
    if missing:
        print(f"warning: no results.json in {', '.join(map(str, missing))}")
    if len(payloads) < 2:
        print("error: need at least two completed runs to compare")
        return 2

    assign_labels(payloads)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = summary_table(payloads, args.experiment)
    curves = layer_curves(payloads, args.experiment)
    fertility = fertility_table(payloads)

    for name, frame in (("summary", summary), ("layer_curves", curves), ("fertility", fertility)):
        if not frame.empty:
            frame.to_csv(out / f"model_{name}_{args.experiment}.csv", index=False)

    if summary.empty:
        print(f"error: no '{args.experiment}' results found in those runs")
        return 2

    print(f"\n=== best per model and language ({args.experiment}) ===")
    print(summary.to_string(index=False))

    if "gain_over_last" in summary.columns and summary["gain_over_last"].notna().any():
        print("\n=== mean gain over the final layer, by model ===")
        print(
            summary.groupby("model")["gain_over_last"]
            .agg(["mean", "min", "max", "count"])
            .round(4)
            .to_string()
        )
    if "best_depth_fraction" in summary.columns and summary["best_depth_fraction"].notna().any():
        print("\n=== where the best single layer sits, as a fraction of depth ===")
        print(summary.groupby("model")["best_depth_fraction"].describe()[["mean", "min", "max"]].round(3).to_string())

    if not fertility.empty:
        print("\n=== tokenizer fertility (higher = more tokens per character) ===")
        pivot = fertility.pivot_table(index="model", columns="language", values="tokens_per_char")
        print(pivot.round(3).to_string())
        if "truncated_fraction" in fertility.columns and (fertility.truncated_fraction > 0.05).any():
            print("\n  NOTE: some model/language pairs truncate >5% of examples at max_length.")
            print("  Those models are seeing less text, which confounds the comparison.")
            worst = fertility[fertility.truncated_fraction > 0.05]
            print(worst[["model", "language", "truncated_fraction"]].to_string(index=False))

    if not args.no_figures:
        figure = plot_relative_curves(curves, out, args.experiment)
        if figure:
            print(f"\nwrote {figure}")
    print(f"tables written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
