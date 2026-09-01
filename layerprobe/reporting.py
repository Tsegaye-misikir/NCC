# -*- coding: utf-8 -*-
"""Turning result rows into tables, plots and a readable summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def to_frame(rows: Sequence[dict]) -> pd.DataFrame:
    """Result rows as a DataFrame, with list columns rendered readably."""

    frame = pd.DataFrame(list(rows))
    for column in ("layers", "train_languages", "macro_f1_per_seed", "layer_weights_mean"):
        if column in frame.columns:
            frame[column] = frame[column].apply(
                lambda v: ",".join(str(round(x, 4) if isinstance(x, float) else x) for x in v)
                if isinstance(v, (list, tuple))
                else v
            )
    return frame


def save_tables(payload: Dict[str, object], output_dir: str | Path) -> Dict[str, str]:
    """Write every result table to ``output_dir`` as CSV, plus a JSON dump."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    for key in ("results", "best", "language_probe", "alignment", "data_summary", "combinations"):
        rows = payload.get(key)
        if not rows:
            continue
        path = out / f"{key}.csv"
        to_frame(rows).to_csv(path, index=False)
        written[key] = str(path)

    ranking = layer_ranking(payload.get("results", []))
    if not ranking.empty:
        path = out / "layer_ranking.csv"
        ranking.to_csv(path, index=False)
        written["layer_ranking"] = str(path)

    path = out / "results.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    written["json"] = str(path)
    return written


def layer_ranking(rows: Sequence[dict], metric: str = "macro_f1_mean") -> pd.DataFrame:
    """Average rank of each combination across languages, per experiment.

    A combination that wins in one language and collapses in another is not
    a useful recommendation; mean rank across evaluation languages is the
    number that survives that.
    """

    frame = pd.DataFrame(list(rows))
    if frame.empty or metric not in frame.columns:
        return pd.DataFrame()
    frame = frame.dropna(subset=[metric])
    frame["rank"] = frame.groupby(["experiment", "eval_language"])[metric].rank(
        ascending=False, method="min"
    )
    grouped = (
        frame.groupby(["experiment", "combination", "kind"])
        .agg(
            mean_rank=("rank", "mean"),
            best_rank=("rank", "min"),
            worst_rank=("rank", "max"),
            mean_macro_f1=(metric, "mean"),
            n_languages=(metric, "size"),
        )
        .reset_index()
    )
    return grouped.sort_values(["experiment", "mean_rank"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _single_layer_frame(rows: Sequence[dict], experiment: str) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    frame = frame[(frame["experiment"] == experiment) & (frame["kind"] == "single")].copy()
    if frame.empty:
        return frame
    frame["layer"] = frame["layers"].apply(lambda v: int(v[0]) if isinstance(v, (list, tuple)) else int(v))
    # ``last`` duplicates the final layer's row; keep one point per layer.
    return frame.drop_duplicates(subset=["eval_language", "layer"]).sort_values("layer")


def plot_layer_curves(
    rows: Sequence[dict], output_dir: str | Path, experiments: Iterable[str] = ("monolingual", "zeroshot")
) -> List[str]:
    """One macro-F1-vs-layer curve per evaluation language, per experiment."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    for experiment in experiments:
        frame = _single_layer_frame(rows, experiment)
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for language, group in frame.groupby("eval_language"):
            group = group.sort_values("layer")
            ax.plot(group["layer"], group["macro_f1_mean"], marker="o", label=str(language))
            if "macro_f1_std" in group:
                ax.fill_between(
                    group["layer"],
                    group["macro_f1_mean"] - group["macro_f1_std"],
                    group["macro_f1_mean"] + group["macro_f1_std"],
                    alpha=0.15,
                )
        final_layer = int(frame["layer"].max())
        ax.axvline(final_layer, color="grey", linestyle="--", linewidth=1)
        ax.annotate(
            "final layer",
            xy=(final_layer, ax.get_ylim()[0]),
            xytext=(-6, 6),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color="grey",
        )
        ax.set_xlabel("encoder layer (0 = embeddings)")
        ax.set_ylabel("macro-F1")
        ax.set_title(f"Emotion probing accuracy by layer -- {experiment}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out / f"layer_curve_{experiment}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))
    return written


def plot_combination_comparison(
    rows: Sequence[dict], output_dir: str | Path, experiment: str = "zeroshot", top_n: int = 12
) -> Optional[str]:
    """Bar chart of the best combinations against the final-layer baseline."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(list(rows))
    if frame.empty or "macro_f1_mean" not in frame.columns:
        return None
    frame = frame[frame["experiment"] == experiment]
    if frame.empty:
        return None

    means = (
        frame.groupby(["combination", "kind"])["macro_f1_mean"].mean().reset_index().sort_values(
            "macro_f1_mean", ascending=False
        )
    )
    baseline_rows = means[means["combination"] == "last"]
    keep = means.head(top_n)
    if not baseline_rows.empty and "last" not in set(keep["combination"]):
        keep = pd.concat([keep, baseline_rows])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colours = ["#c44e52" if c == "last" else "#4c72b0" for c in keep["combination"]]
    ax.bar(keep["combination"], keep["macro_f1_mean"], color=colours)
    if not baseline_rows.empty:
        ax.axhline(
            float(baseline_rows["macro_f1_mean"].iloc[0]),
            color="#c44e52",
            linestyle="--",
            linewidth=1,
            label="final layer",
        )
        ax.legend(fontsize=8)
    ax.set_ylabel("macro-F1 (mean over eval languages)")
    ax.set_title(f"Layer combinations -- {experiment}")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = Path(output_dir) / f"combinations_{experiment}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_scalar_mix_weights(rows: Sequence[dict], output_dir: str | Path) -> Optional[str]:
    """Learned scalar-mix weight per layer, one line per setting."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = [
        r for r in rows if r.get("combination") == "scalar_mix" and r.get("layer_weights_mean")
    ]
    if not entries:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in entries:
        weights = row["layer_weights_mean"]
        ax.plot(
            list(row["layers"]),
            weights,
            marker="o",
            alpha=0.8,
            label=f"{row['experiment']}/{row['eval_language']}",
        )
    ax.set_xlabel("encoder layer (0 = embeddings)")
    ax.set_ylabel("learned softmax weight")
    ax.set_title("What the scalar mix asks for")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(output_dir) / "scalar_mix_weights.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_diagnostics(
    language_rows: Sequence[dict], alignment_rows: Sequence[dict], output_dir: str | Path
) -> Optional[str]:
    """Language identifiability and cross-lingual CKA, side by side by layer."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not language_rows and not alignment_rows:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if language_rows:
        frame = pd.DataFrame(list(language_rows)).sort_values("layer")
        axes[0].plot(frame["layer"], frame["language_id_accuracy"], marker="o", color="#c44e52")
        axes[0].plot(frame["layer"], frame["chance"], linestyle="--", color="grey", label="chance")
        axes[0].legend(fontsize=8)
    axes[0].set_title("Language identifiability by layer")
    axes[0].set_xlabel("encoder layer")
    axes[0].set_ylabel("language-ID accuracy")
    axes[0].grid(alpha=0.3)

    if alignment_rows:
        frame = pd.DataFrame(list(alignment_rows))
        for (source, target), group in frame.groupby(["source", "target"]):
            group = group.sort_values("layer")
            axes[1].plot(group["layer"], group["cka"], marker="o", alpha=0.8, label=f"{source}->{target}")
        axes[1].legend(fontsize=7, ncol=2)
    axes[1].set_title("Cross-lingual CKA by layer")
    axes[1].set_xlabel("encoder layer")
    axes[1].set_ylabel("linear CKA")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = Path(output_dir) / "diagnostics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def make_plots(payload: Dict[str, object], output_dir: str | Path) -> List[str]:
    """Render every plot the payload has data for."""

    rows = payload.get("results", [])
    written = plot_layer_curves(rows, output_dir)
    for maybe in (
        plot_combination_comparison(rows, output_dir, "zeroshot"),
        plot_combination_comparison(rows, output_dir, "monolingual"),
        plot_scalar_mix_weights(rows, output_dir),
        plot_diagnostics(payload.get("language_probe", []), payload.get("alignment", []), output_dir),
    ):
        if maybe:
            written.append(maybe)
    return written


# --------------------------------------------------------------------------
# Text summary
# --------------------------------------------------------------------------


def summary_markdown(payload: Dict[str, object]) -> str:
    """A short human-readable digest, written next to the CSVs."""

    lines: List[str] = ["# Layer-wise emotion probing -- summary", ""]

    config = payload.get("config", {})
    encoder = config.get("encoder", {}) if isinstance(config, dict) else {}
    data = config.get("data", {}) if isinstance(config, dict) else {}
    lines += [
        f"- encoder: `{encoder.get('model_name', '?')}` (pooling: {encoder.get('pooling', '?')})",
        f"- languages: {', '.join(data.get('languages', []))}",
        f"- source languages: {', '.join(data.get('source_languages', []))}",
        f"- seeds: {config.get('seeds')}",
        "",
    ]

    best = payload.get("best", [])
    if best:
        lines += [
            "## Best combination per setting",
            "",
            "| experiment | language | best | macro-F1 | final layer | gain |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in best:
            gain = row.get("gain_over_last")
            lines.append(
                "| {experiment} | {lang} | {best} | {f1:.4f} | {last} | {gain} |".format(
                    experiment=row["experiment"],
                    lang=row["eval_language"],
                    best=row["best_combination"],
                    f1=row["best_macro_f1"],
                    last=(
                        f"{row['last_layer_macro_f1']:.4f}" if "last_layer_macro_f1" in row else "n/a"
                    ),
                    gain=f"{gain:+.4f}" if gain is not None else "n/a",
                )
            )
        lines.append("")

    ranking = layer_ranking(payload.get("results", []))
    if not ranking.empty:
        lines += ["## Combinations by mean rank across languages", ""]
        for experiment, group in ranking.groupby("experiment"):
            lines.append(f"**{experiment}**")
            lines.append("")
            lines.append("| combination | mean rank | mean macro-F1 |")
            lines.append("| --- | --- | --- |")
            for _, row in group.head(8).iterrows():
                lines.append(
                    f"| {row['combination']} | {row['mean_rank']:.2f} | {row['mean_macro_f1']:.4f} |"
                )
            lines.append("")

    negative = [
        r
        for r in payload.get("results", [])
        if r.get("experiment") == "zeroshot" and r.get("combination") == "last" and "transfer_gap" in r
    ]
    if negative:
        lines += ["## Zero-shot transfer gap with the default (final-layer) representation", ""]
        lines.append("| target language | zero-shot macro-F1 | in-language | gap |")
        lines.append("| --- | --- | --- | --- |")
        for row in sorted(negative, key=lambda r: -r["transfer_gap"]):
            lines.append(
                "| {lang} | {zs:.4f} | {inl:.4f} | {gap:+.4f} |".format(
                    lang=row["eval_language"],
                    zs=row["macro_f1_mean"],
                    inl=row.get("in_language_macro_f1", float("nan")),
                    gap=row["transfer_gap"],
                )
            )
        lines.append("")

    correlations = payload.get("diagnostics_correlation") or {}
    if correlations:
        lines += ["## Do the diagnostics predict transfer?", ""]
        for key, value in correlations.items():
            lines.append(f"- `{key}`: {value:.4f}" if isinstance(value, float) else f"- `{key}`: {value}")
        lines.append("")

    return "\n".join(lines)


def save_summary(payload: Dict[str, object], output_dir: str | Path) -> str:
    path = Path(output_dir) / "SUMMARY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_markdown(payload), encoding="utf-8")
    return str(path)
