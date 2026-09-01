# -*- coding: utf-8 -*-
"""Cross-model comparison of finished runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compare_models import (
    assign_labels,
    fertility_table,
    layer_curves,
    load_run,
    main,
    summary_table,
)


def _write_run(directory: Path, model: str, depth: int, best_layer: int, name: str = "run") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"name": name, "encoder": {"model_name": model}},
        "encoder_summary": {
            "model_name": model,
            "pooling": "mean",
            "hidden_size": 64,
            "num_layers": depth,
            "layer_ids": list(range(depth + 1)),
            "causal": False,
        },
        "best": [
            {
                "experiment": "zeroshot",
                "eval_language": "amh",
                "best_combination": f"layer{best_layer}",
                "best_layers": [best_layer],
                "best_macro_f1": 0.55,
                "last_layer_macro_f1": 0.40,
                "gain_over_last": 0.15,
            }
        ],
        "results": [
            {
                "experiment": "zeroshot",
                "eval_language": "amh",
                "combination": f"layer{l}",
                "kind": "single",
                "layers": [l],
                "macro_f1_mean": 0.3 + 0.01 * l,
                "macro_f1_std": 0.01,
            }
            for l in range(depth + 1)
        ],
        "fertility": [
            {"language": "amh", "tokens_per_char": 0.9, "truncated_fraction": 0.3, "mean_tokens": 90.0}
        ],
    }
    (directory / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def test_load_run_returns_none_when_absent(tmp_path):
    assert load_run(tmp_path / "nothing") is None


def test_depth_fraction_makes_models_of_different_depth_comparable(tmp_path):
    """A layer index means nothing across depths; a fraction of depth does."""

    runs = [
        load_run(_write_run(tmp_path / "a", "xlm-roberta-base", 12, 8)),
        load_run(_write_run(tmp_path / "b", "Qwen/Qwen3-0.6B-Base", 28, 19)),
    ]
    assign_labels(runs)
    summary = summary_table(runs)
    fractions = dict(zip(summary["model"], summary["best_depth_fraction"]))
    # layer 8 of 12 and layer 19 of 28 sit at nearly the same relative depth
    assert fractions["xlm-roberta-base"] == pytest.approx(0.667, abs=0.01)
    assert fractions["Qwen/Qwen3-0.6B-Base"] == pytest.approx(0.679, abs=0.01)


def test_labels_disambiguate_runs_of_the_same_model(tmp_path):
    """Two runs of one model must not be averaged together."""

    runs = [
        load_run(_write_run(tmp_path / "a", "same-model", 12, 8, name="mean-pooled")),
        load_run(_write_run(tmp_path / "b", "same-model", 12, 4, name="last-token")),
    ]
    for run, pooling in zip(runs, ("mean", "last_token")):
        run["encoder_summary"]["pooling"] = pooling
    assign_labels(runs)
    labels = {r["_label"] for r in runs}
    assert len(labels) == 2
    assert summary_table(runs)["model"].nunique() == 2


def test_distinct_models_keep_their_plain_names(tmp_path):
    runs = [
        load_run(_write_run(tmp_path / "a", "xlm-roberta-base", 12, 8)),
        load_run(_write_run(tmp_path / "b", "google/gemma-3-1b-pt", 26, 18)),
    ]
    assign_labels(runs)
    assert {r["_label"] for r in runs} == {"xlm-roberta-base", "google/gemma-3-1b-pt"}


def test_layer_curves_cover_every_layer(tmp_path):
    runs = [load_run(_write_run(tmp_path / "a", "m", 6, 4))]
    assign_labels(runs)
    curves = layer_curves(runs)
    assert len(curves) == 7
    assert curves.depth_fraction.min() == 0.0
    assert curves.depth_fraction.max() == 1.0


def test_fertility_table_collects_every_model(tmp_path):
    runs = [
        load_run(_write_run(tmp_path / "a", "m1", 6, 4)),
        load_run(_write_run(tmp_path / "b", "m2", 6, 4)),
    ]
    assign_labels(runs)
    assert set(fertility_table(runs)["model"]) == {"m1", "m2"}


def test_cli_needs_two_runs(tmp_path, capsys):
    _write_run(tmp_path / "only", "m", 6, 4)
    assert main([str(tmp_path / "only")]) == 2
    assert "at least two" in capsys.readouterr().out


def test_cli_writes_tables_and_warns_about_truncation(tmp_path, capsys):
    _write_run(tmp_path / "a", "xlm-roberta-base", 12, 8)
    _write_run(tmp_path / "b", "Qwen/Qwen3-0.6B-Base", 28, 19)
    out = tmp_path / "cmp"
    assert main([str(tmp_path / "a"), str(tmp_path / "b"), "-o", str(out), "--no-figures"]) == 0
    printed = capsys.readouterr().out
    assert "best per model and language" in printed
    # 30% truncation must be surfaced, not buried
    assert "truncate >5%" in printed
    assert (out / "model_summary_zeroshot.csv").exists()
    assert (out / "model_fertility_zeroshot.csv").exists()


def test_cli_reports_an_unknown_experiment(tmp_path, capsys):
    _write_run(tmp_path / "a", "m1", 6, 4)
    _write_run(tmp_path / "b", "m2", 6, 4)
    code = main(
        [str(tmp_path / "a"), str(tmp_path / "b"), "--experiment", "nonexistent",
         "-o", str(tmp_path / "cmp"), "--no-figures"]
    )
    assert code == 2
    assert "no 'nonexistent' results" in capsys.readouterr().out
