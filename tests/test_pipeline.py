# -*- coding: utf-8 -*-
"""Config parsing, experiment runners, reporting, and one end-to-end run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from layerprobe.config import ExperimentConfig, config_from_dict, load_config
from layerprobe.experiments import (
    _resample_train,
    add_transfer_columns,
    best_per_setting,
    run_all,
)
from layerprobe.features import LayerFeatures, extract_corpus
from layerprobe.data import load_corpus
from layerprobe.pipeline import run_experiment
from layerprobe.reporting import layer_ranking, summary_markdown, to_frame
from run_experiments import apply_overrides, build_parser, main


def _fast_config(tmp_path: Path, **overrides) -> ExperimentConfig:
    cfg = ExperimentConfig(
        name="test",
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        seeds=[0, 1],
    )
    cfg.data.source = "synthetic"
    cfg.data.languages = ["eng", "amh"]
    cfg.data.source_languages = ["eng"]
    cfg.data.synthetic_size = 90
    cfg.encoder.model_name = "synthetic"
    cfg.encoder.synthetic_hidden_size = 12
    cfg.encoder.synthetic_num_layers = 3
    cfg.probe.C = [1.0]
    cfg.probe.max_iter = 200
    cfg.scalar_mix.epochs = 5
    cfg.combinations.window_sizes = [2]
    cfg.analysis.cka_max_samples = 40
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_config_round_trips_through_yaml(tmp_path):
    cfg = _fast_config(tmp_path)
    path = tmp_path / "config.yaml"
    cfg.save(path)
    reloaded = load_config(path)
    assert reloaded.to_dict() == cfg.to_dict()


def test_shipped_configs_parse():
    for path in sorted(Path("configs").glob("*.yaml")):
        cfg = load_config(path)
        assert cfg.data.languages
        assert set(cfg.data.source_languages) <= set(cfg.data.languages)


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        config_from_dict({"encoder": {"model_nmae": "typo"}})
    with pytest.raises(ValueError, match="unknown top-level key"):
        config_from_dict({"seedz": [1]})


def test_target_languages_default_to_the_non_sources():
    cfg = config_from_dict({"data": {"languages": ["a", "b", "c"], "source_languages": ["a"]}})
    assert cfg.data.resolved_target_languages() == ["b", "c"]


def test_explicit_target_languages_are_honoured():
    cfg = config_from_dict(
        {"data": {"languages": ["a", "b", "c"], "source_languages": ["a"], "target_languages": ["c"]}}
    )
    assert cfg.data.resolved_target_languages() == ["c"]


# --------------------------------------------------------------------------
# experiment plumbing
# --------------------------------------------------------------------------


def test_resample_train_is_seed_dependent_and_stable():
    features = LayerFeatures(np.zeros((2, 100, 3), dtype=np.float32), [0, 1], "eng", "train")
    labels = np.arange(100)
    a = _resample_train((features, labels), 0.9, seed=0)[1]
    b = _resample_train((features, labels), 0.9, seed=1)[1]
    again = _resample_train((features, labels), 0.9, seed=0)[1]
    assert len(a) == 90
    np.testing.assert_array_equal(a, again)
    assert not np.array_equal(a, b)


def test_resample_train_is_a_noop_when_disabled():
    features = LayerFeatures(np.zeros((2, 50, 3), dtype=np.float32), [0, 1], "eng", "train")
    labels = np.arange(50)
    out = _resample_train((features, labels), None, seed=0)
    assert out[1].shape[0] == 50


def test_transfer_columns_reference_the_in_language_probe():
    rows = [
        {
            "experiment": "monolingual",
            "eval_language": "amh",
            "combination": "last",
            "macro_f1_mean": 0.60,
            "majority_macro_f1": 0.2,
        },
        {
            "experiment": "zeroshot",
            "eval_language": "amh",
            "combination": "last",
            "macro_f1_mean": 0.45,
            "majority_macro_f1": 0.2,
        },
    ]
    annotated = add_transfer_columns(rows)
    zero_shot = annotated[1]
    assert zero_shot["transfer_gap"] == pytest.approx(0.15)
    assert zero_shot["in_language_macro_f1"] == pytest.approx(0.60)
    assert zero_shot["above_majority"] == pytest.approx(0.25)
    assert "transfer_gap" not in annotated[0]


def test_best_per_setting_reports_gain_over_the_final_layer():
    rows = [
        {
            "experiment": "zeroshot",
            "eval_language": "amh",
            "combination": "last",
            "kind": "single",
            "layers": [12],
            "macro_f1_mean": 0.40,
        },
        {
            "experiment": "zeroshot",
            "eval_language": "amh",
            "combination": "layer8",
            "kind": "single",
            "layers": [8],
            "macro_f1_mean": 0.52,
        },
    ]
    best = best_per_setting(rows)[0]
    assert best["best_combination"] == "layer8"
    assert best["gain_over_last"] == pytest.approx(0.12)
    assert best["last_layer_macro_f1"] == pytest.approx(0.40)


def test_run_all_produces_every_experiment(tmp_path):
    cfg = _fast_config(tmp_path)
    corpus = load_corpus(cfg.data, seed=0)
    store, encoder = extract_corpus(corpus, cfg.encoder, cfg.cache_dir, verbose=False)
    payload = run_all(cfg, corpus, store, encoder.layer_ids, verbose=False)

    experiments = {r["experiment"] for r in payload["results"]}
    assert experiments == {"monolingual", "zeroshot", "multilingual"}
    # zero-shot must never evaluate on a training language
    assert all(
        r["eval_language"] not in r["train_languages"]
        for r in payload["results"]
        if r["experiment"] == "zeroshot"
    )
    # every combination appears for every setting
    names = {r["combination"] for r in payload["results"]}
    assert {c["name"] for c in payload["combinations"]} == names
    assert all("macro_f1_mean" in r for r in payload["results"])
    assert any(r.get("layer_weights_mean") for r in payload["results"])


def test_zeroshot_is_skipped_without_targets(tmp_path):
    cfg = _fast_config(tmp_path)
    cfg.data.languages = ["eng"]
    cfg.data.source_languages = ["eng"]
    corpus = load_corpus(cfg.data, seed=0)
    store, encoder = extract_corpus(corpus, cfg.encoder, cfg.cache_dir, verbose=False)
    payload = run_all(cfg, corpus, store, encoder.layer_ids, verbose=False)
    assert {r["experiment"] for r in payload["results"]} == {"monolingual"}


def test_scalar_mix_weights_are_a_distribution_over_layers(tmp_path):
    cfg = _fast_config(tmp_path)
    corpus = load_corpus(cfg.data, seed=0)
    store, encoder = extract_corpus(corpus, cfg.encoder, cfg.cache_dir, verbose=False)
    payload = run_all(cfg, corpus, store, encoder.layer_ids, verbose=False)
    row = next(r for r in payload["results"] if r["combination"] == "scalar_mix")
    weights = row["layer_weights_mean"]
    assert len(weights) == len(encoder.layer_ids)
    assert sum(weights) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def test_layer_ranking_orders_by_mean_rank():
    rows = [
        {"experiment": "zeroshot", "eval_language": lang, "combination": name, "kind": "single",
         "macro_f1_mean": value}
        for lang, values in {"amh": {"good": 0.6, "bad": 0.3}, "hau": {"good": 0.5, "bad": 0.4}}.items()
        for name, value in values.items()
    ]
    ranking = layer_ranking(rows)
    assert list(ranking["combination"]) == ["good", "bad"]
    assert ranking.iloc[0]["mean_rank"] == 1.0
    assert ranking.iloc[0]["n_languages"] == 2


def test_layer_ranking_is_empty_without_scores():
    assert layer_ranking([]).empty


def test_to_frame_renders_list_columns():
    frame = to_frame([{"combination": "avg", "layers": [1, 2, 3], "macro_f1_mean": 0.5}])
    assert frame.loc[0, "layers"] == "1,2,3"


def test_summary_markdown_mentions_the_key_numbers():
    payload = {
        "config": {"encoder": {"model_name": "xlm-roberta-base", "pooling": "mean"},
                   "data": {"languages": ["eng", "amh"], "source_languages": ["eng"]},
                   "seeds": [1]},
        "best": [{"experiment": "zeroshot", "eval_language": "amh", "best_combination": "layer8",
                  "best_macro_f1": 0.52, "last_layer_macro_f1": 0.40, "gain_over_last": 0.12}],
        "results": [{"experiment": "zeroshot", "eval_language": "amh", "combination": "last",
                     "kind": "single", "macro_f1_mean": 0.40, "transfer_gap": 0.2,
                     "in_language_macro_f1": 0.6}],
        "diagnostics_correlation": {"pearson_cka_vs_transfer": 0.8},
    }
    text = summary_markdown(payload)
    assert "xlm-roberta-base" in text
    assert "layer8" in text
    assert "+0.1200" in text
    assert "pearson_cka_vs_transfer" in text


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def test_full_pipeline_writes_every_artifact(tmp_path):
    cfg = _fast_config(tmp_path)
    payload = run_experiment(cfg, make_figures=True, verbose=False)
    out = Path(cfg.output_dir)

    for name in ("results.csv", "best.csv", "layer_ranking.csv", "language_probe.csv",
                 "alignment.csv", "data_summary.csv", "results.json", "SUMMARY.md", "config.yaml"):
        assert (out / name).exists(), f"{name} was not written"
    assert (out / "layer_curve_zeroshot.png").exists()
    assert (out / "scalar_mix_weights.png").exists()
    assert (out / "diagnostics.png").exists()

    with (out / "results.json").open(encoding="utf-8") as fh:
        blob = json.load(fh)
    assert blob["results"] and blob["best"]
    # the saved config must be enough to reproduce the run
    saved = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert saved["encoder"]["model_name"] == "synthetic"
    assert saved["seeds"] == [0, 1]


def test_pipeline_reuses_the_feature_cache(tmp_path):
    cfg = _fast_config(tmp_path)
    run_experiment(cfg, make_figures=False, verbose=False)
    cached = sorted(p.name for p in Path(cfg.cache_dir).glob("*.npz"))
    assert cached  # 2 languages x 3 splits
    run_experiment(cfg, make_figures=False, verbose=False)
    assert sorted(p.name for p in Path(cfg.cache_dir).glob("*.npz")) == cached


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_overrides_are_applied():
    args = build_parser().parse_args(
        ["--model", "xlm-roberta-large", "--languages", "eng", "amh", "--seeds", "7", "--probe", "mlp"]
    )
    cfg = apply_overrides(ExperimentConfig(), args)
    assert cfg.encoder.model_name == "xlm-roberta-large"
    assert cfg.data.languages == ["eng", "amh"]
    assert cfg.seeds == [7]
    assert cfg.probe.kind == "mlp"


def test_cli_rejects_a_source_language_outside_the_language_list(tmp_path, capsys):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        yaml.safe_dump({"data": {"source": "synthetic", "languages": ["eng"], "source_languages": ["zzz"]}}),
        encoding="utf-8",
    )
    assert main(["--config", str(config_path)]) == 2
    assert "not in data.languages" in capsys.readouterr().err


def test_cli_runs_the_smoke_config(tmp_path):
    exit_code = main(
        [
            "--config", "configs/smoke.yaml",
            "--output-dir", str(tmp_path / "out"),
            "--cache-dir", str(tmp_path / "cache"),
            "--seeds", "0",
            "--max-train", "60",
            "--no-figures",
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "out" / "SUMMARY.md").exists()


# --------------------------------------------------------------------------
# config inheritance and multi-model configs
# --------------------------------------------------------------------------


def test_extends_merges_sections_from_the_parent(tmp_path):
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "seeds": [1, 2],
                "data": {"source": "synthetic", "languages": ["eng", "amh"], "source_languages": ["eng"]},
                "encoder": {"model_name": "xlm-roberta-base", "pooling": "mean", "max_length": 128},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        yaml.safe_dump(
            {"extends": "base.yaml", "name": "child", "encoder": {"model_name": "Qwen/Qwen3-0.6B-Base",
                                                                  "pooling": "last_token"}}),
        encoding="utf-8",
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.name == "child"
    assert cfg.encoder.model_name == "Qwen/Qwen3-0.6B-Base"
    assert cfg.encoder.pooling == "last_token"
    # untouched keys survive from the parent, at both levels
    assert cfg.encoder.max_length == 128
    assert cfg.seeds == [1, 2]
    assert cfg.data.languages == ["eng", "amh"]


def test_extends_rejects_a_cycle(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"extends": "b.yaml"}), encoding="utf-8")
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"extends": "a.yaml"}), encoding="utf-8")
    with pytest.raises(ValueError, match="circular 'extends'"):
        load_config(tmp_path / "a.yaml")


def test_extends_reports_a_missing_parent(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"extends": "nope.yaml"}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_config(tmp_path / "a.yaml")


def test_shipped_model_configs_parse_and_share_one_data_block():
    paths = sorted(p for p in Path("configs/models").glob("*.yaml") if not p.name.startswith("_"))
    assert paths, "no model configs found"
    configs = {p.stem: load_config(p) for p in paths}
    for name, cfg in configs.items():
        assert cfg.data.languages == configs["xlmr"].data.languages, name
        assert cfg.data.source_languages == configs["xlmr"].data.source_languages, name
        assert cfg.seeds == configs["xlmr"].seeds, name
        # each model must write somewhere different, or runs overwrite each other
        assert cfg.output_dir != configs["xlmr"].output_dir or name == "xlmr", name
    assert configs["qwen"].encoder.pooling == "last_token"
    assert configs["xlmr"].encoder.pooling == "mean"


def test_model_configs_leave_layer_windows_depth_relative():
    """Hard-coded windows would be wrong for models of differing depth."""

    for path in sorted(Path("configs/models").glob("*.yaml")):
        cfg = load_config(path)
        assert not cfg.combinations.named_windows, f"{path.name} hard-codes named_windows"
        assert not cfg.combinations.concat_groups, f"{path.name} hard-codes concat_groups"
