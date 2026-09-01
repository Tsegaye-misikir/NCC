# -*- coding: utf-8 -*-
"""Data loading: label parsing, split derivation, local files."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from layerprobe.config import DataConfig
from layerprobe.data import concatenate, describe, load_corpus


def test_synthetic_corpus_has_all_splits_and_shapes():
    cfg = DataConfig(source="synthetic", languages=["eng", "amh"], synthetic_size=120)
    corpus = load_corpus(cfg, seed=0)

    assert set(corpus) == {"eng", "amh"}
    for splits in corpus.values():
        assert set(splits) == {"train", "dev", "test"}
        total = sum(len(s) for s in splits.values())
        assert total == 120
        for split in splits.values():
            assert split.labels.shape == (len(split), len(cfg.emotions))
            # every example carries at least one emotion
            assert split.labels.sum(axis=1).min() >= 1


def test_splits_are_disjoint():
    cfg = DataConfig(source="synthetic", languages=["eng"], synthetic_size=100)
    splits = load_corpus(cfg, seed=0)["eng"]
    texts = [set(s.texts) for s in splits.values()]
    assert not (texts[0] & texts[1])
    assert not (texts[0] & texts[2])
    assert not (texts[1] & texts[2])


def test_loading_is_deterministic_for_a_seed():
    cfg = DataConfig(source="synthetic", languages=["eng"], synthetic_size=80)
    first = load_corpus(cfg, seed=7)["eng"]["train"]
    second = load_corpus(cfg, seed=7)["eng"]["train"]
    assert first.texts == second.texts
    np.testing.assert_array_equal(first.labels, second.labels)


def test_stable_seed_survives_hash_randomisation():
    """Python salts hash() per process; seeds must not depend on it.

    Seeding from ``hash("amh")`` silently produced a different corpus on
    every run, which broke both reproducibility and the feature cache. This
    checks the property in a *separate interpreter*, since a same-process
    check cannot see the salt change.
    """

    script = (
        "from layerprobe.data import stable_seed, load_corpus;"
        "from layerprobe.config import DataConfig;"
        "c=DataConfig(source='synthetic', languages=['amh'], synthetic_size=20);"
        "print(stable_seed('amh', 7), load_corpus(c, seed=3)['amh']['train'].texts[0])"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent.parent))
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
        ).stdout
        for _ in range(3)
    }
    assert len(runs) == 1, f"synthetic data differs between processes: {runs}"


def test_emotion_markers_are_shared_across_languages():
    """Cross-lingual transfer must be possible in principle on the toy data."""

    cfg = DataConfig(source="synthetic", languages=["eng", "amh"], synthetic_size=60)
    corpus = load_corpus(cfg, seed=0)

    def markers(language):
        return {
            token
            for text in corpus[language]["train"].texts
            for token in text.split()
            if token.startswith("e")
        }

    assert markers("eng") & markers("amh")
    # ... while every non-emotion token stays language-specific, so language
    # identity is recoverable and the language probe has something to find.
    def filler(language):
        return {
            token
            for text in corpus[language]["train"].texts
            for token in text.split()
            if not token.startswith("e")
        }

    assert not (filler("eng") & filler("amh"))
    # Each sentence carries its language tag, which is what gives the toy
    # encoder a consistent per-language direction to build layers from.
    assert all(text.startswith("Lamh ") for text in corpus["amh"]["train"].texts)


def test_local_multilabel_csv(tmp_path):
    path = tmp_path / "eng_train.csv"
    path.write_text(
        "text,anger,joy\n" + "\n".join(f"sentence {i},{i % 2},{(i + 1) % 2}" for i in range(40)),
        encoding="utf-8",
    )
    cfg = DataConfig(
        source="local", local_dir=str(tmp_path), languages=["eng"], emotions=["anger", "joy"]
    )
    corpus = load_corpus(cfg, seed=0)
    assert sum(len(s) for s in corpus["eng"].values()) == 40
    assert corpus["eng"]["train"].labels.shape[1] == 2


def test_local_singlelabel_csv(tmp_path):
    labels = ["anger", "joy"]
    path = tmp_path / "eng_train.tsv"
    path.write_text(
        "text\tlabel\n" + "\n".join(f"sentence {i}\t{labels[i % 2]}" for i in range(40)),
        encoding="utf-8",
    )
    cfg = DataConfig(
        source="local",
        local_dir=str(tmp_path),
        local_suffix="tsv",
        languages=["eng"],
        emotions=labels,
        task="singlelabel",
    )
    corpus = load_corpus(cfg, seed=0)
    train = corpus["eng"]["train"]
    assert train.labels.ndim == 1
    assert set(np.unique(train.labels)) <= {0, 1}


def test_unknown_single_label_is_rejected(tmp_path):
    path = tmp_path / "eng_train.csv"
    path.write_text("text,label\nhello,ecstatic\n" * 1, encoding="utf-8")
    cfg = DataConfig(
        source="local",
        local_dir=str(tmp_path),
        languages=["eng"],
        emotions=["anger", "joy"],
        task="singlelabel",
    )
    with pytest.raises(ValueError, match="ecstatic"):
        load_corpus(cfg, seed=0)


def test_missing_local_files_raise(tmp_path):
    cfg = DataConfig(source="local", local_dir=str(tmp_path), languages=["eng"])
    with pytest.raises(FileNotFoundError):
        load_corpus(cfg, seed=0)


def test_max_train_per_language_caps_the_training_set():
    cfg = DataConfig(
        source="synthetic", languages=["eng"], synthetic_size=200, max_train_per_language=50
    )
    assert len(load_corpus(cfg, seed=0)["eng"]["train"]) == 50


def test_concatenate_pools_languages():
    cfg = DataConfig(source="synthetic", languages=["eng", "amh"], synthetic_size=60)
    corpus = load_corpus(cfg, seed=0)
    pooled = concatenate([corpus["eng"]["train"], corpus["amh"]["train"]])
    assert len(pooled) == len(corpus["eng"]["train"]) + len(corpus["amh"]["train"])
    assert pooled.labels.shape[0] == len(pooled)


def test_describe_reports_sizes_and_rates():
    cfg = DataConfig(source="synthetic", languages=["eng"], synthetic_size=60)
    row = describe(load_corpus(cfg, seed=0))[0]
    assert row["language"] == "eng"
    assert row["n_train"] > 0
    assert 0.0 <= row["pos_rate_anger"] <= 1.0
