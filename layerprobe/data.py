# -*- coding: utf-8 -*-
"""Loading multilingual emotion data.

Three sources are supported (see :class:`layerprobe.config.DataConfig`):
a Hugging Face dataset with one config per language (how BRIGHTER /
SemEval-2025 Task 11 is distributed), local CSV/TSV files, and a
deterministic synthetic corpus used by the tests and the offline smoke run.

Whatever the source, the rest of the package sees the same object: an
:class:`EmotionSplit` holding raw texts plus a label matrix.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from layerprobe.config import DataConfig

SPLITS = ("train", "dev", "test")


@dataclass
class EmotionSplit:
    """One (language, split) slice of the corpus."""

    language: str
    split: str
    texts: List[str]
    #: ``(n, n_emotions)`` float32 0/1 matrix for multi-label, ``(n,)`` int64
    #: class indices for single-label.
    labels: np.ndarray
    emotions: List[str]
    task: str

    def __len__(self) -> int:
        return len(self.texts)

    def subsample(self, n: Optional[int], seed: int = 0) -> "EmotionSplit":
        """Take a random ``n``-example subset (a no-op when ``n`` is large)."""

        if n is None or n >= len(self.texts):
            return self
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self.texts), size=n, replace=False))
        return EmotionSplit(
            language=self.language,
            split=self.split,
            texts=[self.texts[i] for i in idx],
            labels=self.labels[idx],
            emotions=self.emotions,
            task=self.task,
        )


#: ``{language: {split: EmotionSplit}}``
Corpus = Dict[str, Dict[str, EmotionSplit]]


def _label_matrix(rows: Sequence[dict], emotions: Sequence[str], task: str) -> np.ndarray:
    """Turn raw records into a label array, tolerating the usual encodings."""

    if task == "multilabel":
        out = np.zeros((len(rows), len(emotions)), dtype=np.float32)
        for i, row in enumerate(rows):
            for j, emo in enumerate(emotions):
                if emo not in row:
                    continue
                value = row[emo]
                if value is None or value == "":
                    continue
                out[i, j] = 1.0 if float(value) > 0 else 0.0
        return out
    if task == "singlelabel":
        lookup = {emo: i for i, emo in enumerate(emotions)}
        out = np.zeros(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            raw = row.get("label", row.get("emotion"))
            if raw is None:
                raise KeyError(
                    "single-label data needs a 'label' or 'emotion' column; "
                    f"row {i} has keys {sorted(row)}"
                )
            if isinstance(raw, str):
                if raw not in lookup:
                    raise ValueError(f"label {raw!r} is not in emotions={list(emotions)}")
                out[i] = lookup[raw]
            else:
                out[i] = int(raw)
        return out
    raise ValueError(f"unknown task {task!r}; expected 'multilabel' or 'singlelabel'")


def _split_off(
    split: EmotionSplit, fraction: float, seed: int
) -> tuple[EmotionSplit, EmotionSplit]:
    """Carve ``fraction`` of ``split`` off into a second split."""

    n = len(split)
    n_held = max(1, int(round(n * fraction))) if n > 1 else 0
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    held_idx, keep_idx = np.sort(order[:n_held]), np.sort(order[n_held:])

    def _take(idx: np.ndarray, name: str) -> EmotionSplit:
        return EmotionSplit(
            language=split.language,
            split=name,
            texts=[split.texts[i] for i in idx],
            labels=split.labels[idx],
            emotions=split.emotions,
            task=split.task,
        )

    return _take(keep_idx, split.split), _take(held_idx, "held")


def _complete_splits(splits: Dict[str, EmotionSplit], cfg: DataConfig, seed: int) -> Dict[str, EmotionSplit]:
    """Fill in dev/test by carving them out of train when a corpus lacks them.

    Low-resource emotion corpora frequently ship train+dev but keep test
    labels back, or ship a single labelled file.  Rather than fail, we derive
    the missing splits deterministically so that every language in a run is
    evaluated the same way.
    """

    if "train" not in splits:
        raise ValueError("a train split is required")
    if "test" not in splits:
        train, held = _split_off(splits["train"], cfg.test_fraction, seed)
        held.split = "test"
        splits["train"], splits["test"] = train, held
    if "dev" not in splits:
        train, held = _split_off(splits["train"], cfg.dev_fraction, seed + 1)
        held.split = "dev"
        splits["train"], splits["dev"] = train, held
    return splits


# --------------------------------------------------------------------------
# Hugging Face datasets
# --------------------------------------------------------------------------


def _load_hf_language(cfg: DataConfig, language: str, seed: int) -> Dict[str, EmotionSplit]:
    from datasets import load_dataset  # imported lazily: only the hf path needs it

    if not cfg.hf_path:
        raise ValueError("data.hf_path must be set when data.source == 'hf'")
    name = cfg.hf_name_template.format(language=language)
    raw = load_dataset(cfg.hf_path, name)

    alias = {"train": "train", "validation": "dev", "dev": "dev", "test": "test"}
    splits: Dict[str, EmotionSplit] = {}
    for hf_split, our_split in alias.items():
        if hf_split not in raw:
            continue
        rows = [dict(r) for r in raw[hf_split]]
        if not rows:
            continue
        texts = [str(r[cfg.hf_text_column]) for r in rows]
        labels = _label_matrix(rows, cfg.emotions, cfg.task)
        # Corpora that withhold test labels arrive as an all-zero matrix; such
        # a split cannot be scored, so drop it and let _complete_splits carve
        # a scorable one out of train instead.
        if our_split != "train" and cfg.task == "multilabel" and not labels.any():
            continue
        splits[our_split] = EmotionSplit(language, our_split, texts, labels, list(cfg.emotions), cfg.task)
    return _complete_splits(splits, cfg, seed)


# --------------------------------------------------------------------------
# Local CSV / TSV
# --------------------------------------------------------------------------


def _read_delimited(path: Path) -> List[dict]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter=delimiter)]


def _load_local_language(cfg: DataConfig, language: str, seed: int) -> Dict[str, EmotionSplit]:
    if not cfg.local_dir:
        raise ValueError("data.local_dir must be set when data.source == 'local'")
    root = Path(cfg.local_dir)
    splits: Dict[str, EmotionSplit] = {}
    for split in SPLITS:
        path = root / f"{language}_{split}.{cfg.local_suffix}"
        if not path.exists():
            continue
        rows = _read_delimited(path)
        if not rows:
            continue
        text_col = cfg.hf_text_column if cfg.hf_text_column in rows[0] else "text"
        texts = [str(r[text_col]) for r in rows]
        labels = _label_matrix(rows, cfg.emotions, cfg.task)
        splits[split] = EmotionSplit(language, split, texts, labels, list(cfg.emotions), cfg.task)
    if not splits:
        raise FileNotFoundError(
            f"no files matching {root}/{language}_{{train,dev,test}}.{cfg.local_suffix}"
        )
    return _complete_splits(splits, cfg, seed)


# --------------------------------------------------------------------------
# Synthetic corpus (offline)
# --------------------------------------------------------------------------


def _load_synthetic_language(cfg: DataConfig, language: str, seed: int) -> Dict[str, EmotionSplit]:
    """A toy corpus with a shared emotion signal and language-specific noise.

    Each sentence is built from three kinds of token:

    ``e{k}``
        Emotion markers, *the same* across languages, so a cross-lingual
        probe can succeed in principle.
    ``L{language}``
        A language tag, constant within a language, giving the
        language-identification probe a consistent direction to find.
    ``w{...}``
        Filler drawn from a per-language vocabulary: disjoint across
        languages but carrying no systematic signal.

    Together these let the whole pipeline -- probes and both diagnostics --
    be exercised end to end without touching the network.
    """

    rng = np.random.default_rng(abs(hash((language, seed))) % (2**32))
    n_emotions = len(cfg.emotions)
    n = cfg.synthetic_size
    lang_offset = (abs(hash(language)) % 7 + 1) * 1000

    texts: List[str] = []
    if cfg.task == "multilabel":
        labels = np.zeros((n, n_emotions), dtype=np.float32)
    else:
        labels = np.zeros(n, dtype=np.int64)

    for i in range(n):
        if cfg.task == "multilabel":
            active = rng.random(n_emotions) < 0.25
            if not active.any():
                active[rng.integers(n_emotions)] = True
            labels[i] = active.astype(np.float32)
            active_ids = np.flatnonzero(active)
        else:
            cls = int(rng.integers(n_emotions))
            labels[i] = cls
            active_ids = np.array([cls])

        tokens = [f"w{lang_offset + int(rng.integers(cfg.synthetic_vocab))}" for _ in range(8)]
        for emo_id in active_ids:
            for _ in range(2):
                tokens.insert(int(rng.integers(len(tokens) + 1)), f"e{emo_id}")
        texts.append(" ".join([f"L{language}"] + tokens))

    full = EmotionSplit(language, "train", texts, labels, list(cfg.emotions), cfg.task)
    return _complete_splits({"train": full}, cfg, seed)


# --------------------------------------------------------------------------


_LOADERS = {
    "hf": _load_hf_language,
    "local": _load_local_language,
    "synthetic": _load_synthetic_language,
}


def load_corpus(cfg: DataConfig, seed: int = 0) -> Corpus:
    """Load every configured language, with train/dev/test for each."""

    if cfg.source not in _LOADERS:
        raise ValueError(f"unknown data.source {cfg.source!r}; expected one of {sorted(_LOADERS)}")
    loader = _LOADERS[cfg.source]

    corpus: Corpus = {}
    for language in cfg.languages:
        splits = loader(cfg, language, seed)
        splits["train"] = splits["train"].subsample(cfg.max_train_per_language, seed)
        for split in ("dev", "test"):
            splits[split] = splits[split].subsample(cfg.max_eval_per_language, seed)
        corpus[language] = splits
    return corpus


def concatenate(splits: Sequence[EmotionSplit], language: str = "multi") -> EmotionSplit:
    """Pool several languages into one split (used for joint training)."""

    if not splits:
        raise ValueError("nothing to concatenate")
    first = splits[0]
    return EmotionSplit(
        language=language,
        split=first.split,
        texts=[t for s in splits for t in s.texts],
        labels=np.concatenate([s.labels for s in splits], axis=0),
        emotions=first.emotions,
        task=first.task,
    )


def describe(corpus: Corpus) -> List[dict]:
    """Per-language size and label-density summary, for the run's report."""

    rows = []
    for language, splits in corpus.items():
        row = {"language": language}
        for split in SPLITS:
            row[f"n_{split}"] = len(splits[split])
        train = splits["train"]
        if train.task == "multilabel":
            row["labels_per_example"] = float(train.labels.sum(axis=1).mean())
            for j, emo in enumerate(train.emotions):
                row[f"pos_rate_{emo}"] = float(train.labels[:, j].mean())
        else:
            counts = np.bincount(train.labels, minlength=len(train.emotions))
            for j, emo in enumerate(train.emotions):
                row[f"pos_rate_{emo}"] = float(counts[j] / max(1, len(train)))
        rows.append(row)
    return rows
