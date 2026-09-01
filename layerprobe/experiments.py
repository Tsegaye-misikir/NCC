# -*- coding: utf-8 -*-
"""The three experiments, and the loop that runs them over seeds.

Every experiment answers the same question -- which layer or combination of
layers gives the best emotion representation? -- under a different training
regime:

``monolingual``
    Train and test in the same language.  Establishes the in-language
    ceiling that the cross-lingual numbers are measured against, and shows
    whether the best layer differs between high- and low-resource languages.

``zeroshot``
    Train on the source language(s), test on a target language never seen in
    training.  This is where representation choice pays off or hurts: the
    ``transfer_gap`` column (in-language minus zero-shot macro-F1) is the
    negative-transfer measure, and ``above_majority`` says whether the probe
    transferred anything at all.

``multilingual``
    Train once on all languages pooled, test per language.  Checks whether a
    good layer choice also reduces the interference between languages that
    joint training introduces.

Only the probe is retrained per seed; features are extracted once and
cached, because a frozen encoder is deterministic.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from layerprobe.combinations import LayerCombination, build_combinations, layer_tensor, materialize
from layerprobe.config import ExperimentConfig
from layerprobe.data import Corpus, EmotionSplit, concatenate
from layerprobe.features import FeatureStore, LayerFeatures, stack_features
from layerprobe.metrics import majority_baseline, summarise, transfer_gap
from layerprobe.probes import ScalarMixProbe, build_probe

#: Metrics carried through to the aggregated tables.
REPORTED = ("macro_f1", "micro_f1")


def _resample_train(
    train: tuple[LayerFeatures, np.ndarray], fraction: Optional[float], seed: int
) -> tuple[LayerFeatures, np.ndarray]:
    """Draw this seed's training subset, so seeds differ for a linear probe.

    Drawn once per (setting, seed) and shared by every combination, so the
    comparison between layers stays paired: they all see the same sentences.
    """

    features, labels = train
    n = labels.shape[0]
    if fraction is None or fraction >= 1.0 or n < 4:
        return train
    size = max(2, int(round(n * fraction)))
    if size >= n:
        return train
    idx = np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))
    return features.subset(idx), labels[idx]


def _design(features: LayerFeatures, combo: LayerCombination) -> np.ndarray:
    """The array a probe consumes for this combination."""

    if combo.kind == "scalarmix":
        return layer_tensor(features, combo)
    return materialize(features, combo)


def _fit_one(
    cfg: ExperimentConfig,
    combo: LayerCombination,
    train: tuple[LayerFeatures, np.ndarray],
    dev: tuple[LayerFeatures, np.ndarray],
    emotions: Sequence[str],
    task: str,
    seed: int,
):
    """Fit one probe for one combination, at one seed."""

    if combo.kind == "scalarmix":
        probe = ScalarMixProbe(cfg.probe, cfg.scalar_mix, task, emotions, seed)
    else:
        probe = build_probe(cfg.probe, task, emotions, seed)
    return probe.fit(_design(train[0], combo), train[1], _design(dev[0], combo), dev[1])


def _aggregate(
    per_seed: Dict[str, List[dict]],
    combos: Sequence[LayerCombination],
    context: Dict[str, object],
) -> List[dict]:
    """Collapse per-seed outcomes into one row per combination."""

    by_name = {c.name: c for c in combos}
    rows: List[dict] = []
    for name, outcomes in per_seed.items():
        combo = by_name[name]
        row: Dict[str, object] = dict(context)
        row.update(
            {
                "combination": name,
                "kind": combo.kind,
                "layers": list(combo.layers),
                "n_layers": len(combo.layers),
            }
        )
        row.update(summarise([o["test"] for o in outcomes], REPORTED))
        row["macro_f1_per_seed"] = [float(o["test"]["macro_f1"]) for o in outcomes]
        row["dev_macro_f1_mean"] = float(np.mean([o["dev"]["macro_f1"] for o in outcomes]))
        weights = [o["layer_weights"] for o in outcomes if o.get("layer_weights")]
        if weights:
            row["layer_weights_mean"] = [float(v) for v in np.mean(weights, axis=0)]
        rows.append(row)
    return rows


def _probe_sweep_multi(
    cfg: ExperimentConfig,
    combos: Sequence[LayerCombination],
    train: tuple[LayerFeatures, np.ndarray],
    dev: tuple[LayerFeatures, np.ndarray],
    tests: Dict[str, tuple[LayerFeatures, np.ndarray]],
    emotions: Sequence[str],
    task: str,
    contexts: Dict[str, Dict[str, object]],
    verbose: bool = True,
) -> List[dict]:
    """Fit each (combination, seed) once, then score it on every test set.

    The cross-lingual experiments train a single probe and evaluate it on
    many target languages.  Fitting is by far the dominant cost, so it is
    hoisted out of the loop over targets; the results are identical to
    refitting per target, because the fit depends only on train and dev.
    """

    per_seed: Dict[str, Dict[str, List[dict]]] = {
        language: {c.name: [] for c in combos} for language in tests
    }
    for seed in cfg.seeds:
        seed_train = _resample_train(train, cfg.probe.train_subsample, seed)
        for combo in combos:
            probe = _fit_one(cfg, combo, seed_train, dev, emotions, task, seed)
            for language, test in tests.items():
                outcome = probe.outcome(_design(test[0], combo), test[1])
                per_seed[language][combo.name].append(
                    {"dev": outcome.dev, "test": outcome.test, "layer_weights": outcome.layer_weights}
                )

    rows: List[dict] = []
    for language, by_combo in per_seed.items():
        if verbose:
            best = max(
                by_combo.items(), key=lambda kv: np.mean([o["test"]["macro_f1"] for o in kv[1]])
            )
            best_score = float(np.mean([o["test"]["macro_f1"] for o in best[1]]))
            print(f"    {language}: best {best[0]} (macro-F1 {best_score:.4f})", flush=True)
        rows.extend(_aggregate(by_combo, combos, contexts[language]))
    return rows


def _probe_sweep(
    cfg: ExperimentConfig,
    combos: Sequence[LayerCombination],
    train: tuple[LayerFeatures, np.ndarray],
    dev: tuple[LayerFeatures, np.ndarray],
    test: tuple[LayerFeatures, np.ndarray],
    emotions: Sequence[str],
    task: str,
    context: Dict[str, object],
    verbose: bool = True,
) -> List[dict]:
    """Run every combination over every seed for one train/test setting."""

    language = str(context.get("eval_language", "test"))
    return _probe_sweep_multi(
        cfg, combos, train, dev, {language: test}, emotions, task, {language: context}, verbose
    )


def run_monolingual(
    cfg: ExperimentConfig,
    corpus: Corpus,
    store: FeatureStore,
    combos: Sequence[LayerCombination],
    verbose: bool = True,
) -> List[dict]:
    """Train and test within each language."""

    rows: List[dict] = []
    for language in cfg.data.languages:
        if verbose:
            print(f"  [monolingual] {language}", flush=True)
        splits = corpus[language]
        feats = store[language]
        rows.extend(
            _probe_sweep(
                cfg,
                combos,
                (feats["train"], splits["train"].labels),
                (feats["dev"], splits["dev"].labels),
                (feats["test"], splits["test"].labels),
                splits["train"].emotions,
                splits["train"].task,
                {
                    "experiment": "monolingual",
                    "train_languages": [language],
                    "eval_language": language,
                    "n_train": len(splits["train"]),
                    "majority_macro_f1": majority_baseline(
                        splits["train"].labels,
                        splits["test"].labels,
                        splits["train"].task,
                        splits["train"].emotions,
                    )["macro_f1"],
                },
                verbose,
            )
        )
    return rows


def _pooled_setting(
    corpus: Corpus, store: FeatureStore, languages: Sequence[str], split: str
) -> tuple[LayerFeatures, np.ndarray]:
    """Pool several languages' features and labels for one split."""

    feats = stack_features([store[lg][split] for lg in languages])
    labels = concatenate([corpus[lg][split] for lg in languages]).labels
    return feats, labels


def run_zeroshot(
    cfg: ExperimentConfig,
    corpus: Corpus,
    store: FeatureStore,
    combos: Sequence[LayerCombination],
    verbose: bool = True,
) -> List[dict]:
    """Train on the source language(s), test on each held-out target."""

    sources = [lg for lg in cfg.data.source_languages if lg in corpus]
    targets = [lg for lg in cfg.data.resolved_target_languages() if lg in corpus]
    if not sources or not targets:
        return []

    train = _pooled_setting(corpus, store, sources, "train")
    # Model selection stays in the source language: peeking at target dev
    # would make the setting few-shot rather than zero-shot.
    dev = _pooled_setting(corpus, store, sources, "dev")
    reference = corpus[sources[0]]["train"]

    if verbose:
        print(f"  [zero-shot] {'+'.join(sources)} -> {', '.join(targets)}", flush=True)
    tests, contexts = {}, {}
    for target in targets:
        test_split = corpus[target]["test"]
        tests[target] = (store[target]["test"], test_split.labels)
        contexts[target] = {
            "experiment": "zeroshot",
            "train_languages": list(sources),
            "eval_language": target,
            "n_train": train[1].shape[0],
            "majority_macro_f1": majority_baseline(
                train[1], test_split.labels, reference.task, reference.emotions
            )["macro_f1"],
        }
    return _probe_sweep_multi(
        cfg, combos, train, dev, tests, reference.emotions, reference.task, contexts, verbose
    )


def run_multilingual(
    cfg: ExperimentConfig,
    corpus: Corpus,
    store: FeatureStore,
    combos: Sequence[LayerCombination],
    verbose: bool = True,
) -> List[dict]:
    """Train once on all languages pooled, then test language by language."""

    languages = [lg for lg in cfg.data.languages if lg in corpus]
    if len(languages) < 2:
        return []

    train = _pooled_setting(corpus, store, languages, "train")
    dev = _pooled_setting(corpus, store, languages, "dev")
    reference = corpus[languages[0]]["train"]

    if verbose:
        print(f"  [multilingual] eval on {', '.join(languages)}", flush=True)
    tests, contexts = {}, {}
    for language in languages:
        test_split = corpus[language]["test"]
        tests[language] = (store[language]["test"], test_split.labels)
        contexts[language] = {
            "experiment": "multilingual",
            "train_languages": list(languages),
            "eval_language": language,
            "n_train": train[1].shape[0],
            "majority_macro_f1": majority_baseline(
                train[1], test_split.labels, reference.task, reference.emotions
            )["macro_f1"],
        }
    return _probe_sweep_multi(
        cfg, combos, train, dev, tests, reference.emotions, reference.task, contexts, verbose
    )


def add_transfer_columns(rows: Sequence[dict]) -> List[dict]:
    """Annotate cross-lingual rows with their gap to the in-language probe.

    ``transfer_gap`` is positive when training out-of-language costs
    performance, which is the negative-transfer case; ``above_majority`` is
    the sanity check that any transfer happened at all.
    """

    rows = [dict(r) for r in rows]
    in_language = {
        (r["eval_language"], r["combination"]): r.get("macro_f1_mean")
        for r in rows
        if r["experiment"] == "monolingual"
    }
    for row in rows:
        macro = row.get("macro_f1_mean")
        if macro is None:
            continue
        if "majority_macro_f1" in row:
            row["above_majority"] = float(macro - row["majority_macro_f1"])
        if row["experiment"] == "monolingual":
            continue
        reference = in_language.get((row["eval_language"], row["combination"]))
        if reference is not None:
            row["transfer_gap"] = transfer_gap(macro, reference)
            row["in_language_macro_f1"] = float(reference)
    return rows


def best_per_setting(rows: Sequence[dict], metric: str = "macro_f1_mean") -> List[dict]:
    """The winning combination for each (experiment, language) pair.

    Also records what the default choice -- the final layer -- would have
    scored, since ``gain_over_last`` is the number the study is ultimately
    reporting.
    """

    grouped: Dict[tuple, List[dict]] = {}
    for row in rows:
        grouped.setdefault((row["experiment"], row["eval_language"]), []).append(row)

    out: List[dict] = []
    for (experiment, language), group in grouped.items():
        scored = [r for r in group if r.get(metric) is not None]
        if not scored:
            continue
        best = max(scored, key=lambda r: r[metric])
        last = next((r for r in group if r["combination"] == "last"), None)
        entry = {
            "experiment": experiment,
            "eval_language": language,
            "best_combination": best["combination"],
            "best_kind": best["kind"],
            "best_layers": best["layers"],
            "best_macro_f1": float(best[metric]),
            "best_macro_f1_std": float(best.get("macro_f1_std", 0.0)),
        }
        if last is not None and last.get(metric) is not None:
            entry["last_layer_macro_f1"] = float(last[metric])
            entry["gain_over_last"] = float(best[metric] - last[metric])
        if "transfer_gap" in best:
            entry["best_transfer_gap"] = float(best["transfer_gap"])
        if last is not None and "transfer_gap" in last:
            entry["last_layer_transfer_gap"] = float(last["transfer_gap"])
        out.append(entry)
    return sorted(out, key=lambda r: (r["experiment"], r["eval_language"]))


def run_all(
    cfg: ExperimentConfig,
    corpus: Corpus,
    store: FeatureStore,
    layer_ids: Sequence[int],
    verbose: bool = True,
) -> Dict[str, object]:
    """Run every enabled experiment and return the assembled result tables."""

    combos = build_combinations(cfg.combinations, layer_ids)
    if verbose:
        print(f"comparing {len(combos)} layer combinations over {len(cfg.seeds)} seed(s)", flush=True)

    rows: List[dict] = []
    started = time.time()
    if cfg.run_monolingual:
        rows.extend(run_monolingual(cfg, corpus, store, combos, verbose))
    if cfg.run_zeroshot:
        rows.extend(run_zeroshot(cfg, corpus, store, combos, verbose))
    if cfg.run_multilingual:
        rows.extend(run_multilingual(cfg, corpus, store, combos, verbose))
    rows = add_transfer_columns(rows)

    return {
        "results": rows,
        "best": best_per_setting(rows),
        "combinations": [
            {"name": c.name, "kind": c.kind, "layers": list(c.layers)} for c in combos
        ],
        "runtime_seconds": round(time.time() - started, 2),
    }
