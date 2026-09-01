# -*- coding: utf-8 -*-
"""Light-weight classifiers trained on frozen representations.

Probes are deliberately weak.  A strong classifier can recover the emotion
signal from almost any layer given enough capacity, which would flatten
exactly the differences we are trying to measure; a linear probe reports how
*linearly accessible* the signal already is in a layer.  The MLP probe is
provided as a robustness check, not as the headline.

Three implementations share one interface:

:class:`LogisticProbe`
    One-vs-rest (multi-label) or multinomial (single-label) logistic
    regression, with the regularisation strength picked on dev.
:class:`MLPProbe`
    One hidden layer, trained with Adam in torch.
:class:`ScalarMixProbe`
    A linear classifier fed a softmax-weighted mixture of layers, with the
    mixture weights learned jointly.  Its learned weights are read out and
    reported: they are the study's answer to "which layers does the task
    want?", arrived at without a grid search over combinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from layerprobe.config import ProbeConfig, ScalarMixConfig
from layerprobe.metrics import score, tune_threshold


@dataclass
class ProbeOutcome:
    """Everything one probe fit produced."""

    dev: Dict[str, float]
    test: Dict[str, float]
    #: Softmax layer weights, present only for the scalar mix.
    layer_weights: Optional[List[float]] = None
    #: Hyper-parameters chosen on dev (C, threshold, ...).
    chosen: Dict[str, float] = field(default_factory=dict)


class _Standardizer:
    """StandardScaler that is a no-op when standardisation is switched off."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.scaler = StandardScaler() if enabled else None

    def fit(self, X: np.ndarray) -> "_Standardizer":
        if self.scaler is not None:
            self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return X
        return self.scaler.transform(X)


def _fit_logreg_multilabel(X: np.ndarray, Y: np.ndarray, C: float, max_iter: int) -> List[Optional[LogisticRegression]]:
    """One binary classifier per emotion; ``None`` where a label is constant."""

    models: List[Optional[LogisticRegression]] = []
    for j in range(Y.shape[1]):
        y = Y[:, j].astype(int)
        if len(np.unique(y)) < 2:
            # This emotion never occurs (or always does) in this training set;
            # a degenerate column, so predict its constant rate.
            models.append(None)
            continue
        clf = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")
        clf.fit(X, y)
        models.append(clf)
    return models


def _predict_logreg_multilabel(
    models: Sequence[Optional[LogisticRegression]], X: np.ndarray, constants: Sequence[float]
) -> np.ndarray:
    out = np.zeros((X.shape[0], len(models)), dtype=np.float64)
    for j, clf in enumerate(models):
        if clf is None:
            out[:, j] = constants[j]
        else:
            out[:, j] = clf.predict_proba(X)[:, 1]
    return out


class LogisticProbe:
    """Linear probe with a dev-set sweep over ``C`` (and the threshold)."""

    def __init__(self, cfg: ProbeConfig, task: str, emotions: Sequence[str]):
        self.cfg = cfg
        self.task = task
        self.emotions = list(emotions)

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ProbeOutcome:
        scaler = _Standardizer(self.cfg.standardize).fit(X_train)
        Xtr, Xdev, Xte = (scaler.transform(X) for X in (X_train, X_dev, X_test))

        best = None
        for C in self.cfg.C:
            if self.task == "multilabel":
                constants = y_train.mean(axis=0)
                models = _fit_logreg_multilabel(Xtr, y_train, C, self.cfg.max_iter)
                dev_prob = _predict_logreg_multilabel(models, Xdev, constants)
                threshold = (
                    self.cfg.threshold
                    if self.cfg.threshold is not None
                    else tune_threshold(y_dev, dev_prob)
                )
                dev_scores = score(y_dev, dev_prob, self.task, threshold, self.emotions)
                state = (models, constants, threshold)
            else:
                clf = LogisticRegression(C=C, max_iter=self.cfg.max_iter, class_weight="balanced")
                clf.fit(Xtr, y_train)
                dev_prob = self._dense_proba(clf, Xdev)
                threshold = 0.5
                dev_scores = score(y_dev, dev_prob, self.task, threshold, self.emotions)
                state = (clf, None, threshold)
            if best is None or dev_scores["macro_f1"] > best[0]["macro_f1"]:
                best = (dev_scores, state, C)

        assert best is not None
        dev_scores, state, C = best
        if self.task == "multilabel":
            models, constants, threshold = state
            test_prob = _predict_logreg_multilabel(models, Xte, constants)
        else:
            clf, _, threshold = state
            test_prob = self._dense_proba(clf, Xte)
        test_scores = score(y_test, test_prob, self.task, threshold, self.emotions)
        return ProbeOutcome(dev_scores, test_scores, None, {"C": float(C), "threshold": float(threshold)})

    def _dense_proba(self, clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
        """``predict_proba`` widened to the full emotion inventory.

        scikit-learn only emits columns for classes it saw in training; a
        low-resource split can easily miss one, and the metric functions
        expect a column per emotion.
        """

        proba = clf.predict_proba(X)
        out = np.zeros((X.shape[0], len(self.emotions)), dtype=np.float64)
        for col, cls in enumerate(clf.classes_):
            out[:, int(cls)] = proba[:, col]
        return out


# --------------------------------------------------------------------------
# torch probes
# --------------------------------------------------------------------------


def _torch_setup(seed: int):
    import torch

    torch.manual_seed(seed)
    return torch


class _TorchTrainer:
    """Shared Adam training loop for the two torch probes."""

    def __init__(self, task: str, emotions: Sequence[str], seed: int):
        self.task = task
        self.emotions = list(emotions)
        self.seed = seed
        self.torch = _torch_setup(seed)

    def loss_fn(self, logits, targets):
        torch = self.torch
        if self.task == "multilabel":
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        return torch.nn.functional.cross_entropy(logits, targets)

    def to_targets(self, y: np.ndarray):
        torch = self.torch
        if self.task == "multilabel":
            return torch.as_tensor(y, dtype=torch.float32)
        return torch.as_tensor(y, dtype=torch.long)

    def activate(self, logits):
        torch = self.torch
        if self.task == "multilabel":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def train(self, module, inputs, targets, epochs: int, lr: float, weight_decay: float, batch_size: int):
        torch = self.torch
        optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
        n = targets.shape[0]
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(epochs):
            order = torch.randperm(n, generator=generator)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad()
                loss = self.loss_fn(module(inputs[..., idx, :]), targets[idx])
                loss.backward()
                optimizer.step()
        return module

    def predict(self, module, inputs) -> np.ndarray:
        torch = self.torch
        module.eval()
        with torch.no_grad():
            out = self.activate(module(inputs)).cpu().numpy()
        module.train()
        return out


class MLPProbe:
    """One-hidden-layer probe, as a check that findings are not linear-only."""

    def __init__(self, cfg: ProbeConfig, task: str, emotions: Sequence[str], seed: int = 0):
        self.cfg = cfg
        self.task = task
        self.emotions = list(emotions)
        self.seed = seed

    def run(self, X_train, y_train, X_dev, y_dev, X_test, y_test) -> ProbeOutcome:
        trainer = _TorchTrainer(self.task, self.emotions, self.seed)
        torch = trainer.torch

        scaler = _Standardizer(self.cfg.standardize).fit(X_train)
        Xtr, Xdev, Xte = (
            torch.as_tensor(scaler.transform(X), dtype=torch.float32) for X in (X_train, X_dev, X_test)
        )
        n_out = len(self.emotions)
        module = torch.nn.Sequential(
            torch.nn.Linear(Xtr.shape[1], self.cfg.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.cfg.hidden_size, n_out),
        )
        trainer.train(
            module,
            Xtr,
            trainer.to_targets(y_train),
            self.cfg.epochs,
            self.cfg.learning_rate,
            self.cfg.weight_decay,
            batch_size=64,
        )
        dev_prob = trainer.predict(module, Xdev)
        if self.task != "multilabel":
            threshold = 0.5
        elif self.cfg.threshold is not None:
            threshold = float(self.cfg.threshold)
        else:
            threshold = tune_threshold(y_dev, dev_prob)
        dev_scores = score(y_dev, dev_prob, self.task, threshold, self.emotions)
        test_scores = score(y_test, trainer.predict(module, Xte), self.task, threshold, self.emotions)
        return ProbeOutcome(dev_scores, test_scores, None, {"threshold": threshold})


def _build_scalar_mix(torch, n_layers: int, hidden_size: int, n_out: int, cfg: ScalarMixConfig):
    """ELMo-style mixture ``gamma * sum_l softmax(w)_l * norm(h_l)``, plus a head.

    Defined inside a function because ``torch`` is imported lazily -- the
    scikit-learn probes must stay usable without it.
    """

    class ScalarMix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mix_weights = torch.nn.Parameter(torch.zeros(n_layers))
            self.gamma = torch.nn.Parameter(torch.ones(1))
            self.norm = torch.nn.LayerNorm(hidden_size) if cfg.layer_norm else torch.nn.Identity()
            self.head = torch.nn.Linear(hidden_size, n_out)

        def softmax_weights(self):
            return torch.softmax(self.mix_weights / cfg.temperature, dim=0)

        def forward(self, x):
            """``x`` is ``(n_layers, batch, hidden)``."""

            mixed = (self.softmax_weights().view(-1, 1, 1) * self.norm(x)).sum(dim=0)
            if cfg.learn_gamma:
                mixed = self.gamma * mixed
            return self.head(mixed)

        def layer_weights(self) -> List[float]:
            with torch.no_grad():
                return [float(v) for v in self.softmax_weights().cpu().numpy()]

    return ScalarMix()


class ScalarMixProbe:
    """Linear probe over a *learned* weighted combination of layers."""

    def __init__(
        self,
        cfg: ProbeConfig,
        mix_cfg: ScalarMixConfig,
        task: str,
        emotions: Sequence[str],
        seed: int = 0,
    ):
        self.cfg = cfg
        self.mix_cfg = mix_cfg
        self.task = task
        self.emotions = list(emotions)
        self.seed = seed

    def run(self, T_train, y_train, T_dev, y_dev, T_test, y_test) -> ProbeOutcome:
        """Inputs are ``(n_layers, n, d)`` tensors, not flat matrices."""

        trainer = _TorchTrainer(self.task, self.emotions, self.seed)
        torch = trainer.torch

        # Standardise per layer using train statistics, so that a layer with a
        # large activation norm cannot win the mixture on scale alone.
        mean = T_train.mean(axis=1, keepdims=True)
        std = T_train.std(axis=1, keepdims=True) + 1e-6
        tensors = [
            torch.as_tensor((T - mean) / std, dtype=torch.float32) for T in (T_train, T_dev, T_test)
        ]
        Ttr, Tdev, Tte = tensors

        mix = _build_scalar_mix(torch, Ttr.shape[0], Ttr.shape[2], len(self.emotions), self.mix_cfg)
        trainer.train(
            mix,
            Ttr,
            trainer.to_targets(y_train),
            self.mix_cfg.epochs,
            self.mix_cfg.learning_rate,
            self.mix_cfg.weight_decay,
            self.mix_cfg.batch_size,
        )
        dev_prob = trainer.predict(mix, Tdev)
        if self.task == "multilabel":
            threshold = (
                float(self.cfg.threshold) if self.cfg.threshold is not None else tune_threshold(y_dev, dev_prob)
            )
        else:
            threshold = 0.5
        dev_scores = score(y_dev, dev_prob, self.task, threshold, self.emotions)
        test_scores = score(y_test, trainer.predict(mix, Tte), self.task, threshold, self.emotions)
        return ProbeOutcome(dev_scores, test_scores, mix.layer_weights(), {"threshold": threshold})


def build_probe(cfg: ProbeConfig, task: str, emotions: Sequence[str], seed: int = 0):
    """Instantiate the flat-feature probe named by the config."""

    if cfg.kind == "logreg":
        return LogisticProbe(cfg, task, emotions)
    if cfg.kind == "mlp":
        return MLPProbe(cfg, task, emotions, seed)
    raise ValueError(f"unknown probe.kind {cfg.kind!r}; expected 'logreg' or 'mlp'")
