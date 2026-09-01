# -*- coding: utf-8 -*-
"""Extracting one sentence vector per transformer layer.

The encoder is used strictly as a frozen feature extractor: a batch of texts
goes in, an array of shape ``(n_layers, n_examples, hidden_size)`` comes out,
where layer 0 is the embedding layer and layer ``L`` the final one.  Nothing
here is fine-tuned -- the whole point of the study is to compare the
representations the pretrained model already has.

Two implementations:

:class:`HFEncoder`
    The real thing, wrapping any Hugging Face encoder that can return hidden
    states (``xlm-roberta-base`` by default).

:class:`SyntheticEncoder`
    A deterministic stand-in that needs no downloads.  It fabricates a
    layer profile in which language identity dominates the bottom layers,
    is suppressed in the middle, and partly returns at the top.  That shape
    is a *simulation* chosen to exercise every code path offline -- it is
    not evidence about any real model.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional, Sequence

import numpy as np

from layerprobe.config import EncoderConfig


def _pool(states: "np.ndarray | object", mask, pooling: str):
    """Pool ``(batch, seq, hidden)`` token states down to ``(batch, hidden)``.

    Works on torch tensors; ``mask`` is the attention mask so that padding
    never contributes to a mean or a max.
    """

    import torch

    mask = mask.unsqueeze(-1).to(states.dtype)
    if pooling == "cls":
        return states[:, 0, :]
    if pooling == "mean":
        summed = (states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    if pooling == "max":
        neg_inf = torch.finfo(states.dtype).min
        return (states.masked_fill(mask == 0, neg_inf)).max(dim=1).values
    raise ValueError(f"unknown pooling {pooling!r}; expected 'mean', 'cls' or 'max'")


class BaseEncoder:
    """Interface shared by the real and the offline encoder."""

    hidden_size: int
    num_layers: int  # transformer blocks; total layer count is this + 1

    @property
    def layer_ids(self) -> List[int]:
        """All available layer indices, embeddings (0) through final."""

        return list(range(self.num_layers + 1))

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class HFEncoder(BaseEncoder):
    """Frozen Hugging Face encoder returning every hidden layer.

    ``model`` and ``tokenizer`` can be supplied directly, which is how the
    test-suite exercises this class without reaching the network.
    """

    def __init__(self, cfg: EncoderConfig, model=None, tokenizer=None):
        import torch

        self.cfg = cfg
        self.torch = torch
        if cfg.device:
            self.device = torch.device(cfg.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is None or tokenizer is None:
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name)
            model_config = AutoConfig.from_pretrained(cfg.model_name, output_hidden_states=True)
            model = model or AutoModel.from_pretrained(cfg.model_name, config=model_config)
        self.tokenizer = tokenizer
        self.model = model
        # Some checkpoints default to not returning hidden states; insist.
        self.model.config.output_hidden_states = True
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        if cfg.half_precision and self.device.type == "cuda":
            self.model.half()

        self.hidden_size = int(self.model.config.hidden_size)
        self.num_layers = int(self.model.config.num_hidden_layers)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        outputs: List[np.ndarray] = []
        for start in range(0, len(texts), self.cfg.batch_size):
            batch = [str(t) for t in texts[start : start + self.cfg.batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden_states = self.model(**encoded, output_hidden_states=True).hidden_states
            if hidden_states is None:
                raise RuntimeError(
                    f"{self.cfg.model_name} returned no hidden states; the study needs an "
                    "encoder that exposes every layer"
                )
            mask = encoded["attention_mask"]
            pooled = torch.stack(
                [_pool(layer, mask, self.cfg.pooling) for layer in hidden_states], dim=0
            )
            outputs.append(pooled.float().cpu().numpy())
        if not outputs:
            return np.zeros((self.num_layers + 1, 0, self.hidden_size), dtype=np.float32)
        return np.concatenate(outputs, axis=1)


class SyntheticEncoder(BaseEncoder):
    """Offline stand-in with a deterministic, hand-built layer profile.

    Every token maps to a fixed pseudo-random vector via a hash, so the same
    text always yields the same representation and no state is downloaded.
    The synthetic corpus marks its tokens (see
    ``layerprobe.data._load_synthetic_language``), which lets this encoder
    build each layer as a mixture of a *shared* emotion component
    (``e{k}`` markers) and a *language* component (the ``L{language}`` tag),
    over a bed of filler noise.  The mixing weight varies with depth to mimic
    the U-shaped language-neutrality curve reported for multilingual
    encoders: language identity dominates at the bottom, is suppressed in the
    middle, and partly returns at the top.
    """

    #: How much of the pooled vector is filler that carries no signal.
    FILLER_WEIGHT = 0.35

    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        self.hidden_size = cfg.synthetic_hidden_size
        self.num_layers = cfg.synthetic_num_layers
        self._cache: dict[str, np.ndarray] = {}
        # A fixed per-layer rotation stands in for the transformer's
        # per-layer feature basis, so that no two layers are collinear.
        self._rotations = np.stack(
            [
                np.random.default_rng(1000 + layer)
                .normal(size=(self.hidden_size, self.hidden_size))
                .astype(np.float32)
                / np.sqrt(self.hidden_size)
                for layer in range(self.num_layers + 1)
            ]
        )

    def _token_vector(self, token: str) -> np.ndarray:
        cached = self._cache.get(token)
        if cached is None:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "little"))
            cached = rng.normal(size=self.hidden_size).astype(np.float32)
            self._cache[token] = cached
        return cached

    def _language_weight(self, layer: int) -> float:
        """How strongly language-specific tokens colour a given layer."""

        depth = layer / max(1, self.num_layers)
        # 1.0 at the embeddings, a minimum around two thirds depth, rising again.
        return float(0.15 + 0.85 * (2.0 * depth - 1.3) ** 2)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        n_layers = self.num_layers + 1
        out = np.zeros((n_layers, len(texts), self.hidden_size), dtype=np.float32)
        zero = np.zeros(self.hidden_size, np.float32)
        for i, text in enumerate(texts):
            tokens = str(text).split() or ["<empty>"]

            def _mean(chosen: List[str]) -> np.ndarray:
                if not chosen:
                    return zero
                return np.mean([self._token_vector(t) for t in chosen], axis=0)

            # emotion markers (shared across languages), the language tag
            # (constant within a language), and per-language filler noise
            shared_vec = _mean([t for t in tokens if t.startswith("e")])
            lang_vec = _mean([t for t in tokens if t.startswith("L")])
            filler_vec = _mean([t for t in tokens if not t.startswith(("e", "L"))])
            # Noise is seeded from the text, so the same sentence always gets
            # the same representation and the feature cache stays valid.
            digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
            noise_rng = np.random.default_rng(int.from_bytes(digest, "little"))
            noise = noise_rng.normal(
                scale=self.cfg.synthetic_noise, size=(n_layers, self.hidden_size)
            ).astype(np.float32)
            for layer in range(n_layers):
                w = self._language_weight(layer)
                mixed = (1.0 - w) * shared_vec + w * lang_vec + self.FILLER_WEIGHT * filler_vec
                out[layer, i] = mixed @ self._rotations[layer] + noise[layer]
        return out


def build_encoder(cfg: EncoderConfig) -> BaseEncoder:
    """Instantiate the encoder named by the config."""

    if cfg.model_name == "synthetic":
        return SyntheticEncoder(cfg)
    return HFEncoder(cfg)


def select_layers(features: np.ndarray, layers: Optional[Iterable[int]]) -> np.ndarray:
    """Keep only the requested layers of a ``(n_layers, n, d)`` array."""

    if layers is None:
        return features
    idx = list(layers)
    return features[idx]
